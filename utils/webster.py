#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用法示例:
    python utils/webster.py MyNet

脚本功能:
    1. 读取 data/MyNet/data.net.xml 与 data/MyNet/data.rou.xml
    2. 在 intersection_1_1 处，根据 <connection dir="l/s/r"> 信息，
       解析每辆车实际是左转/直行/右转，且区分东西(EW)/南北(NS)方向。
    3. 统计 (EW,s)/(NS,s)/(EW,l)/(NS,l) 的车流量，用简化 Webster 法计算
       4 相位(无黄灯) 的绿灯时长 [g0,g1,g2,g3]。
    4. 修改原 net.xml 前 4 个 <phase> 的时长 + state，
       输出新的 data/MyNet/data_new.net.xml：
       - phase0 => 南北直行
       - phase1 => 南北左转
       - phase2 => 东西直行
       - phase3 => 东西左转
"""

import sys
import os
import xml.etree.ElementTree as ET

########################
# A. 解析 net.xml 中 <connection>
########################
def parse_connections_from_net(net_file, tls_id="intersection_1_1"):
    """
    读取 net.xml, 返回与 tls_id(如 "intersection_1_1") 相关的转向映射:
      connection_dict[(fromEdge, toEdge)] = 'l'/'s'/'r' (dir属性).
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    connection_dict = {}
    for conn in root.findall('connection'):
        frm = conn.get('from')  # fromEdge
        to  = conn.get('to')    # toEdge
        dir_attr = conn.get('dir')  # 'l','s','r'
        tl_attr = conn.get('tl')    # traffic light ID

        if tl_attr == tls_id and frm and to and dir_attr:
            connection_dict[(frm, to)] = dir_attr.lower()
    return connection_dict


########################
# B. 解析 rou.xml: 找到每辆车在 intersection_1_1 的实际转向
########################
def parse_turns_from_rou(rou_file, connection_dict):
    """
    rou.xml 每辆车 <vehicle><route edges="e1 e2 ...">
    找相邻 (fE, tE) 是否在 connection_dict 中，从而识别 'l'/'s'/'r'。
    再按 edge_id 判定是 EW 还是 NS，把 (direction, turn) 累加到统计中。
    """
    def judge_EW_NS(edge_id):
        """
        这里示例:
          若 '0_1_0'/'2_1_2' => EW
          若 '1_0_1'/'1_2_3' => NS
        其余不计
        """
        if "0_1_0" in edge_id or "2_1_2" in edge_id:
            return "EW"
        elif "1_0_1" in edge_id or "1_2_3" in edge_id:
            return "NS"
        else:
            return None

    turn_count = {}

    tree = ET.parse(rou_file)
    root = tree.getroot()

    for veh in root.findall('vehicle'):
        route_elem = veh.find('route')
        if route_elem is None:
            continue
        edges = route_elem.get('edges').split()

        # 遍历相邻对 (fE, tE) 看是否在 connection_dict
        for i in range(len(edges) - 1):
            fE = edges[i]
            tE = edges[i + 1]
            if (fE, tE) in connection_dict:
                turn_dir = connection_dict[(fE, tE)]  # 'l'/'s'/'r'
                direction = judge_EW_NS(fE)           # 'EW'/'NS'
                if direction is not None and turn_dir in ['l','s']:
                    turn_count[(direction, turn_dir)] = turn_count.get((direction, turn_dir), 0) + 1

    return turn_count


########################
# C. 汇总为 4 相位流量
########################
def aggregate_flows_to_phases(turn_count):
    """
    这里固定: 0=(EW,s), 1=(NS,s), 2=(EW,l), 3=(NS,l).
    返回 [q0, q1, q2, q3].
    """
    q0 = turn_count.get(('EW','s'), 0)  # EW直行
    q1 = turn_count.get(('NS','s'), 0)  # NS直行
    q2 = turn_count.get(('EW','l'), 0)  # EW左转
    q3 = turn_count.get(('NS','l'), 0)  # NS左转
    return [q0, q1, q2, q3]


########################
# D. Webster 计算(无黄灯)
########################
def calc_webster_4phases(phase_counts,
                         sim_end_time=3600,
                         saturation_flow=3600,
                         lost_time_per_phase=3,
                         min_green_time=10):
    """
    phase_counts: [q0,q1,q2,q3] => [EW直,NS直,EW左,NS左].
    sim_end_time: 统计时段(s)
    saturation_flow: 每相位(或每车道)的饱和流率 veh/h
    lost_time_per_phase: 每相位损失时间(秒)
    min_green_time: 最小绿灯(秒)

    返回 [g0,g1,g2,g3] 纯绿灯
    """
    # 1) 折算成 veh/h
    flows_h = []
    for count in phase_counts:
        q_h = count * (3600.0 / sim_end_time)
        flows_h.append(q_h)

    y_list = [q / saturation_flow for q in flows_h]
    Y = sum(y_list)
    n = 4
    L = n * lost_time_per_phase

    if Y >= 1.0:
        raise ValueError("流量比 Y>=1，路口已饱和或超饱和。")

    # Webster
    C0 = (1.5 * L + 5.0) / (1.0 - Y)
    cycle_approx = round(C0)

    effective_green = cycle_approx - L
    if effective_green < 0:
        raise ValueError("无效周期, (C0 - L)<0")

    green_times = []
    for y_i in y_list:
        g_i = (y_i / Y) * effective_green
        g_i_int = max(min_green_time, round(g_i))
        green_times.append(g_i_int)

    return green_times


########################
# E. 更新 net.xml: 重写 4 相位 (无黄灯)
########################
def update_netxml_no_yellow(net_in, net_out, g0, g1, g2, g3):
    """
    将 intersection_1_1 的前4相位改成:
      相位0 => 南北直行
      相位1 => 南北左转
      相位2 => 东西直行
      相位3 => 东西左转
    并设置 duration= g0,g1,g2,g3.
    """
    # 根据题意：四相位的 state:
    #  0 => 南北直行: "rrGGrrrrrrGGrrrr"
    #  1 => 南北左转: "rrrrrrGGrrrrrrGG"
    #  2 => 东西直行: "GGrrrrrrGGrrrrrr"
    #  3 => 东西左转: "rrrrGGrrrrrrGGrr"
    states_4 = [
        "rrGGrrrrrrGGrrrr",  # 0: NS straight
        "rrrrrrGGrrrrrrGG",  # 1: NS left
        "GGrrrrrrGGrrrrrr",  # 2: EW straight
        "rrrrGGrrrrrrGGrr"   # 3: EW left
    ]
    durations = [g0, g1, g2, g3]

    tree = ET.parse(net_in)
    root = tree.getroot()

    for tls in root.findall('tlLogic'):
        if tls.get('id') == 'intersection_1_1':
            phases = tls.findall('phase')
            if len(phases) < 4:
                raise ValueError("intersection_1_1 的 <phase> 数量不足4个，请检查 net.xml。")

            # 修改前4个 <phase> 的 state/duration
            for i in range(4):
                ph_elem = phases[i]
                ph_elem.set('state', states_4[i])
                ph_elem.set('duration', str(durations[i]))

    tree.write(net_out, encoding='UTF-8', xml_declaration=True)
    print(f"[update_netxml_no_yellow] 已写入新的 net 文件 => {net_out}")


########################
# 主函数: 命令行执行
########################
def main():
    """
    用法:
        python utils/webster.py <NetName>

    例如:
        python utils/webster.py MyNet

    将读取:
        data/<NetName>/data.net.xml
        data/<NetName>/data.rou.xml
    输出新的:
        data/<NetName>/data_new.net.xml
    """
    if len(sys.argv) < 2:
        print("用法: python utils/webster.py <NetName>")
        sys.exit(1)

    net_name = sys.argv[1]
    base_dir = os.path.join("data", net_name)
    net_in  = os.path.join(base_dir, "data.net.xml")
    rou_in  = os.path.join(base_dir, "data.rou.xml")
    net_out = os.path.join(base_dir, "data_new.net.xml")

    # A. 解析 net.xml => connection_dict
    connection_dict = parse_connections_from_net(net_in, tls_id="intersection_1_1")

    # B. 解析 rou.xml => turn_count
    turn_count = parse_turns_from_rou(rou_in, connection_dict)

    # C. 汇总 => [q0, q1, q2, q3] (EW直,NS直,EW左,NS左)
    phase_counts = aggregate_flows_to_phases(turn_count)
    print(f"四相位车辆统计: {phase_counts}")

    # D. Webster计算(无黄灯)
    #   如果你的路口有2条同向车道，可把 saturation_flow=3600 之类
    green_times = calc_webster_4phases(
        phase_counts,
        sim_end_time=3600,
        saturation_flow=3600,  # 或 3600, 自行调
        lost_time_per_phase=5,
        min_green_time=0
    )

    print(f"Webster计算得到的绿灯时长[NS直,NS左,EW直,EW左]: {green_times}")

    # E. 更新 net.xml (相位顺序: 0=NS直,1=NS左,2=EW直,3=EW左)
    update_netxml_no_yellow(net_in, net_out, *green_times)

    print("\nDone! 现在可使用 data_new.net.xml 进行仿真。")


if __name__ == "__main__":
    main()
