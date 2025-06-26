#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修正后的示例：使用 Yu & Long (2022) 完整分段油耗模型，并正确处理 kN 和 ml/kJ 的单位。
命令行用法：
  python my_ecodrive_fixed.py /path/trajectory_data.csv
"""

import sys
import csv
from collections import defaultdict


def calculate_stop_and_go(speed_list, theta_go=2, theta_stop=2, speed_stop=0.1):
    """
    计算 'stop-and-go' 次数
    ---------------------------------------------------
    speed_list: 该车各时刻速度, 单位 m/s
    theta_go:   前 theta_go 个连续时刻速度 > speed_stop
    theta_stop: 后 theta_stop 个连续时刻速度 < speed_stop
    speed_stop: 判断停的速度阈值
    """
    stop_count = 0
    for i in range(theta_go, len(speed_list) - theta_stop):
        if (all(s > speed_stop for s in speed_list[i - theta_go : i]) and
            all(s < speed_stop for s in speed_list[i : i + theta_stop])):
            stop_count += 1
    return stop_count


def yulong_fuel_rate(v, a,
                     alpha=0.444,   # (ml/s),  表 I
                     beta1=0.09,   # (ml/kJ), 表 I
                     beta2=0.04,   # (ml/kJ), 表 I
                     b1_kN=0.333,  # (kN),    表 I
                     b2_kN=0.00108,# (kN),    表 I
                     M=1200.0):
    """
    Yu & Long(2022) 论文中 Table I 参数与公式(3)的分段油耗模型(单位严格对应)：
    ---------------------------------------------------
    若  a < -(b1 + b2 v^2)/M:
        F = alpha
    若 -(b1 + b2 v^2)/M <= a < 0:
        F = alpha + beta1 * v * (b1 + b2 v^2 + M a)_{(kJ/s)}
    若 a >= 0:
        F = alpha + beta1 * v*(b1 + b2 v^2 + M a)_{(kJ/s)}
                   + beta2 * M * v * a^2_{(kJ/s)}

    - b1,b2 原是 kN => 先 *1000 转成 N
    - (b1 + b2 v^2 + M a)*v => J/s => 除以1000 => kJ/s => 再乘 beta1 => ml/s
    - 同理对 beta2 项也要在“kJ/s”维度下计算
    - alpha(ml/s) 为怠速油耗
    """
    # 1) 先把 kN -> N
    b1 = b1_kN * 1000.0  # (N)
    b2 = b2_kN * 1000.0  # (N/(m^2/s^2))

    # 2) 判断三个加速度区间
    threshold = -(b1 + b2*(v**2)) / M  # (m/s^2)

    if a < threshold:
        # 完全制动阶段
        return alpha  # ml/s

    # 若处于中间区间 或 a>=0，都要先算 “功率”
    # R = (b1 + b2 v^2 + M a)  (单位 N)
    # Power(J/s) = R * v
    # => kJ/s = (R*v)/1000
    R_val = (b1 + b2 * v**2 + M*a)
    power_kJs = (R_val * v) / 1000.0  # (kJ/s)

    if a < 0:
        # -(b1 + b2 v^2)/M <= a < 0
        # F = alpha + beta1 * v*(b1 + b2 v^2 + M a)
        #    => alpha + beta1 * [power_kJs]
        fuel_rate = alpha + beta1 * power_kJs
        if fuel_rate < 0:  # 公式里还有 max(...,0)，以防出现负值
            fuel_rate = alpha
        return fuel_rate

    else:
        # a >= 0
        # F = alpha + beta1*v*(b1 + b2 v^2 + M a) + beta2*M*v*a^2
        # 后者同样要转成 kJ/s => (M*a^2)*v => N*(m/s^2)^2 * m/s = N*m^3/s^3 ? 
        # 直接沿用论文给出的简写: beta2 * M * v * a^2(同维度kJ?). 
        # 最简做法: 先算 main = alpha + beta1*power_kJs
        # 再加 beta2*M*v*a^2 => 仍需把(M*v*a^2) * ? => J/s => /1000 => kJ/s => * beta2
        main_rate = alpha + beta1 * power_kJs

        # 额外项: (beta2 * M * v * a^2) => 这也要先变成 kJ/s
        # E = M * a^2 * v => kg * (m/s^2)^2 * m/s => kg*m^3/s^4 => 1 kg*m/s^2=1N
        # => N*(m/s^2)* => 还要再 /1000
        extra_kJs = (M * a**2 * v) / 1000.0
        fuel_rate = main_rate + beta2 * extra_kJs
        if fuel_rate < 0:
            fuel_rate = alpha
        return fuel_rate


def analyze_data(csv_path, 
                 wait_speed_thresh=0.1,  # m/s
                 free_flow_speed=15.0):  # m/s
    """
    统计指标: 
      avgwaitingtime (s), avgdelay (s),
      avgfuelconsumption (ml), avgstopandgo (次), avgtraveltime (s)
    """
    vehicle_dict = defaultdict(lambda: {
        'steps': [],
        'speeds': [],
        'distances': []
    })
    
    # 1) 读取CSV
    with open(csv_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            vehid = row['vehid']
            step = float(row['step'])     # s
            speed = float(row['speed'])  # m/s
            dist = float(row['distance_to_intersection'])  # m
            vehicle_dict[vehid]['steps'].append(step)
            vehicle_dict[vehid]['speeds'].append(speed)
            vehicle_dict[vehid]['distances'].append(dist)
            
    
    total_wait_time     = 0.0
    total_delay         = 0.0
    total_fuel          = 0.0  # ml
    total_stop_and_go   = 0.0
    total_travel_time   = 0.0
    
    vehicle_count = len(vehicle_dict)
    if vehicle_count == 0:
        print("No vehicle data found.")
        return {}
    
    for vehid, data in vehicle_dict.items():
        steps     = data['steps']
        speeds    = data['speeds']
        distances = data['distances']

        if not steps:
            vehicle_count -= 1
            continue
        
        # 排序
        idx_sorted = sorted(range(len(steps)), key=lambda i: steps[i])
        steps     = [steps[i] for i in idx_sorted]
        speeds    = [speeds[i] for i in idx_sorted]
        distances = [distances[i] for i in idx_sorted]
        
        # 行程时间
        travel_time = steps[-1] - steps[0]

        # 等待时间(若step=1s)
        wait_time = sum(1 for s in speeds if s < wait_speed_thresh)
        
        # 延误 = travel_time - free_flow_time
        dist_travel = abs(distances[0] - distances[-1])
        if dist_travel > 1e-3:
            free_flow_time = dist_travel / free_flow_speed
            delay = travel_time - free_flow_time
        else:
            delay = 0.0
        
        # 逐时刻油耗 (假定相邻 step≈1s)
        fuel_sum = 0.0
        for i in range(len(speeds)):
            v_now = speeds[i]
            if i == 0:
                v_prev = v_now
            else:
                v_prev = speeds[i-1]
            # 简化加速度
            a_now = v_now - v_prev

            # 调用修正后分段函数
            fc_rate = yulong_fuel_rate(v_now, a_now)  # ml/s
            fuel_sum += fc_rate  # step=1s => 累加
        
        # stop-and-go
        sng_count = calculate_stop_and_go(speeds)
        
        total_wait_time   += wait_time
        total_delay       += delay
        total_fuel        += fuel_sum
        total_stop_and_go += sng_count
        total_travel_time += travel_time
    
    if vehicle_count > 0:
        avgwaitingtime     = total_wait_time / vehicle_count
        avgdelay           = total_delay / vehicle_count
        avgfuelconsumption = total_fuel / vehicle_count
        avgstopandgo       = total_stop_and_go / vehicle_count
        avgtraveltime      = total_travel_time / vehicle_count
    else:
        avgwaitingtime     = 0
        avgdelay           = 0
        avgfuelconsumption = 0
        avgstopandgo       = 0
        avgtraveltime      = 0

    return {
        'avgwaitingtime (s)': avgwaitingtime,
        'avgdelay (s)': avgdelay,
        'avgfuelconsumption (ml)': avgfuelconsumption,
        'avgstopandgo (count)': avgstopandgo,
        'avgtraveltime (s)': avgtraveltime
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python my_ecodrive_fixed.py /path/trajectory_data.csv")
        sys.exit(1)
    csv_file_path = sys.argv[1]
    stats = analyze_data(csv_file_path)
    if stats:
        print("---- Yu & Long (2022)修正后油耗统计结果 ----")
        for k,v in stats.items():
            print(f"{k} = {v:.3f}")
