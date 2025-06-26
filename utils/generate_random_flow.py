import json
import subprocess
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

def rou_xml_process(data_dir, file_prefix, src_edg, dst_edg):
    """get sumo src.xml dst.xml file

    """
    src_root = ET.Element('edgedata') 
    
    num_interval = int(3600 / interval)

    for i in range(num_interval):
        interval_node = ET.Element("interval")
        interval_node.attrib = {"begin": str(i * interval), "end":str(min((i + 1)*interval, 3600))} 


        for e in src_edg:
            edge_node = ET.Element("edge")
            prob = round(np.random.uniform(min_prob, max_prob), 2)
            edge_node.attrib = {"id": e, "value": str(prob)}
            interval_node.append(edge_node)
        src_root.append(interval_node)

    src_root_str = xml_postprocess(src_root)

    with open(os.path.join(data_dir, file_prefix + "{}_{}_{}_{}.src.xml".format(min_prob, max_prob, interval, random_seed)), 'wb+') as f:
        f.write(src_root_str)

    dst_root = ET.Element('edgedata') 
    

    for i in range(num_interval):
        interval_node = ET.Element("interval")
        interval_node.attrib = {"begin": str(i * interval), "end":str(min((i + 1)*interval, 3600))} 


        for e in dst_edg:
            edge_node = ET.Element("edge")
            prob = round(np.random.uniform(min_prob, max_prob), 2)
            edge_node.attrib = {"id": e, "value": str(prob)}
            interval_node.append(edge_node)
        dst_root.append(interval_node)

    dst_root_str = xml_postprocess(dst_root)

    with open(os.path.join(data_dir, file_prefix + "{}_{}_{}_{}.dst.xml".format(min_prob, max_prob, interval, random_seed)), 'wb+') as f:
        f.write(dst_root_str)

def xml_postprocess(root):
    """process Element tree, get well-formatted string

    :param root: root node of xml tree
    :return: string of xml file
    """
    root_str = ET.tostring(root, 'utf-8')
    root_str = minidom.parseString(root_str)

    root_str = root_str.toprettyxml(indent="\t", encoding='utf-8')

    return root_str

def get_src_dst_edg(data_dir, file_prefix):
    con_file = os.path.join(data_dir, file_prefix + ".con.xml")

    con_tree = ET.parse(con_file)
    con_root = con_tree.getroot()

    origin_src_edg = []
    origin_dst_edg = []

    for c in con_root.findall('connection'):
        att = c.attrib
        origin_src_edg.append(att["from"])
        origin_dst_edg.append(att["to"])
    
    src_edg = sorted(list(set(origin_src_edg) - set(origin_dst_edg)))
    dst_edg = sorted(list(set(origin_dst_edg) - set(origin_src_edg)))

    return src_edg, dst_edg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data convert')
    parser.add_argument('roadnet', type=str, help='roadnet')
    parser.add_argument('flow_prob', type=float, help='flow prob')
    parser.add_argument('flow_bino', type=int, help='flow bino')
    parser.add_argument('--interval', type=int, default=600, help='src dst interval')
    parser.add_argument('--min_prob', type=float, default=0.5, help='min prob')
    parser.add_argument('--max_prob', type=float, default=1.0, help='max prob')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--src_prefix', type=str, default=None, help='src prefix')

    args = parser.parse_args()

    min_prob, max_prob = args.min_prob, args.max_prob
    interval = args.interval
    random_seed = args.random_seed
    prob = args.flow_prob
    bino = args.flow_bino

    np.random.seed(random_seed)
    roadnet_dir = args.roadnet
    # roadnet_dir = os.path.join("../data", roadnet)
    file_prefix = "data" 

    src_edg, dst_edg = get_src_dst_edg(roadnet_dir, file_prefix)
    rou_xml_process(roadnet_dir, file_prefix, src_edg, dst_edg)

    weight_prefix = args.src_prefix or "data{}_{}_{}_{}".format(min_prob, max_prob, interval, random_seed)

    weight_prefix = os.path.join(roadnet_dir, weight_prefix)

    cmd = 'python utils/randomTrips.py -n {}/data.net.xml -t "type=\\"vtype0\\"" -a {}/data.add.xml -s {} -p {} --binomial {} --remove-loops --random-depart --random-departpos --weights-prefix {} -o {}/data.trips.xml -r {}/data.rou.xml'.format(roadnet_dir, roadnet_dir, random_seed, prob, bino, weight_prefix, roadnet_dir, roadnet_dir)



    cmd_result = subprocess.run(cmd, shell=True, check=True)

    # 检查返回码
    if cmd_result.returncode == 0:
        print("Command executed successfully")
    else:
        print("Command failed")
    print(cmd)
    # os.popen(cmd)

    remove_loop_cmd = "python utils/remove_loops.py {}".format(roadnet_dir)
    # print(remove_loop_cmd)

    final_cmd = cmd + " && " + remove_loop_cmd
    # print(final_cmd)
    # os.popen(remove_loop_cmd)
