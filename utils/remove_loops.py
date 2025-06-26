import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def remove_loop(roadnet, prefix, calculate_od=True):
    rou_file = os.path.join(roadnet, prefix + ".rou.xml")
    net_file = os.path.join(roadnet, prefix + ".net.xml")

    rou_tree = ET.parse(rou_file)
    net_tree = ET.parse(net_file)

    rou_root = rou_tree.getroot()
    net_root = net_tree.getroot()

    # id: (from, to)
    edges = {}
    od_dic = {}
    for edge in net_root.findall('edge'):
        att = edge.attrib
        if 'from' in att:
            edges[att['id']] = (att['from'], att['to'])

    for child in rou_root.findall('vehicle'):
        if child.tag == 'vehicle':
            
            # get route of vehicle
            child.attrib["departLane"] = "best"
            rou = child[0].attrib['edges']
            rou_list = rou.split(' ')
            start_edge, end_edge = rou_list[0], rou_list[-1]

            # loops
            if edges[start_edge][0] == edges[end_edge][1]: 
                rou_root.remove(child)
            elif is_single and len(rou_list) > 2:
                rou_root.remove(child)
            else:
                if calculate_od:
                    od_str = edges[start_edge][0] + " " + edges[end_edge][1]
                    od_dic[od_str] = od_dic.get(od_str, 0) + 1
    
    # root_str = ET.tostring(rou_root, 'utf-8')

    # root_str = minidom.parseString(root_str)

    # root_str = root_str.toprettyxml(indent="\t", encoding='utf-8')
    # root_str = root_str.toprettyxml(encoding='utf-8')
    with open(os.path.join(roadnet, file_prefix + ".rou.xml"), 'wb+') as f:
        rou_tree.write(f, encoding='utf-8')
    return od_dic

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data convert')
    parser.add_argument('roadnet', type=str, help='roadnet')
    parser.add_argument('--is_single', action='store_false', help='is single inter')
    args = parser.parse_args()

    roadnet = args.roadnet

    # roadnet = "../data/syn_1x1_bin"

    file_prefix = "data" 
    # get cityflow roadnet file 
    is_single = args.is_single

    od = remove_loop(roadnet, file_prefix)

    print(od)

    res = sum(od.values())

    print("total vehicle: {}".format(res))

    # python utils/randomTrips.py -n data/syn_1x1_bin/data.net.xml -b 0 -e 3600 -p 1.1 --binomial 3 --random-depart --additional-file data/syn_1x1_bin/data.add.xml --trip-attributes="type=\"vtype0\" departLane=\"best\"" --remove-loops -o data/syn_1x1_bin/data.trip.xml -r data/syn_1x1_bin/data.rou.xml 
 