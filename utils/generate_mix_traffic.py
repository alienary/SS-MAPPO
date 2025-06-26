import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from copy import deepcopy

def xml_postprocess(root):
    """process Element tree, get well-formatted string

    :param root: root node of xml tree
    :return: string of xml file
    """
    root_str = ET.tostring(root, 'utf-8')
    root_str = minidom.parseString(root_str)

    root_str = root_str.toprettyxml(indent="\t", encoding='utf-8')

    return root_str


def generate_mix_flow(roadnet, penestrate_rate, random_seed):
    """convert flow to mix traffic flow with desired penestrate rate.
    In converted flow, vtype1 is CV, vtype0 is HDV. The penestrate rate is the ratio of CVs in total vehicles.

    Args:
        roadnet (str): roadnet dir
        penestrate_rate (float): penestrate rate
        random_seed (int): random seed
    """

    cmd = f'cp -rf {roadnet}/. {roadnet}_{penestrate_rate}_{random_seed}'

    os.system(cmd)

    new_roadnet = f'{roadnet}_{penestrate_rate}_{random_seed}'

    np.random.seed(random_seed)
    total_vehicle = 0
    cv_vehicle = 0

    file_prefix = "data"

    rou_file = os.path.join(new_roadnet, f'{file_prefix}.rou.xml')
    rou_tree = ET.parse(rou_file)
    rou_root = rou_tree.getroot()

    vtype_node = rou_root.find('vType')
    # define vType
    new_vtype_node = ET.Element("vType")
    new_attrib = deepcopy(vtype_node.attrib)
    new_attrib['id'] = 'vtype1'
    new_vtype_node.attrib = new_attrib
    rou_root.insert(1, new_vtype_node)

    for child in rou_root.findall('vehicle'):
        if child.tag == 'vehicle':
            
            # get route of vehicle
            p = np.random.random()
            if p < penestrate_rate:
                child.attrib["type"] = "vtype1"
                cv_vehicle += 1
            total_vehicle += 1
    
    # root_str = xml_postprocess(rou_root)
    # with open(f'{new_roadnet}/{file_prefix}.rou.xml', 'wb+') as f:
    #     f.write(root_str)

    rou_tree.write(f'{new_roadnet}/{file_prefix}.rou.xml', encoding='utf-8', xml_declaration=True, method='xml')
    

    print(f'{cv_vehicle} CVs in {total_vehicle} vehicles, penestrate rate: {cv_vehicle / total_vehicle}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data convert')
    parser.add_argument('roadnet', type=str, help='roadnet')
    parser.add_argument('prob', type=float, help='penestration rate')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')   
    args = parser.parse_args()

    roadnet = args.roadnet
    penestrate_rate = args.prob
    random_seed = args.random_seed


    generate_mix_flow(roadnet, penestrate_rate, random_seed)

