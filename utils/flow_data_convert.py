"""
convert a cityflow flow file to a sumo flow file
"""
import json
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

def rou_xml_process(r, data_dir, file_prefix):
    """get sumo nod.xml file

    :param r: dict, cityflow roadnet dict
    :param data_dir: str, save dir
    :param file_prefix: str, save file prefix
    """
    root = ET.Element('routes') 

    root.attrib = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance" ,"xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"}


    # define vType
    vtype_node = ET.Element("vType")
    veh = r[0]["vehicle"]
    veh_att = {}
    veh_att["id"] = "vtype0"
    veh_att["length"] = str(veh["length"])
    veh_att["width"] = str(veh["width"])
    veh_att["minGap"] = str(veh["minGap"])
    veh_att["accel"] = str(veh["usualPosAcc"])
    veh_att["decel"] = str(veh["usualNegAcc"])
    veh_att["maxSpeed"] = str(veh["maxSpeed"])
    vtype_node.attrib = veh_att
    root.append(vtype_node)

    for i, veh in enumerate(r):
        att = dict()
        att["id"] = str(i)
        att["type"] = "vtype0"
        att["depart"] = str(veh["startTime"])
        att["departLane"] = "best"
        
        node = ET.Element("vehicle")
        node.attrib = att

        rou_str = " ".join(veh["route"]) 
        rou_node = ET.Element("route")
        rou_node.attrib = {
            "edges": rou_str
        }

        node.append(rou_node)
        
        root.append(node)

    root_str = xml_postprocess(root)

    with open(os.path.join(data_dir, file_prefix + ".rou.xml"), 'wb+') as f:
        f.write(root_str)


def xml_postprocess(root):
    """process Element tree, get well-formatted string

    :param root: root node of xml tree
    :return: string of xml file
    """
    root_str = ET.tostring(root, 'utf-8')
    root_str = minidom.parseString(root_str)

    root_str = root_str.toprettyxml(indent="\t", encoding='utf-8')

    return root_str


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data convert')
    parser.add_argument('roadnet', type=str, help='roadnet')
    args = parser.parse_args()

    roadnet = args.roadnet
    roadnet_dir = os.path.join("../data", roadnet)
    file_prefix = "data" 

    # get cityflow roadnet file 
    with open(os.path.join(roadnet_dir, "flow.json")) as f:
        r = json.load(f)

    rou_xml_process(r, roadnet_dir, file_prefix)