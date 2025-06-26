"""
convert a cityflow roadnet file to sumo roadnet file
"""
from copy import deepcopy
import json
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 4 phase

# 2 lane(left and straight)
# 2x2x4 connections
phase_2 = [
    		"GGrrrrrrGGrrrrrr",
		    "rrGGrrrrrrGGrrrr",
		    "rrrrGGrrrrrrGGrr",
		    "rrrrrrGGrrrrrrGG"
]

# 3 lane(right, straight and left)
# 3x3x4 connections
phase_3 = [
		"GGGGGGrrrGGGrrrrrrGGGGGGrrrGGGrrrrrr",
		"GGGrrrGGGGGGrrrrrrGGGrrrGGGGGGrrrrrr",
		"GGGrrrrrrGGGGGGrrrGGGrrrrrrGGGGGGrrr",
		"GGGrrrrrrGGGrrrGGGGGGrrrrrrGGGrrrGGG",
]
def node_xml_process(r, data_dir, file_prefix):
    """get sumo nod.xml file

    :param r: dict, cityflow roadnet dict
    :param data_dir: str, save dir
    :param file_prefix: str, save file prefix
    """
    root = ET.Element('nodes') 

    root.attrib = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance" ,"xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"}

    intersetcions = r['intersections']

    for inter in intersetcions:
        att = dict()
        att["id"] = inter["id"]
        att["x"] = str(inter["point"]["x"])
        att["y"] = str(inter["point"]["y"])
        if inter["virtual"]:
            att["type"] = "priority"
        else:
            att["type"] = "traffic_light"
        
        node = ET.Element("node")
        node.attrib = att
        
        root.append(node)

    root_str = xml_postprocess(root)

    with open(os.path.join(data_dir, file_prefix + ".nod.xml"), 'wb+') as f:
        f.write(root_str)

def edge_xml_process(r, data_dir, file_prefix):
    """get sumo edg.xml file

    :param r: dict, cityflow roadnet dict
    :param data_dir: str, save dir
    :param file_prefix: str, save file prefix
    """
    root = ET.Element('edges') 

    root.attrib = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance" ,"xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"}

    roads = r['roads']

    for road in roads:
        att = dict()
        att["id"] = road["id"]
        att["from"] = road["startIntersection"]
        att["to"] = road["endIntersection"]
        att["priority"] = "-1"
        att["numLanes"] = str(len(road["lanes"]))
        max_speed = -1
        for lane in road["lanes"]:
            max_speed = max(max_speed, lane["maxSpeed"])
        
        att["speed"] = str(max_speed)

        # TODO: points

        node = ET.Element("edge")
        node.attrib = att
        root.append(node)
    
    root_str = xml_postprocess(root)


    with open(os.path.join(data_dir, file_prefix + ".edg.xml"), 'wb+') as f:
        f.write(root_str)

def conn_xml_process(r, data_dir, file_prefix):
    """get sumo con.xml file

    :param r: dict, cityflow roadnet dict
    :param data_dir: str, save dir
    :param file_prefix: str, save file prefix
    """
    root = ET.Element('connections') 

    root.attrib = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance" ,"xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/connections_file.xsd"}

    root_tls = ET.Element('tlLogics')
    root_tls.attrib = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance" ,"xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/tllogic_file.xsd"}

    intersections = r["intersections"]
    roads = r["roads"]

    roads_idx = {road["id"]: i for (i, road) in enumerate(roads)}

    for inter in intersections:
        if inter["virtual"]:
            continue
        
        node_tls = ET.Element("tlLogic")
        node_tls.attrib = {
            "id": inter["id"],
            "offset": "0",
            "programID": "0",
            "type": "static"
        }

        conn_cnt = 0
        conn_cnt_list = []
        for road_link in inter["roadLinks"]:

            att = dict()
            att["from"] = road_link["startRoad"]
            att["to"] = road_link["endRoad"]

            start_road = roads[roads_idx[att["from"]]]
            start_road_lane = len(start_road["lanes"])
            end_road = roads[roads_idx[att["to"]]]
            end_road_lane = len(end_road["lanes"])
        
            for lane_link in road_link["laneLinks"]:
                att_lane = deepcopy(att)

                # lane index in cityflow and sumo is opposite
                # sumo: 0 rightmost lane, num_lane - 1 leftmost lane
                # cityflow: 0 inner lane(left), num_lane -1 outer lane(right)
                att_lane["fromLane"] = str(start_road_lane - 1 - lane_link["startLaneIndex"])
                att_lane["toLane"] = str(end_road_lane - 1 - lane_link["endLaneIndex"])
                # TODO: points
                node = ET.Element("connection")
                node.attrib = att_lane
                root.append(node)
                
                conn_cnt += 1
            
            conn_cnt_list.append(conn_cnt)

        # traffic light

        # currently only for 4 phase and four leg intersection
        num_phase = 4
        for i in range(num_phase):

            # start_idx = conn_cnt_list[i-1] if i > 0 else 0
            # end_idx = conn_cnt_list[i]
            # p_state[start_idx: end_idx] = 'G' * (end_idx - start_idx)
            
            # start_idx2 = conn_cnt_list[i+3]
            # end_idx2 = conn_cnt_list[i+4]
            # p_state[start_idx2: end_idx2] = 'G' * (end_idx - start_idx)

            if conn_cnt == 36:
                p_state = phase_3[i]
            else:
                p_state = phase_2[i]

            node_phase = ET.Element("phase")
            node_phase.attrib = {
                "duration": "30",
                "state": "".join(p_state)
            }
            node_tls.append(node_phase)
        
        root_tls.append(node_tls)


    root_str = xml_postprocess(root)
    root_tls_str = xml_postprocess(root_tls)


    with open(os.path.join(data_dir, file_prefix + ".con.xml"), 'wb+') as f:
        f.write(root_str)

    with open(os.path.join(data_dir, file_prefix + ".tll.xml"), 'wb+') as f:
        f.write(root_tls_str)

def cfg_xml_process(data_dir, file_prefix):
    """get sumo config file

    :param data_dir: str, save dir
    :param file_prefix: str, save file prefix
    """
    root = ET.Element('configuration') 

    node = ET.Element('input')
    node_net = ET.Element('net-file')
    node_net.attrib = {
        "value": "{}.net.xml".format(file_prefix)
    }
    node.append(node_net)
    node_route = ET.Element('route-files')
    node_route.attrib = {
        "value": "{}.rou.xml".format(file_prefix)
    }
    node.append(node_route)

    root.append(node)

    node = ET.Element("time")

    node_begin = ET.Element("begin")
    node_begin.attrib = {
        "value": "0"
    }
    node.append(node_begin)

    
    node_end = ET.Element("end")
    node_end.attrib = {
        "value": "4000"
    }
    node.append(node_end)

    root.append(node)
    root_str = xml_postprocess(root)


    with open(os.path.join(data_dir, file_prefix + ".sumocfg"), 'wb+') as f:
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
    with open(os.path.join(roadnet_dir, "roadnet.json")) as f:
        r = json.load(f)

    node_xml_process(r, roadnet_dir, file_prefix)
    edge_xml_process(r, roadnet_dir, file_prefix)
    conn_xml_process(r, roadnet_dir, file_prefix)

    dir = os.path.join(roadnet_dir, file_prefix)
    netconvert_cmd = "netconvert -n {}.nod.xml -e {}.edg.xml -x {}.con.xml -i {}.tll.xml -o {}.net.xml".format(dir, dir, dir, dir, dir)
    os.popen(netconvert_cmd)

    cfg_xml_process(roadnet_dir, file_prefix)