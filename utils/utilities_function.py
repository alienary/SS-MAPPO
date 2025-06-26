import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random
import numpy as np
import pandas as pd
import torch
import os
import pylab as mpl
matplotlib.use('Agg')


agent_name_map = {
    "PhaseLight": "PhaseLight", 
    "PhaseLight4": "w/o phase attention(4)",
    "PhaseLightNomap": "w/o map",
    "PhaseLightNoencoder": "w/o phase encoder",
    "PhaseLight2": "w/o phase attention(2)" 
}
# 设置随机数种子
def set_random_seeds(random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def visualize(result, title, log_dir):
    """
    visualize all_results
    """
    # assert "episode_train_reward" in result[0]
    # colors = ['seagreen', 'magenta', 'cornflowerblue', 'purple',  "brown", "palevioletred", "dimgray", "yellow", "green", "black"]
    colors = ['seagreen', 'magenta', 'cornflowerblue', 'purple',  "brown", "palevioletred", "dimgray", "yellow", "green", "black"]
    plt.figure()
    plt.grid(axis="both")
    # sns.set_style('whitegrid')

    for i, (agent_name, agent_result) in enumerate(result.items()):

        all_result = np.asarray(agent_result)
        if all_result.shape[0] == 0:
            continue

        if all_result.shape[0] == 1: # for SOTL, Maxpressure
            # plt.axhline(all_result[0], label=agent_name, color=colors[i], linestyle='--')
            plt.axhline(all_result[0], label=agent_name, linestyle='--', color = colors[i])
            continue

        x_vals = list(range(all_result.shape[1]))
        mean, std = np.mean(all_result, axis=0), np.std(all_result, axis=0)
        mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
        # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.plot(x_vals, mean, label=agent_name)
        plt.fill_between(x_vals, y1=mean + std, y2=mean - std, alpha=0.2)
        # plt.plot(x_vals, mean, label=agent_name, color=colors[i])
        # plt.fill_between(x_vals, y1=mean + std, y2=mean - std, alpha=0.2, color=colors[i])


    # plt.ylim((-3, -1))
    plt.xlabel("episode")
    # plt.ylabel('Average Travel Time/s')
    plt.ylabel(title)
    plt.legend()
    fig_name = os.path.join(log_dir, f"{title}.png")
    plt.savefig(fig_name, dpi=800)


def metric_calculate(result, agent_name):
    res = {}
    for i, name in enumerate(agent_name):
        agent_result = result[i]
        result_agent = {}
        for k in agent_result[0]:
            all_train_res = [res[k] for res in agent_result]
            all_train_res = np.asarray(all_train_res)
            avg_train_res = np.mean(all_train_res, axis=0)
            result_agent[k] = avg_train_res[0]
        res[name] = result_agent
    return res


def read_traing_result(log_dir):
    episode_reward = {}
    episode_travel_time = {}

    for agent in os.listdir(log_dir):
        agent_log_dir = os.path.join(log_dir, agent)
        result_episode_reward = []
        result_episode_travel_time = []

        if not os.path.isdir(agent_log_dir):
            continue
        cnt = 0
        for root, dirs, files in os.walk(agent_log_dir):
            for file in files:
                if file == "episode_info.csv":
                    cnt += 1
                    res_file = pd.read_csv(os.path.join(root, file)).to_dict('list')
                    train_reward = res_file['episode_reward']
                    train_travel_time = res_file['duration']

                    result_episode_reward.append(train_reward)
                    result_episode_travel_time.append(train_travel_time)

        # if cnt != 5:
        #     print("file number error: {}".format(name))
        name = agent_name_map.get(agent, agent)
        episode_reward[name] = result_episode_reward
        episode_travel_time[name] = result_episode_travel_time

    return episode_reward, episode_travel_time

def read_test_result(log_dir):
    res_waiting_time = {}
    res_throughput = {}
    res_travel_time = {}

    
    for agent in os.listdir(log_dir):
        agent_log_dir = os.path.join(log_dir, agent)
        
        if not os.path.isdir(agent_log_dir):
            continue

        agent_waiting_time = []
        agent_throughput = []
        agent_travel_time = []

        for root, dirs, files in os.walk(agent_log_dir):
            for file in files:
                if file == "out_stat.xml":
                    try:
                        tree = ET.parse(os.path.join(root, file))
                    except:
                        continue
                    else:
                        root = tree.getroot()

                    agent_throughput.append(int(root[1].attrib["inserted"]) - int(root[1].attrib["running"]))
                    agent_waiting_time.append(float(root[6].attrib["waitingTime"]))
                    agent_travel_time.append(float(root[6].attrib["duration"]))
        
        name = agent_name_map.get(agent, agent)
        res_throughput[name] = np.mean(agent_throughput)
        res_waiting_time[name] = np.mean(agent_waiting_time)
        res_travel_time[name] = np.mean(agent_travel_time)

    return {"waiting_time": res_waiting_time, 
            "throughput": res_throughput, 
            "travel_time": res_travel_time}
