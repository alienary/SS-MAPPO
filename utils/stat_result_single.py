from utilities_function import visualize, read_traing_result, read_test_result
import pprint
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('log_dir', default="", help='log_dir')
parser.add_argument('--test', action='store_true', help='whether to test')
args = parser.parse_args()


test = args.test
prefix = ""

root_log_dir = args.log_dir
all_df_list = []
for roadnet in os.listdir(root_log_dir):
    log_dir = os.path.join(root_log_dir, roadnet)
    if not os.path.isdir(log_dir):
        continue
    print(log_dir)

    if not test:
        episode_reward, episode_travel_time = read_traing_result(log_dir)
        visualize(episode_reward, "trainging reward" + prefix, log_dir)
        visualize(episode_travel_time, "travel time" + prefix, log_dir)

    metric = read_test_result(log_dir)

    pprint.pprint(metric)


    df = pd.DataFrame(metric).rename_axis("model").reset_index()
    df["roadnet"] = roadnet
    df.to_csv(os.path.join(log_dir, 'res{}.csv'.format(prefix)))
    all_df_list.append(df)

all_df = pd.concat(all_df_list)
group_df = all_df.groupby(["model"]).agg({"waiting_time": 'mean', "travel_time": "mean", "throughput": "mean"}).reset_index()
group_df.to_csv(os.path.join(root_log_dir, 'res{}.csv'.format(prefix)))
all_df.to_csv(os.path.join(root_log_dir, 'res_all{}.csv'.format(prefix)))
print(group_df)

