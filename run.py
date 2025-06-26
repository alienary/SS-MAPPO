import argparse  
import logging
import logging.handlers
import os
import csv
import torch
import numpy as np
from environment.environment import MultiAgentEnvironment
from agent import PPOAgent
from environment.world import Intersection
from model import PolicyNetwork, ValueNetwork, init_weights, preprocess_obs
from config import get_config
from utils.utilities_function import set_random_seeds
from multiprocessing import Process, set_start_method
from datetime import datetime
import wandb
import re
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_args():
    parser = argparse.ArgumentParser(description='Run Multi-Agent PPO')
    parser.add_argument('--roadnet', type=str, required=True, help='roadnet')
    parser.add_argument('--data_dir', type=str, default=None, help='data dir storing roadnet file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--episodes', type=int, default=None, help='training episodes')
    parser.add_argument('--log_dir', type=str, default=None, help='directory for logs and model saving')
    parser.add_argument('--device', type=str, default=None, help='device for computing')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--n_exp', type=int, default=None, help='number of experiments')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the saved model')
    parser.add_argument('--continue', dest='continue_training', action='store_true', help='Continue training from a saved model')
    parser.add_argument('--sumo_only', action='store_true', help='Run simulation using only SUMO without reinforcement learning agent')
    args = parser.parse_args()
    return args

def setup_distributed_training(local_rank, world_size, master_port):
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

def logger_process(log_queue, log_path, debug=False):
    """
    Listen to the log queue and write logs to both file and console.
    """

    logger = logging.getLogger("MultiAgentPPO_mix")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    while True:
        try:
            record = log_queue.get()
            if record is None:  # Sentinel to stop the listener
                break
            logger.handle(record)
        except Exception:
            import sys
            import traceback
            print('Problem handling log record:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def setup_logger(log_queue):
    """
    Configure the logger in a subprocess, using QueueHandler to send logs to the main process.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    logger.propagate = False
    
    return logger

def parse_train_log(train_log_path):
    """
    Parse configuration parameters from the train.log file, excluding specified keys.
    """

    cfg_dict = {}
    exclude_keys = ['log_dir', 'model_path']
    with open(train_log_path, 'r') as f:
        for line in f:
            match = re.search(r'- INFO - (\w+): (.*)', line)
            if match:
                key = match.group(1)
                if key in exclude_keys:
                    continue 
                value = match.group(2).strip()
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  
                cfg_dict[key] = value
    return cfg_dict

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def train(cfg, log_path, log_queue, process_rank):
    # initialize logger
    local_rank = cfg["local_rank"]
    world_size = cfg["n_gpus"]
    master_port = cfg.get("master_port", 29500)  

    is_main = (world_size > 1 and local_rank == 0) or (world_size == 1)

    if world_size > 1:
        setup_distributed_training(local_rank, world_size, master_port)

    seed = cfg["seed"]
    set_random_seeds(seed + local_rank)  

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + local_rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger = setup_logger(log_queue)
    logger.info(f"Process {process_rank} started with local_rank {local_rank}.")

    if is_main:
        logger.info("Starting new training with the following hyperparameters:")
        for key, value in cfg.items():
            logger.info(f"{key}: {value}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project='MultiAgentPPO_mix',
            config=cfg,
            name=f"{cfg['roadnet']}_{timestamp}",
            mode="disabled" if cfg.get("debug") else "online",
            settings=wandb.Settings(start_method='thread', _disable_stats=True)
        )
        logger.info("wandb initialized in train function.")

    # initialize environment
    env = MultiAgentEnvironment(cfg)
    obs = env.reset()

    num_lanes = env.num_lanes
    max_vehicle_num = env.max_vehicle_num

    policy_net = PolicyNetwork(cfg["state_dim"]).to(device)
    value_net = ValueNetwork(cfg["state_dim"]).to(device)

    if world_size > 1:
        policy_net = DDP(policy_net, device_ids=[local_rank])
        value_net = DDP(value_net, device_ids=[local_rank])

    agent = PPOAgent(cfg, policy_net, value_net, device)

    if cfg.get("model_path") and cfg.get("continue_training"):
        checkpoint = torch.load(cfg["model_path"], map_location=device)

        if isinstance(policy_net, DDP):
            policy_net.module.load_state_dict(remove_module_prefix(checkpoint['policy_net_state_dict']))
            value_net.module.load_state_dict(remove_module_prefix(checkpoint['value_net_state_dict']))
        else:
            policy_net.load_state_dict(remove_module_prefix(checkpoint['policy_net_state_dict']))
            value_net.load_state_dict(remove_module_prefix(checkpoint['value_net_state_dict']))

        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_main:
            logger.info("Loaded model and optimizer state for continued training.")
    else:
        policy_net.apply(init_weights)
        value_net.apply(init_weights)
        if is_main:
            logger.info("Initialized new model weights.")

    best_average_reward = -np.inf

    last_observations = {}
    last_actions = {}
    last_log_probs = {}
    last_values = {}

    for episode in range(cfg["episodes"]):
        if is_main:
            logger.debug(f"Starting episode {episode + 1}")
        observations = env.reset()
        trajectories = {}
        trajectory_data = []
        total_average_reward_per_step = 0  
        step = 0

        while True:
            if len(env.get_all_vehicle_ids()) == 0 or step>= cfg["max_steps"]:
                break
            actions = {}
            log_probs = {}
            values = {}

            if not observations:
                logger.debug("No observations for vtype1 vehicles. Skipping action selection.")
                next_observations, rewards, dones, infos = env.step(actions)
                step += 1
                observations = next_observations
                continue

            processed_observations = {}
            for veh_id, obs in observations.items():
                veh_type = env.vehicle_types.get(veh_id)
                if veh_type == 'vtype1':
                    processed_obs = preprocess_obs(obs, device=device)
                    if torch.isnan(processed_obs).any() or torch.isinf(processed_obs).any():
                        logger.error(f"Processed obs contains NaN or Inf: {processed_obs}")
                        raise ValueError(f"Processed obs contains NaN or Inf: {processed_obs}")

                    logger.debug(f"Processed obs for veh_id {veh_id}: {processed_obs}")
                    processed_observations[veh_id] = processed_obs
                    logger.debug(f"Select action for veh_id {veh_id}")
                    action, log_prob, value = agent.select_action(processed_obs)
                    actions[veh_id] = action
                    log_probs[veh_id] = log_prob
                    values[veh_id] = value

            for veh_id in processed_observations.keys():
                veh_type = env.vehicle_types.get(veh_id)
                if veh_type == 'vtype1':
                    if veh_id not in trajectories:
                        trajectories[veh_id] = {
                            'states': [], 'actions_acc': [], 'actions_lane': [], 'log_probs_acc': [], 'log_probs_lane': [],
                            'rewards': [], 'masks': [], 'values': []
                        }
                    trajectories[veh_id]['states'].append(processed_observations[veh_id].detach().cpu())
                    trajectories[veh_id]['actions_acc'].append(actions[veh_id]['acceleration'].detach())
                    trajectories[veh_id]['actions_lane'].append(actions[veh_id]['lane_change'].detach())
                    trajectories[veh_id]['log_probs_acc'].append(log_probs[veh_id]['acceleration'].detach())
                    trajectories[veh_id]['log_probs_lane'].append(log_probs[veh_id]['lane_change'].detach())
                    trajectories[veh_id]['values'].append(values[veh_id].detach())

            next_observations, rewards, dones, infos = env.step(actions)
            step += 1

            vehicle_ids = env.get_all_vehicle_ids()
            for veh_id in vehicle_ids:
                vtype, lane_id, distance_to_intersection, speed = env.get_vehicle_data(veh_id)
                if vtype is None:
                    continue
                signal_phase = env.get_signal_phase(lane_id)
                trajectory_data.append({
                    'step': step,
                    'vehid': veh_id,
                    'vtype': vtype,
                    'lane_id': lane_id,
                    'distance_to_intersection': distance_to_intersection,
                    'speed': speed,
                    'signal_phase': signal_phase
                })

            # update total_average_reward_per_step
            if rewards:
                total_reward = sum(rewards.values())
                total_average_reward_per_step += total_reward / len(rewards)

            #store rewards and dones
            for veh_id in rewards.keys():
                veh_type = env.vehicle_types.get(veh_id)
                if veh_type == 'vtype1' and veh_id in trajectories:
                    reward = rewards[veh_id]
                    done = dones.get(veh_id, False)
                    trajectories[veh_id]['rewards'].append(torch.tensor(reward, dtype=torch.float32).to(device))
                    trajectories[veh_id]['masks'].append(torch.tensor(1 - done, dtype=torch.float32).to(device))
                    logger.debug(f"Vehicle ID: {veh_id} - Reward recorded: {reward}, Done: {done}")

            for veh_id in dones:
                if dones[veh_id] and veh_id in trajectories:
                    last_obs = observations.get(veh_id, None)
                    if last_obs is not None:
                        processed_last_obs = preprocess_obs(last_obs, device=device)
                        logger.debug(f"Select action for done veh_id {veh_id}")
                        _, _, last_value = agent.select_action(processed_last_obs)
                        trajectories[veh_id]['last_value'] = last_value.detach()  
                    else:
                        last_value = torch.zeros(1, device=device)
                        trajectories[veh_id]['last_value'] = last_value

            for veh_id in list(observations.keys()):
                if dones.get(veh_id, False):
                    del observations[veh_id]
                    logger.debug(f"Vehicle ID: {veh_id} is done and removed from observations.")

            observations = next_observations

            env.remove_left_vehicles()

        
        if not trajectories:
            if is_main:
                logger.warning("No trajectories collected in this episode. Skipping update.")
            continue  

        if step > 0:
            average_reward_per_vehicle = total_average_reward_per_step / step
        else:
            average_reward_per_vehicle = 0

        average_waiting_time = env.get_average_waiting_time()
        average_delay = env.get_average_delay()
        average_fuel_consumption = env.get_average_fuel_consumption()
        average_emissions = env.get_average_emissions()  
        average_travel_time = env.get_average_travel_time()
        throughput = env.get_throughput()
        average_stop_and_go = env.get_average_stop_and_go()

        emission_types = ['CO', 'CO2', 'HC', 'NOx', 'PMx']

        for veh_id, traj in trajectories.items():
            values_list = torch.stack(traj['values'])  # len = N
            last_value = traj.get('last_value', None)
            if last_value is None:
                with torch.no_grad():
                    last_obs = observations.get(veh_id, None)
                    if last_obs is not None:
                        processed_last_obs = preprocess_obs(last_obs, device=device)
                        logger.debug(f"Select action for value for veh_id {veh_id}")
                        _, _, last_value = agent.select_action(processed_last_obs)
                    else:
                        last_value = torch.zeros(1, device=device)
                last_value = last_value.unsqueeze(0)

            if last_value.dim() == 0:
                last_value = torch.tensor([0.0], device=device)
            elif last_value.dim() == 1 and last_value.shape[0] != 1:
                logger.warning(f"Vehicle ID: {veh_id} - last_value shape is {last_value.shape}, expected [1]. Assigning default value 0.")
                last_value = torch.tensor([0.0], device=device)

            logger.debug(f"Vehicle ID: {veh_id} - values_list shape: {values_list.shape}, last_value shape: {last_value.shape}")

            values_with_last = torch.cat([values_list, last_value], dim=0)  # len = N + 1

            rewards = torch.stack(traj['rewards'])
            masks = torch.stack(traj['masks'])
            assert rewards.shape[0] == masks.shape[0] == values_with_last.shape[0] - 1, \
                f"Length mismatch for vehicle {veh_id}: rewards={rewards.shape[0]}, masks={masks.shape[0]}, values_with_last={values_with_last.shape[0]}"

            returns, advantages = agent.compute_returns(rewards, masks, values_with_last)
            traj['returns'] = returns
            traj['advantages'] = advantages
            traj['old_values'] = values_list  

            logger.debug(f"Vehicle ID: {veh_id} - Trajectory length: {len(traj['states'])}, Total Reward: {sum(traj['rewards']).item()}")

        if any(len(traj['actions_acc']) == 0 for traj in trajectories.values()):
            if is_main:
                logger.warning("Some trajectories have empty actions_acc.")
                logger.debug(f"Trajectories: {trajectories}")
            continue 

        combined_trajectories = {
            'states': torch.stack([state for traj in trajectories.values() for state in traj['states']]),
            'actions_acc': torch.stack([action for traj in trajectories.values() for action in traj['actions_acc']]),
            'actions_lane': torch.stack([action for traj in trajectories.values() for action in traj['actions_lane']]),
            'log_probs_acc': torch.stack([log_prob for traj in trajectories.values() for log_prob in traj['log_probs_acc']]),
            'log_probs_lane': torch.stack([log_prob for traj in trajectories.values() for log_prob in traj['log_probs_lane']]),
            'returns': torch.stack([ret for traj in trajectories.values() for ret in traj['returns']]),
            'advantages': torch.stack([adv for traj in trajectories.values() for adv in traj['advantages']]),
            'old_values': torch.stack([val for traj in trajectories.values() for val in traj['old_values']]),  # 新增
        }

        loss_dict = agent.update(combined_trajectories)

        # Reduce data across processes (collective operation, all processes must participate)
        if world_size > 1:
            action_loss_tensor = torch.tensor(loss_dict.get('action_loss', 0), device=device)
            value_loss_tensor = torch.tensor(loss_dict.get('value_loss', 0), device=device)
            entropy_loss_tensor = torch.tensor(loss_dict.get('entropy_loss', 0), device=device)
            vf_explained_var_tensor = torch.tensor(loss_dict.get('vf_explained_var', 0), device=device)
            average_reward_tensor = torch.tensor(average_reward_per_vehicle, device=device)
            average_waiting_time_tensor = torch.tensor(average_waiting_time, device=device)
            average_delay_tensor = torch.tensor(average_delay, device=device)
            average_fuel_consumption_tensor = torch.tensor(average_fuel_consumption, device=device)
            average_travel_time_tensor = torch.tensor(average_travel_time, device=device)
            throughput_tensor = torch.tensor(throughput, device=device)
            average_stop_and_go_tensor = torch.tensor(average_stop_and_go, device=device)

            average_emission_tensors = {}
            for emission_type in emission_types:
                emission_value = average_emissions.get(emission_type, 0.0)
                average_emission_tensors[emission_type] = torch.tensor(emission_value, device=device)

            torch.distributed.reduce(action_loss_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(value_loss_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(entropy_loss_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(vf_explained_var_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_reward_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_waiting_time_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_delay_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_fuel_consumption_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_travel_time_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(throughput_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(average_stop_and_go_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            for emission_type in emission_types:
                torch.distributed.reduce(average_emission_tensors[emission_type], dst=0, op=torch.distributed.ReduceOp.SUM)

            if is_main:
                num_processes = world_size
                action_loss_avg = action_loss_tensor.item() / num_processes
                value_loss_avg = value_loss_tensor.item() / num_processes
                entropy_loss_avg = entropy_loss_tensor.item() / num_processes
                vf_explained_var_avg = vf_explained_var_tensor.item() / num_processes
                average_reward_avg = average_reward_tensor.item() / num_processes
                average_waiting_time_avg = average_waiting_time_tensor.item() / num_processes
                average_delay_avg = average_delay_tensor.item() / num_processes
                average_fuel_consumption_avg = average_fuel_consumption_tensor.item() / num_processes
                average_travel_time_avg = average_travel_time_tensor.item() / num_processes
                throughput_avg = throughput_tensor.item() / num_processes
                average_stop_and_go_avg = average_stop_and_go_tensor.item() / world_size
                
                average_emissions_avg = {}
                for emission_type in emission_types:
                    avg_value = average_emission_tensors[emission_type].item() / num_processes
                    average_emissions_avg[emission_type] = avg_value
        else:
            action_loss_avg = loss_dict.get('action_loss', 0)
            value_loss_avg = loss_dict.get('value_loss', 0)
            entropy_loss_avg = loss_dict.get('entropy_loss', 0)
            vf_explained_var_avg = loss_dict.get('vf_explained_var', 0)
            average_reward_avg = average_reward_per_vehicle
            average_waiting_time_avg = average_waiting_time
            average_delay_avg = average_delay
            average_fuel_consumption_avg = average_fuel_consumption
            average_emissions_avg = average_emissions  # 字典
            average_travel_time_avg = average_travel_time
            throughput_avg = throughput
            average_stop_and_go_avg = average_stop_and_go

        # Update best_average_reward and save model if necessary
        if is_main:
            if average_reward_avg > best_average_reward:
                best_average_reward = average_reward_avg
                # Save the model as before
                model_save_path = os.path.join(log_path, "best_model.pt")
                if isinstance(policy_net, DDP):
                    torch.save({
                        'policy_net_state_dict': policy_net.module.state_dict(),
                        'value_net_state_dict': value_net.module.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, model_save_path)
                else:
                    torch.save({
                        'policy_net_state_dict': policy_net.state_dict(),
                        'value_net_state_dict': value_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, model_save_path)
                logger.info(f"Best model saved at episode {episode + 1} with average reward {average_reward_avg:.2f}")

                try:
                    csv_file_path = os.path.join(log_path, 'best_model_trajectory_data.csv')
                    with open(csv_file_path, 'w', newline='') as csvfile:
                        fieldnames = ['step', 'vehid', 'vtype', 'lane_id', 
                                      'distance_to_intersection', 'speed', 'signal_phase']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in trajectory_data:
                            writer.writerow(data)
                    logger.info(f"Best-model trajectory data saved to {csv_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save best-model trajectory data: {e}")


            # Log to wandb
            log_data = {
                'action_loss': action_loss_avg,
                'value_loss': value_loss_avg,
                'entropy_loss': entropy_loss_avg,
                'vf_explained_var':vf_explained_var_avg,
                'average_reward_per_vehicle': average_reward_avg,
                'average_waiting_time': average_waiting_time_avg,
                'average_delay': average_delay_avg,
                'average_fuel_consumption': average_fuel_consumption_avg,
                'average_travel_time': average_travel_time_avg,
                'throughput': throughput_avg,
                'average_stop_and_go': average_stop_and_go,
            }

            for emission_type in emission_types:
                log_data[f'average_emission_{emission_type}'] = average_emissions_avg.get(emission_type, 0.0)

            wandb.log(log_data)

            # Log training progress
            emissions_log_str = ' - '.join([f"{emission_type}: {average_emissions_avg.get(emission_type, 0.0):.2f} mg" for emission_type in emission_types])

            # Log training progress
            logger.info(f"Episode {episode + 1}/{cfg['episodes']} - "
                f"Average Reward per Vehicle: {average_reward_avg:.2f} - "
                f"Average Waiting Time: {average_waiting_time_avg:.2f} s - "
                f"Average Delay: {average_delay_avg:.2f} s - "
                f"Average Fuel Consumption: {average_fuel_consumption_avg:.4f} ml - "
                f"Average Travel Time: {average_travel_time_avg:.2f} s - "
                f"Throughput: {throughput_avg} vehicles - "
                f"Emissions - {emissions_log_str} - "
                f"Average StopAndGo: {average_stop_and_go:.2f} - "
                f"Action Loss: {action_loss_avg:.4f} - "
                f"Value Loss: {value_loss_avg:.4f} - "
                f"Entropy Loss: {entropy_loss_avg:.4f} - "
                f"vf_explained_var: {vf_explained_var_avg:.4f}")
        else:
            # Other processes can log minimal info if desired
            logger.info(f"Process {process_rank} completed episode {episode + 1}/{cfg['episodes']}.")

    if is_main:
        wandb.finish()


def test(cfg, model_path, process_rank): 
    """
    Test the trained model
    """

    logger = logging.getLogger(f"TestProcess_{process_rank}")
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
    
    logger.info(f"Process {process_rank} started for testing.")

    model_dir = os.path.dirname(model_path)
    parts = model_dir.split(os.sep) # parts = ["logs", "roadnet_A", "20240101_123000"]
    roadnet = parts[1]  
    cfg["roadnet"] = roadnet
    
    env = MultiAgentEnvironment(cfg)
    observations = env.reset()


    policy_net = PolicyNetwork(cfg["state_dim"]).to(cfg["device"])
    value_net = ValueNetwork(cfg["state_dim"]).to(cfg["device"])

    agent = PPOAgent(cfg, policy_net, value_net, cfg["device"])
    agent.load(model_path, set_eval=True)

    step = 0 

    done = False
    trajectory_data = []

    while True:
        actions = {}
        for veh_id, obs in observations.items():
            veh_type = env.vehicle_types.get(veh_id)
            if veh_type == 'vtype1':
                processed_obs = preprocess_obs(obs, device=cfg["device"])

                if torch.isnan(processed_obs).any() or torch.isinf(processed_obs).any():
                    logger.error(f"Processed obs contains NaN or Inf for veh_id {veh_id}: {processed_obs}")
                    continue  

                action, _, _ = agent.select_action(processed_obs)  
                actions[veh_id] = action

        next_observations, rewards, dones, infos = env.step(actions)
        step += 1

        vehicle_ids = env.get_all_vehicle_ids()
        for veh_id in vehicle_ids:
            vtype, lane_id, distance_to_intersection, speed = env.get_vehicle_data(veh_id)
            if vtype is None:
                continue  
            signal_phase = env.get_signal_phase(lane_id)
            trajectory_data.append({
                'step': step,
                'vehid': veh_id,
                'vtype': vtype,
                'lane_id': lane_id,
                'distance_to_intersection': distance_to_intersection,
                'speed': speed,
                'signal_phase': signal_phase
            })

        observations = next_observations
        if len(env.get_all_vehicle_ids()) == 0 or step>= cfg["max_steps"]:
            break
        
    csv_file_path = os.path.join(os.path.dirname(model_path), 'trajectory_data.csv')
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['step', 'vehid', 'vtype', 'lane_id', 'distance_to_intersection', 'speed', 'signal_phase']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in trajectory_data:
                writer.writerow(data)
        logger.info(f"Trajectory data saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Failed to save trajectory data: {e}")

def sumo_only(cfg, log_path, log_queue):
    """
    Use SUMO's built-in logic for control instead of a reinforcement learning agent.
    After the simulation ends, output and record the average waiting time, average delay, average fuel consumption, and emissions.
    """


    print("Starting SUMO-only mode...")
    
    trajectory_data = []

    try:
        env = MultiAgentEnvironment(cfg)
        observations = env.reset()

        step = 0

        while True:
            actions = {} 

            next_observations, rewards, dones, infos = env.step(actions)
            step += 1

            vehicle_ids = env.get_all_vehicle_ids()
            for veh_id in vehicle_ids:
                vtype, lane_id, distance_to_intersection, speed = env.get_vehicle_data(veh_id)
                if vtype is None:
                    continue
                signal_phase = env.get_signal_phase(lane_id)
                trajectory_data.append({
                    'step': step,
                    'vehid': veh_id,
                    'vtype': vtype,
                    'lane_id': lane_id,
                    'distance_to_intersection': distance_to_intersection,
                    'speed': speed,
                    'signal_phase': signal_phase
                })

            observations = next_observations

            env.remove_left_vehicles()

            if len(env.get_all_vehicle_ids()) == 0 or step>= cfg["max_steps"]:
                break

        average_waiting_time = env.get_average_waiting_time()
        average_delay = env.get_average_delay()
        average_fuel_consumption = env.get_average_fuel_consumption()
        average_emissions = env.get_average_emissions() 
        average_travel_time = env.get_average_travel_time()
        throughput = env.get_throughput()
        average_stop_and_go = env.get_average_stop_and_go()

        print(f"sumo only mode completed.")
        print(f"average waiting time: {average_waiting_time:.2f} 秒")
        print(f"average travel time: {average_travel_time:.2f} 秒")
        print(f"throughput: {throughput:.2f} 辆/秒")
        print(f"average delay: {average_delay:.2f} 秒")
        print(f"average fuel consumption: {average_fuel_consumption:.4f} ml")
        print(f"average stop and go: {average_stop_and_go:.2f} ")
        print("average emissions:")
        for emission_type, value in average_emissions.items():
            print(f"{emission_type}: {value:.2f} mg")

    except Exception as e:
        print(f"sumo_only mode encountered an error: {e}")
    finally:
        env.close()

        csv_file_path = os.path.join(log_path, 'trajectory_data_sumo_only.csv')
        try:
            with open(csv_file_path, 'w', newline='') as csvfile:
                fieldnames = ['step', 'vehid', 'vtype', 'lane_id', 'distance_to_intersection', 'speed', 'signal_phase']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in trajectory_data:
                    writer.writerow(data)
            print(f"SUMO-only trajectory data saved to {csv_file_path}")
        except Exception as e:
            print(f"Failed to save SUMO-only trajectory data: {e}")

def main_worker(local_rank, cfg, log_path, log_queue):
    cfg["local_rank"] = local_rank
    process_rank = local_rank  
    train(cfg, log_path, log_queue, process_rank)

def main():
    args = parse_args()
    cfg = get_config()

    if args.model_path and args.continue_training:
        cfg["model_path"] = args.model_path
        model_dir = os.path.dirname(cfg["model_path"])
        train_log_path = os.path.join(model_dir, 'train.log')
        loaded_cfg = parse_train_log(train_log_path)
        cfg.update(loaded_cfg)
        print("Loaded configuration from previous training.")

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    if args.sumo_only:
        cfg["sumo_only"] = True
    elif args.test:
        cfg["sumo_only"] = False
    else:
        cfg["sumo_only"] = False

    if torch.cuda.is_available():
        cfg["device"] = "cuda"
    else:
        cfg["device"] = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg["log_dir"], cfg["roadnet"], timestamp)
    os.makedirs(log_path, exist_ok=True)

    log_queue = mp.Queue()

    listener = mp.Process(target=logger_process, args=(log_queue, log_path, cfg.get("debug", False)))
    listener.start()

    cfg["log_queue"] = log_queue

    if cfg.get("sumo_only"):
        sumo_only_process = Process(target=sumo_only, args=(cfg, log_path, log_queue))
        sumo_only_process.start()
        sumo_only_process.join()
    elif cfg.get("test"):
        if not cfg.get("model_path"):
            temp_logger = setup_logger(log_queue)
            temp_logger.error("please provide the model path for testing mode using --model_path argument.")
            raise ValueError("Model path is required for testing mode.")
        test_process = Process(target=test, args=(cfg, cfg["model_path"], 0))
        test_process.start()
        test_process.join()
    else:
        if cfg["n_gpus"] > 1:
            mp.spawn(main_worker, nprocs=cfg["n_gpus"], args=(cfg, log_path, log_queue))
        else:
            train(cfg, log_path, log_queue, process_rank=cfg["local_rank"])

    log_queue.put_nowait(None)
    listener.join()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()