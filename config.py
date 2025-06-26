def get_config():
    cfg = {
        # Data and environment settings
        "data_dir": "data",
        "roadnet": "syn1",  # Path to the road network file
        "log_dir": "logs",
        "sumo_cfg_file": "data.sumocfg",  # SUMO configuration file
        "seed": 42,  # Random seed for reproducibility
        "action_interval": 1,  # Number of SUMO simulation steps per action
        "target_speed": 15,  # Desired speed (m/s)
        "episodes": 500,
        "yellow_time": 1,
        "left_turn_green_time": 16,
        "straight_green_time": 32,
        "straight_flow": 300,
        "left_flow": 150,

        # PPO hyperparameters
        "gamma": 0.9,  # Reward discount factor
        "clip_range": 0.2,  # Clipping range for PPO policy updates
        "clip_range_value": 0.2,
        "lr": 2e-4,  # Learning rate for the optimizer
        "batch_size": 256,  # Batch size for PPO updates (may need tuning based on GPU count)
        "entropy_coef": 0.01,  # Entropy regularization coefficient
        "vf_coef": 0.5,  # Value loss coefficient in PPO
        "max_grad_norm": 0.5,  # Maximum norm for gradient clipping
        "n_epochs": 4,  # Number of epochs per PPO update
        "lam": 0.95,  # GAE lambda for advantage estimation

        # Model settings
        "device": "cuda",  # In DDP mode, device should be 'cuda', assigned via local_rank
        "state_dim": 28,  # State dimension size for policy and value networks

        "max_steps": 1056,  # Max steps per episode (to avoid infinite episodes)

        # Action constraints
        "max_acceleration": 1.5,  # Maximum vehicle acceleration (m/s^2)
        "min_acceleration": -1.5,  # Minimum vehicle acceleration (m/s^2)

        # Distributed training
        "n_gpus": 1,
        "local_rank": 0,  # GPU rank assigned in distributed training, updated at runtime
        "master_port": 12316,
    }
    return cfg
