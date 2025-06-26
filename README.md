# SS-MAPPO: Simulation-Supervised Multi-Agent Proximal Policy Optimization

The **SS-MAPPO** project implements a simulation-supervised multi-agent reinforcement learning framework for traffic control. It focuses on training multiple reinforcement learning agents using the Proximal Policy Optimization (PPO) algorithm to optimize traffic flow in road intersections. The project uses a custom-built environment for multi-agent simulations, powered by the popular SUMO traffic simulator, and is based on Python 3.11.10.

The code is organized in several key components:

* **environment/**: Contains the environment setup for agent interactions and traffic simulation.
* **agent/**: Contains the PPO agent responsible for decision-making and training.
* **model/**: Houses the neural network architectures for the policy and value networks.
* **utils/**: Includes utility functions for data processing, plotting, and other auxiliary tasks.

> **Note**: The code inside the **environment** folder (specifically `environment.py` and `world.py`) and the data files used for training will be made publicly available once the paper is accepted for publication.

## Table of Contents

* [Installation](#installation)
* [Requirements](#requirements)
* [Usage](#usage)

  * [Training](#training)
  * [Testing](#testing)
  * [SUMO-Only Mode](#sumo-only-mode)
* [Logging and Monitoring](#logging-and-monitoring)
* [Configuration](#configuration)

## Installation

### Install Anaconda

We recommend installing [Anaconda](https://www.anaconda.com/products/individual) to manage dependencies and create isolated environments for the project.

1. Create a conda environment:

   ```bash
   conda create -n ss-mappo python=3.11
   ```

2. Activate the environment:

   ```bash
   conda activate ss-mappo
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Install SUMO and RLlib

* **SUMO**: The traffic simulation platform used for generating environments for training.

  Follow the [SUMO installation instructions](https://sumo.dlr.de/docs/Downloads.php) for your system.

* **RLlib**: A reinforcement learning library used to implement the PPO algorithm.

  Install RLlib using:

  ```bash
  pip install "ray[rllib]"
  ```

### Optional: Install WandB

For experiment tracking and visualization, you can optionally install [WandB](https://www.wandb.com/):

```bash
pip install wandb
```

## Requirements

The **requirements.txt** file contains all the dependencies needed to run the project. You can install them using the following command:

```bash
pip install -r requirements.txt
```

This will install all required libraries, including:

* `torch`: For neural network training.
* `ray[rllib]`: For reinforcement learning algorithms.
* `sumo`: For the SUMO traffic simulation.
* `wandb`: For real-time experiment tracking (optional).

## Usage

### Training

To start the training process for the multi-agent PPO, use the following command:

```bash
python train.py --roadnet <roadnet_file> --episodes <number_of_episodes> --log_dir <log_directory> --device <device>
```

* `--roadnet`: Path to the road network file (required).
* `--episodes`: Number of training episodes (optional).
* `--log_dir`: Directory to save logs and models (optional).
* `--device`: Device for computation (`cuda` for GPU, `cpu` for CPU) (optional).
* `--n_exp`: Number of experiments to run (optional).
* `--model_path`: Path to the pretrained model (optional).
* `--debug`: Run in debug mode to get detailed logs (optional).
* `--test`: Enable test mode (optional).

#### Example Command:

```bash
python train.py --roadnet roadnet_config.xml --episodes 100 --log_dir ./logs --device cuda
```

This command will start training for 100 episodes using the specified road network configuration file and log the results to `./logs`.

### Testing

To test a trained model, use the following command:

```bash
python train.py --test --model_path <path_to_trained_model>
```

Ensure the path to the saved model is provided using the `--model_path` argument.

### SUMO-Only Mode

If you want to run the simulation using only SUMO (without reinforcement learning agents), you can start the SUMO-only mode with:

```bash
python train.py --sumo_only
```

This will run the simulation with the default SUMO traffic control logic and output various traffic statistics like waiting time, delay, fuel consumption, and emissions.

## Logging and Monitoring

The training process is logged both to the console and to a log file. The logs include information about episode rewards, losses, and other statistics. Additionally, logs are sent to **WandB** for real-time tracking and visualization.

If you choose to run in a distributed setup, the logs are aggregated from all processes and stored in a centralized directory for better analysis.

## Configuration

The **config.py** file contains several configurable parameters that control the behavior of the simulation and the PPO agent training. Below is a summary of key settings:

### Data and Environment Settings:

* **data\_dir**: Directory where the dataset is stored.
* **roadnet**: The road network configuration file (e.g., `"syn1"`).
* **log\_dir**: Directory for saving logs and models.
* **sumo\_cfg\_file**: The SUMO configuration file (e.g., `"data.sumocfg"`).
* **seed**: Random seed for reproducibility.
* **action\_interval**: Number of SUMO simulation steps per action.
* **target\_speed**: Desired speed for vehicles (in meters per second).
* **episodes**: Number of training episodes.


### Model Settings:

* **device**: Device for computation (`"cuda"` for GPU or `"cpu"` for CPU).
* **max\_steps**: Maximum number of steps per episode.


### Distributed Training:

* **n\_gpus**: Number of GPUs to use for distributed training.
* **local\_rank**: Local rank for each GPU in a distributed setup.
* **master\_port**: Port for setting up communication between processes in distributed training.

You can customize these settings in **config.py** according to your experimental setup.

