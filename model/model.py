import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def preprocess_obs(obs, device):
    """
    Preprocess the observation: convert it to a PyTorch tensor and handle NaN or Inf values.
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(device)
    elif isinstance(obs, torch.Tensor):
        obs = obs.to(device)

    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


    return obs

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.action_mean = nn.Linear(128, 1)
        self.action_logstd = nn.Parameter(torch.zeros(1))

        self.lane_logits = nn.Linear(128, 3)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))

        action_mean = self.action_mean(x)
        action_logstd = self.action_logstd.expand_as(action_mean)

        lane_logits = self.lane_logits(x)

        return action_mean, action_logstd, lane_logits

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        value = self.value_head(x)
        value = value.squeeze(-1)
        return value

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)