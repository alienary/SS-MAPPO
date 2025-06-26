import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal, TransformedDistribution, TanhTransform
import logging
import numpy as np
from collections import OrderedDict
import math

class PPOAgent:
    def __init__(self, cfg, policy_net, value_net, device):
        self.logger = logging.getLogger("PPOAgent")
        self.logger.debug("Initializing PPOAgent.")

        self.policy_net = policy_net.module if hasattr(policy_net, 'module') else policy_net
        self.value_net = value_net.module if hasattr(value_net, 'module') else value_net
        self.device = device

        self.policy_net = self.policy_net.float()
        self.value_net = self.value_net.float()

        self.gamma = cfg["gamma"]
        self.clip_range = cfg["clip_range"]
        self.clip_range_value = cfg.get("clip_range_value", 0.2)  
        self.lr = cfg["lr"]
        self.batch_size = cfg["batch_size"]
        self.entropy_coef = cfg["entropy_coef"]
        self.vf_coef = cfg["vf_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.n_epochs = cfg["n_epochs"]
        self.lam = cfg.get("lam", 0.95) 

        self.optimizer = optim.AdamW(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=self.lr
        )

        self.logger.debug("PPOAgent initialization complete.")

    def select_action(self, obs):
        obs = obs.to(self.device)

        action_mean, action_logstd, lane_logits = self.policy_net(obs)
        self.logger.debug(f"Action mean: {action_mean}, Action logstd: {action_logstd}")

        action_logstd = torch.clamp(action_logstd, min=-20, max=2)
        action_std = torch.exp(action_logstd)

        self.logger.debug(f"Action std after clamping and exp: {action_std}")

        if (action_std <= 0).any():
            self.logger.error("action_std contains non-positive values.")
            raise ValueError("action_std must be positive.")

        normal_dist = Normal(action_mean, action_std)

        tanh_normal = TransformedDistribution(normal_dist, TanhTransform(cache_size=1))

        tanh_action = tanh_normal.rsample()
        self.logger.debug(f"Tanh action: {tanh_action}")

        action_scale = 1.5
        scaled_action = tanh_action * action_scale

        log_prob_acc = tanh_normal.log_prob(tanh_action)
        log_prob_acc -= math.log(action_scale)  

        if log_prob_acc.dim() > 1:
            log_prob_acc = log_prob_acc.sum(-1)

        lane_dist = Categorical(logits=lane_logits)
        lane_change = lane_dist.sample()
        log_prob_lane = lane_dist.log_prob(lane_change)

        with torch.no_grad():
            value = self.value_net(obs).squeeze(-1)

        action = {
            "acceleration": scaled_action,
            "lane_change": lane_change,
        }
        action_log_prob = {
            "acceleration": log_prob_acc,
            "lane_change": log_prob_lane,
        }

        return action, action_log_prob, value

    def compute_returns(self, rewards, masks, values_with_last):
        self.logger.debug("Computing returns and advantages.")
        returns = []
        gae = 0

        rewards = rewards.to(self.device)
        masks = masks.to(self.device)
        values_with_last = values_with_last.to(self.device)

        self.logger.debug(f"Rewards shape: {rewards.shape}, Masks shape: {masks.shape}, Values_with_last shape: {values_with_last.shape}")

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_with_last[step + 1] * masks[step] - values_with_last[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values_with_last[step])

        returns = torch.stack(returns)
        advantages = returns - values_with_last[:-1]

        self.logger.debug(f"Returns shape: {returns.shape}, Advantages shape: {advantages.shape}")

        return returns.detach(), advantages.detach()

    def update(self, trajectories):
        self.logger.debug("Starting update process.")

        self.policy_net.train()
        self.value_net.train()

        states_tensor = trajectories['states'].to(self.device)
        actions_acc = trajectories['actions_acc'].to(self.device)
        actions_lane = trajectories['actions_lane'].to(self.device)
        old_log_probs_acc = trajectories['log_probs_acc'].to(self.device)
        old_log_probs_lane = trajectories['log_probs_lane'].to(self.device)
        returns = trajectories['returns'].to(self.device)
        advantages = trajectories['advantages'].to(self.device)
        old_values = trajectories['old_values'].to(self.device)

        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std < 1e-5:
            self.logger.warning(f"Advantages standard deviation too small: {advantages_std.item()}, adding 1e-5")
            advantages_std += 1e-5
        advantages = (advantages - advantages_mean) / advantages_std

        dataset_size = states_tensor.size(0)
        self.logger.debug(f"Dataset size: {dataset_size}")

        total_action_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.n_epochs):
            indices = torch.randperm(dataset_size).to(self.device)
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states_tensor[batch_indices]
                batch_actions_acc = actions_acc[batch_indices]
                batch_actions_lane = actions_lane[batch_indices]
                batch_old_log_probs_acc = old_log_probs_acc[batch_indices]
                batch_old_log_probs_lane = old_log_probs_lane[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]

                action_mean, action_logstd, lane_logits = self.policy_net(batch_states)

                action_logstd = torch.clamp(action_logstd, min=-20, max=2)
                action_std = torch.exp(action_logstd)

                normal_dist = Normal(action_mean, action_std)
                tanh_normal = TransformedDistribution(normal_dist, TanhTransform(cache_size=1))

                action_scale = 1.5
                tanh_actions = batch_actions_acc / action_scale
                tanh_actions = torch.clamp(tanh_actions, min=-0.999999, max=0.999999)

                new_log_probs_acc = tanh_normal.log_prob(tanh_actions)
                new_log_probs_acc -= math.log(action_scale)  

                if new_log_probs_acc.dim() > 1:
                    new_log_probs_acc = new_log_probs_acc.sum(-1)

                lane_dist = Categorical(logits=lane_logits)
                new_log_probs_lane = lane_dist.log_prob(batch_actions_lane)

                ratio_acc = (new_log_probs_acc - batch_old_log_probs_acc).exp()
                ratio_lane = (new_log_probs_lane - batch_old_log_probs_lane).exp()

                surr1_acc = ratio_acc * batch_advantages
                surr2_acc = torch.clamp(ratio_acc, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                surr1_lane = ratio_lane * batch_advantages
                surr2_lane = torch.clamp(ratio_lane, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages

                action_loss_acc = torch.min(surr1_acc, surr2_acc).mean()
                action_loss_lane = torch.min(surr1_lane, surr2_lane).mean()
                action_loss = -(action_loss_acc + action_loss_lane)

                value_pred = self.value_net(batch_states).squeeze(-1)
                value_pred_clipped = batch_old_values + (value_pred - batch_old_values).clamp(-self.clip_range_value, self.clip_range_value)
                value_loss_unclipped = (value_pred - batch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_acc = tanh_normal.base_dist.entropy().mean()
                entropy_lane = lane_dist.entropy().mean()
                entropy_loss = entropy_acc + entropy_lane

                loss = action_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                total_action_loss += action_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        num_batches = self.n_epochs * math.ceil(dataset_size / self.batch_size)
        avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
        avg_entropy_loss = total_entropy_loss / num_batches if num_batches > 0 else 0.0

        with torch.no_grad():
            all_value_preds = self.value_net(states_tensor).squeeze(-1)
        vf_explained_var = 1 - torch.var(returns - all_value_preds) / (torch.var(returns) + 1e-8)
        vf_explained_var = vf_explained_var.item()

        return {
            'action_loss': avg_action_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'vf_explained_var': vf_explained_var,
        }

    def load_state_dict_custom(self, model, state_dict, prefix='module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        self.logger.debug(f"Loaded state_dict for {model.__class__.__name__} with prefix '{prefix}' removed.")

    def load(self, model_path, set_eval=True):
        """Load model state dictionary"""
        self.logger.debug(f"Loading model from {model_path}.")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.load_state_dict_custom(self.policy_net, checkpoint['policy_net_state_dict'], prefix='module.')
        self.load_state_dict_custom(self.value_net, checkpoint['value_net_state_dict'], prefix='module.')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.debug("Optimizer state loaded.")

        if set_eval:
            self.policy_net.eval()
            self.value_net.eval()
            self.logger.debug("Models set to evaluation mode.")
        else:
            self.policy_net.train()
            self.value_net.train()
            self.logger.debug("Models set to training mode.")
