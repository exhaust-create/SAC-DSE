import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pdb

import os, sys

sys.path.append('..')
from networks.Networks import Actor_NN_no_option, V_Critic
from buffer.storage import RolloutStorage

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.01
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-1, b=1)
        nn.init.constant_(layer.bias, 0.1)
    
class PPO2_v0(nn.Module):
    """
    Description
    -------
    This `PPO2_v0` use `Actor_NN_no_option` as the actor.
    """
    def __init__(self, config, env, max_episodes, device='cpu'):
        super(PPO2_v0, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.k_step_update = config["k_step_update"]    # If set it to a very large num, then an episode only stop when `terminated/truncated=True`.
        self.num_epochs = config["num_epochs"]  # The num of training epochs of a minibatch.
        self.clip_param = config["clip_param"]
        self.entropy_coef = config["entropy_coef"]
        self.lr = config["lr"]
        self.max_episodes = max_episodes
        self.lr_gamma = 0.98
        self.rollout_storage = RolloutStorage(self.k_step_update,self.env.observation_space.nvec, self.env.action_space)
        self.actor = Actor_NN_no_option(state_dim = env.option_space.n, output_dim = self.env.action_space.n, device=device)
        # self.actor.apply(init_weights)
        self.critic = V_Critic(state_dim = env.option_space.n, device=device)
        # self.critic.apply(init_weights)
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.lr},
                                    {'params': self.critic.parameters(), 'lr': self.lr}],
                                    )
        self.optim_schedule = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        self.total_num_explored_designs = 0

        self.entropy = []
    
    def run_an_episode(self):
        """
        Description
        -------
        Run an episode and then update the networks.
        """
        terminated = False
        truncated = False
        done = False
        step_counts = 0
        current_state, info = self.env.zero_reset()
        self.rollout_storage.observations[0].copy_(torch.tensor(current_state).to(self.device))
        # Actually, `done` is controlled by both `terminated` or `truncated`. Once the time is over or the env arrives the final state, the episode should be ended.
        # But the value of the final state can only be counted when `terminated = True`.
        while not done:
            while step_counts < self.k_step_update and not truncated:
                state_value = self.critic(torch.tensor(current_state).to(self.device))
                option = torch.tensor(step_counts%8 + 2).view(1,1).to(self.device)
                action_mask = self.env.generate_action_mask(current_state, option.detach().cpu().numpy())
                action_mask = action_mask.reshape(option.shape[0],-1)
                action, action_log_prob = self.select_action(torch.tensor(current_state).to(self.device), option, action_mask)
                # Store trajectory
                # print("component_idx: {}, action: {}".format(option.cpu().detach().numpy(), action.cpu().detach().numpy()))
                next_state, reward, terminated, truncated, info = self.env.step(option, action)
                self.rollout_storage.insert(next_state, option.view(-1), action, action_log_prob, state_value, reward, 1-torch.tensor(terminated).int()) # Don't use `value_pred`.
                current_state = next_state.copy()
                step_counts += 1
                if terminated or truncated:
                    self.total_num_explored_designs += 1    # If `zero_reset()`, then plus 1 at each terminal, otherwise plus `len(self.env._explored_designs)`.
                    print("Num of explored_designs: {}, total num of explored_design: {}".format(len(self.env._explored_designs), self.total_num_explored_designs))
                    done = True
                    break
            # Record entropy
            if done:
                actor_entropy = self.compute_actor_entropy(torch.tensor(current_state).to(self.device), action_mask)
                self.entropy.append(actor_entropy)

            last_state_value = self.critic(torch.tensor(next_state).to(self.device))
            # self.rollout_storage.normalize_rewards()
            self.rollout_storage.compute_returns(last_state_value, gamma=self.config["gamma"], use_gae=True)
            self.rollout_storage.to_device(device=self.device)
            self.update_model(self.rollout_storage)
            self.rollout_storage.reset_buffer()
            step_counts = 0
            self.rollout_storage.to_device(device='cpu')

    def select_action(self, state_embed, option, action_mask):
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask).to(self.device)
        # Get action
        action_probs = self.actor(state_embed, action_mask)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)
        return action, action_log_prob

    def update_model(self, rollouts):
        states, options, actions, action_log_probs, value_preds, rewards, returns, masks = rollouts.get_buffer()
        states = states.view(-1, states.shape[-1])
        for i in range(self.num_epochs):
            action_mask = self.env.generate_action_mask(states[:-1].detach().cpu().numpy(),options.detach().cpu().numpy())
            action_mask = action_mask.reshape(options.shape[0], -1)
            action_mask = torch.tensor(action_mask).to(self.device)
            new_action_probs = self.actor(states[:-1], action_mask)
            new_action_log_probs = torch.log(new_action_probs.gather(1, actions))

            new_state_values = self.critic(states[:-1])

            ratio = torch.exp(new_action_log_probs - action_log_probs.detach())
            adv = returns[:-1].detach() - value_preds[:-1].detach()
            adv = (adv - adv.mean())/(adv.std()+1e-30)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy loss
            entropy = -self.entropy_coef * torch.sum(new_action_probs * new_action_log_probs, dim=1).mean()
            # Critic loss
            critic_loss = 0.5 * nn.functional.mse_loss(returns[:-1].detach(), new_state_values)
            total_loss = actor_loss + critic_loss - entropy
            # print("actor_loss: {}, critic_loss: {}, total_loss: {}".
            #       format(actor_loss.detach().cpu().item(), 
            #              critic_loss.detach().cpu().item(),
            #              total_loss.detach().cpu().item()))

            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            
            self.optimizer.step()
        # self.optim_schedule.step()

    def compute_actor_entropy(self, state, action_mask):
        with torch.no_grad():
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask).to(self.device)
            # Get action
            action_probs = self.actor(state, action_mask)
            action_distribution = Categorical(action_probs)
        return action_distribution.entropy().item()
    
    def schedule_lr(self, episode):
        self.lr = self.lr - \
            (episode / self.max_episodes) * self.lr
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr

class PPO2_v1(nn.Module):
    """
    Description
    -------
    This `PPO2_v1` flatten all component options to construct a action space.

    Parameters
    -------
    env:
        Must be the "env_no_option".
    """
    def __init__(self, config, env, device='cpu', embedding=False):
        super(PPO2_v1, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.embedding = embedding  # Whether embed the state.
        self.k_step_update = config["k_step_update"]    # If set it to a very large num, then an episode only stop when `terminated/truncated=True`.
        self.num_epochs = config["num_epochs"]  # The num of training epochs of a minibatch.
        self.clip_param = config["clip_param"]
        self.max_steps = config["max_steps"]
        self.entropy_coef = config["entropy_coef"]
        # self.gamma = 0.98
        # self.lr = 0.0005
        self.gamma = 0.95
        self.lr = 0.0001
        self.lr_gamma = 0.95
        self.frame_skipped = 1
        self.rollout_storage = RolloutStorage(self.max_steps, env._first_microarch_comp.shape, self.env.action_space)
        if embedding:
            self._state_embed_dim = env.dataset.microarch_embedding.embedding_dim * env.observation_space.n
            self.actor = Actor_NN_no_option(state_dim = self._state_embed_dim, output_dim = self.env.action_space.n, device=device)
            self.critic = V_Critic(state_dim = self._state_embed_dim, device=device)
            self.embed_optim = optim.Adam(env.dataset.microarch_embedding.parameters(), lr=self.lr)
        else:
            self.actor = Actor_NN_no_option(state_dim = env.observation_space.n, output_dim = self.env.action_space.n, device=device)
            # self.actor.apply(init_weights)
            self.critic = V_Critic(state_dim = env.observation_space.n, device=device)
            # self.critic.apply(init_weights)
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.lr},
                                    {'params': self.critic.parameters(), 'lr': self.lr}],
                                    )
        self.optim_schedule = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        self.total_num_explored_designs = 0
        self.indices = []

        self.entropy = []
    
    def run_an_episode(self):
        """
        Description
        -------
        Run an episode and then update the networks.
        """
        step_counts = 0
        terminated = False
        truncated = False
        done = False
        current_state, info = self.env.reset()
        while not done:
            step_counts = 0
            self.rollout_storage.observations[0].copy_(torch.tensor(current_state).to(self.device))
            with torch.no_grad():
                # Collect k-step transitions.
                while step_counts < self.k_step_update:
                    # Get the action mask for the current state
                    action_mask = self.env.generate_action_mask(current_state.astype(int))
                    if self.embedding:
                        current_state_embed = self.env.dataset.microidx_2_microembedding(torch.tensor(current_state).to(self.device))
                        current_state_embed = current_state_embed.flatten(-2, -1)
                        state_value = self.critic(current_state_embed.to(self.device))
                        action, action_log_prob = self.select_action(current_state_embed, action_mask)
                    else:
                        state_value = self.critic(torch.tensor(current_state).to(self.device))
                        action, action_log_prob = self.select_action(current_state, action_mask)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)

                    # For calculation of the num of simulated designs.
                    micro_idx = self.env.locate_microarch_idx(next_state)
                    self.indices.append(micro_idx)

                    # Skip several frames to reduce data collection.
                    if (step_counts + 1) % self.frame_skipped > 0 and not (terminated or truncated):
                        reward = torch.Tensor([0])
                    self.rollout_storage.insert(current_state, np.array([0]), action, action_log_prob, state_value, reward, np.array([1-terminated]))
                    current_state = next_state.copy()
                    step_counts += 1
                    if terminated or truncated:
                        done = True
                        self.total_num_explored_designs = len(set(self.indices))
                        print("Done! Total num of simulated designs: {}".format(self.total_num_explored_designs))
                        break
                # If the next state isn't the last state of the episode, `last_option_value` is useful for the returns.
                # But if the next state is the last state, `last_option_value` can be anything because it will be masked when computing the returns.
                if self.embedding:
                    last_state_embedding = self.env.dataset.microidx_2_microembedding(torch.tensor(next_state).to(self.device))
                    last_state_embedding = last_state_embedding.flatten(-2, -1)
                    last_state_value = self.critic(last_state_embedding)
                    # Record entropy
                    actor_entropy = self.compute_actor_entropy(current_state_embed, action_mask)
                else:
                    last_state_value = self.critic(torch.tensor(next_state).to(self.device))
                    # Record entropy
                    actor_entropy = self.compute_actor_entropy(current_state, action_mask)
                # If done, then record entropy.
                if done:
                    self.entropy.append(actor_entropy.item())
            self.rollout_storage.normalize_rewards()
            self.rollout_storage.compute_returns(last_state_value, gamma=self.gamma, use_gae=True)
            # self.rollout_storage.normalize_returns()
            self.rollout_storage.to_device(device=self.device)
            self.update_model(self.rollout_storage)
            self.rollout_storage.reset_buffer()
            self.rollout_storage.to_device(device='cpu')  

    def select_action(self, state, action_mask):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask).to(self.device)
        # Get action
        action_probs = self.actor(state, action_mask)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)
        return action.numpy(), action_log_prob

    def update_model(self, rollouts):
        states, options, actions, action_log_probs, value_preds, rewards, returns, masks = rollouts.get_buffer()
        states = states.view(-1, states.shape[-1])
        for i in range(self.num_epochs):
            action_mask = self.env.generate_action_mask(states[:-1].numpy())
            action_mask = torch.tensor(action_mask).view(options.shape[0], -1)
            if self.embedding:
                states_embedding = self.env.dataset.microidx_2_microembedding(states)
                states_embedding = states_embedding.flatten(-2, -1)
                new_action_probs = self.actor(states_embedding[:-1], action_mask)
                new_state_values = self.critic(states_embedding[:-1])
            else:
                new_action_probs = self.actor(states[:-1], action_mask)
                new_state_values = self.critic(states[:-1])
            new_action_log_probs = torch.log(new_action_probs.gather(1, actions))

            ratio = torch.exp(new_action_log_probs - action_log_probs.detach())
            adv = returns[:-1].detach() - value_preds[:-1].detach()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy loss
            entropy = -self.entropy_coef * torch.sum(new_action_probs * new_action_log_probs, dim=1).mean()
            # Critic loss
            critic_loss = 0.5 * nn.functional.mse_loss(returns[:-1].detach(), new_state_values)
            total_loss = actor_loss + critic_loss - entropy

            # Update parameters
            self.optimizer.zero_grad()
            if self.embedding:
                self.embed_optim.zero_grad()
            total_loss.backward()
            
            self.optimizer.step()
            if self.embedding:
                self.embed_optim.step()
        # self.optim_schedule.step()

    def compute_actor_entropy(self, state, action_mask):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask).to(self.device)
            # Get action
            action_probs = self.actor(state, action_mask)
        action_distribution = Categorical(action_probs)
        return action_distribution.entropy()
