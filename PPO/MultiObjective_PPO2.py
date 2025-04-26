import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pdb

import os, sys

sys.path.append('..')
from networks.Networks import Q_Critic_NN
from buffer.storage import RolloutStorage

class Actor_NN(nn.Module):
    def __init__(self, state_dim, output_dim, fc_dim=None, device='cpu'):
        super(Actor_NN, self).__init__()
        self.device = device
        self.output_dim = torch.tensor(output_dim)
        fc_dim = 4*state_dim if fc_dim is None else fc_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state, action_mask, temperature=None):
        """
        Args
        -------
        temperature:
            Should be a torch.Tensor.
        """
        temp = torch.sqrt(self.output_dim) if temperature is None else temperature
        x = F.elu(self.fc1(state.float()))
        x = F.elu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = action_probs + ~action_mask*(-1e9)
        action_probs = F.softmax(action_probs/temp, dim=1)
        action_probs = action_probs + action_mask*1e-9
        return action_probs

class MultiObjective_PPO(nn.Module):
    """
    Description
    -------
    This `PPO2_v0` use `Actor_NN_no_option` as the actor.
    """
    def __init__(self, config, env, max_episodes, device='cpu'):
        super(MultiObjective_PPO, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.k_step_update = config["k_step_update"]    # If set it to a very large num, then an episode only stop when `terminated/truncated=True`.
        self.num_epochs = config["num_epochs"]  # The num of training epochs of a minibatch.
        self.clip_param = config["clip_param"]
        self.entropy_coef = config["entropy_coef"]
        self.lr = config["lr"]
        self.max_episodes = max_episodes
        self.preference = np.array([[0.2,0.1,0.7],[0.3,0.6,0.1],[0.7,0.2,0.1]])
        self.lr_gamma = 0.98
        self._temperature = 10
        self._beta = 0.5
        self.rollout_storage = RolloutStorage(self.k_step_update, self.env.observation_space.nvec, self.env.action_space, 
                                              reward_space=self.preference.shape[1], value_space=self.preference.shape[1],return_space=self.preference.shape[1])
        self.actor = Actor_NN(state_dim = env.option_space.n + self.preference.shape[1], output_dim = self.env.action_space.n, fc_dim=128, device=device)
        # self.actor.apply(init_weights)
        self.critic = Q_Critic_NN(input_dim = env.option_space.n + self.preference.shape[1], output_dim = self.preference.shape[1], hidden_dim=256, device=device)
        # self.critic.apply(init_weights)
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.lr},
                                    {'params': self.critic.parameters(), 'lr': self.lr}],
                                    )
        # self.optim_schedule = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        
        self.total_num_explored_designs = 0
        self.entropy = []

    def run_an_episode(self):
        """
        Description
        -------
        Run an episode and collect the data.
        """
        terminated = False
        truncated = False
        done = False
        step_counts = 0
        current_state, info = self.env.zero_reset()
        self.rollout_storage.observations[0].copy_(torch.tensor(current_state).to(self.device))
        # Randomly pick a vector from "self.preference"
        pref_idx = np.random.choice(len(self.preference))
        preference = self.preference[pref_idx].reshape(1,-1)
        while not done:
            while step_counts < self.k_step_update and not truncated:
                current_state_with_pref = np.concatenate((current_state, preference),axis=-1)   # Concatenate state & pref.
                state_value = self.critic(torch.tensor(current_state_with_pref).to(self.device))
                option = torch.tensor(step_counts%8 + 2).view(1,1).to(self.device)
                action_mask = self.env.generate_action_mask(current_state, option.detach().cpu().numpy())
                action_mask = action_mask.reshape(option.shape[0],-1)
                action, action_log_prob = self.select_action(torch.tensor(current_state_with_pref).to(self.device), action_mask)
                # Store trajectory
                # print("component_idx: {}, action: {}".format(option.cpu().detach().numpy(), action.cpu().detach().numpy()))
                next_state, reward, terminated, truncated, info = self.env.step(option, action)     # reward: [1,3]
                next_state_with_pref = np.concatenate((next_state, preference),axis=-1)   # Concatenate state & pref.
                self.rollout_storage.insert(next_state, option.view(-1), action, action_log_prob, state_value, reward, 1-torch.tensor(terminated).int()) # Don't use `value_pred`.
                current_state = next_state.copy()
                step_counts += 1
                if terminated or truncated:
                    self.total_num_explored_designs += 1    # If `zero_reset()`, then plus 1 at each terminal, otherwise plus `len(self.env._explored_designs)`.
                    # print("Num of explored_designs: {}, total num of explored_design: {}".format(len(self.env._explored_designs), self.total_num_explored_designs))
                    done = True
                    break
            # Record entropy
            if done:
                actor_entropy = self.compute_actor_entropy(torch.tensor(current_state_with_pref).to(self.device), action_mask)
                self.entropy.append(actor_entropy)

            # If the next state isn't the last state of the episode, `last_option_value` is useful for the returns.
            # But if the next state is the last state, `last_option_value` can be anything because it will be masked when computing the returns.
            # If we don't use GAE, the returns will be `cumsum(all_rewards)`.
            # Then the target of the critic will be fitting the relationship between states and the cumulative rewards.
            last_state_value = self.critic(torch.tensor(next_state_with_pref).to(self.device))
            # self.rollout_storage.normalize_rewards()
            self.rollout_storage.compute_returns(last_state_value, gamma=self.config["gamma"], use_gae=True)
            self.rollout_storage.to_device(device=self.device)
            self.update_model(self.rollout_storage, preference)
            self.temperature_decay()
            self.rollout_storage.reset_buffer()
            step_counts = 0
            self.rollout_storage.to_device(device='cpu')

    def select_action(self, state_with_pref, action_mask):
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask).to(self.device)
        # Get action
        action_probs = self.actor(state_with_pref, action_mask, self._temperature)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)
        return action, action_log_prob
    
    def temperature_decay(self):
        self._temperature = 0.01 + 0.99*self._temperature

    def compute_actor_entropy(self, state_with_pref, action_mask):
        with torch.no_grad():
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask).to(self.device)
            # Get action
            action_probs = self.actor(state_with_pref, action_mask, self._temperature)
            action_distribution = Categorical(action_probs)
        return action_distribution.entropy().item()
    
    def update_model(self, rollouts, preference):
        states, options, actions, action_log_probs, value_preds, rewards, returns, masks = rollouts.get_buffer()
        states = states.view(-1, states.shape[-1])
        for i in range(self.num_epochs):
            action_mask = self.env.generate_action_mask(states[:-1].detach().cpu().numpy(),options.detach().cpu().numpy())
            action_mask = action_mask.reshape(options.shape[0], -1)
            action_mask = torch.tensor(action_mask).to(self.device)

            # Repeat the preference
            preference = torch.tensor(preference).to(torch.float).clone().detach()
            pref_repeated = preference.repeat(states.shape[0], 1)
            states_with_pref = torch.cat((states, pref_repeated), dim=1)

            new_action_probs = self.actor(states_with_pref[:-1], action_mask, self._temperature)
            new_action_log_probs = torch.log(new_action_probs.gather(1, actions))

            new_state_values = self.critic(states_with_pref[:-1])

            ratio = torch.exp(new_action_log_probs - action_log_probs.detach())
            adv = returns[:-1].detach() - value_preds[:-1].detach()
            adv_w = torch.matmul(adv, preference.view(-1))  # Return shape[n,]
            adv_w = (adv_w - adv_w.mean())/(adv_w.std() + 1e-30)

            surr1 = ratio * adv_w
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_w
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy loss
            entropy = -self.entropy_coef * torch.sum(new_action_probs * new_action_log_probs, dim=1).mean()
            # Critic loss
            critic_loss_1 = nn.functional.mse_loss(torch.matmul(new_state_values,preference.view(-1)),torch.matmul(returns[:-1].detach(),preference.view(-1)))
            critic_loss_2 = nn.functional.mse_loss(new_state_values, returns[:-1].detach())
            critic_loss = 0.5*(self._beta*critic_loss_1 + (1-self._beta)*critic_loss_2)
            total_loss = actor_loss + critic_loss - entropy
            # print("actor_loss: {}, critic_loss: {}, total_loss: {}".
            #       format(actor_loss.detach().cpu().item(), 
            #              critic_loss.detach().cpu().item(),
            #              total_loss.detach().cpu().item()))

            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def schedule_lr(self, episode):
        self.lr = self.lr - \
            (episode / self.max_episodes) * self.lr
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr

    def test(self):
        last_states, norm_ppas, projs = [], [], []
        for preference in self.preference:
            preference = preference.reshape(1, -1)
            terminated = False
            truncated = False
            done = False
            step_counts = 0
            current_state, info = self.env.zero_reset()
            while not done:
                while step_counts < self.k_step_update and not truncated:
                    current_state_with_pref = np.concatenate((current_state, preference),axis=-1)   # Concatenate state & pref.
                    option = torch.tensor(step_counts%8 + 2).view(1,1).to(self.device)
                    action_mask = self.env.generate_action_mask(current_state, option.detach().cpu().numpy())
                    action_mask = action_mask.reshape(option.shape[0],-1)
                    action, action_log_prob = self.select_action(torch.tensor(current_state_with_pref).to(self.device), action_mask)
                    # Store trajectory
                    # print("component_idx: {}, action: {}".format(option.cpu().detach().numpy(), action.cpu().detach().numpy()))
                    next_state, reward, terminated, truncated, info = self.env.step(option, action)     # reward: [1,3]
                    current_state = next_state.copy()
                    step_counts += 1
                    if terminated or truncated:
                        # self.total_num_explored_designs += 1    # If `zero_reset()`, then plus 1 at each terminal, otherwise plus `len(self.env._explored_designs)`.
                        # print("Num of explored_designs: {}, total num of explored_design: {}".format(len(self.env._explored_designs), self.total_num_explored_designs))
                        done = True
                        break
            proj = self.env.compute_projection(reward, preference.reshape(-1))
            projs.append(proj)
            last_states.append(current_state)
            norm_ppas.append(reward)
        projs = np.array(projs).reshape(len(self.preference))     # shape(num_of_pref)
        last_states = np.array(last_states).reshape(len(self.preference), -1)     # shape(num_of_pref,n)
        norm_ppas = np.array(norm_ppas)
        original_ppas = self.env.renormalize_ppa(norm_ppas)
        original_ppas = -original_ppas.reshape(len(self.preference), -1)[:,1:]    # shape(num_of_pref, m)
        print("proj_217: {}, proj_361: {}, proj_721: {}".format(projs[0], projs[1], projs[2]))
        return last_states, original_ppas, projs