import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, mem_size, obs_shape, action_space, reward_space=1, value_space=1, return_space=1):
        """
        Description
        -------
        Please `reset_buffer` at the beginning of each episode, then set `self.observations[0]` to the first state of the episode.
        """
        self.value_space = value_space
        self.observations = torch.zeros(mem_size + 1, *obs_shape)
        # self.states = torch.zeros(mem_size + 1, num_processes, state_size)
        self.rewards = torch.zeros(mem_size, reward_space)
        self.value_preds = torch.zeros(mem_size + 1, value_space)
        self.returns = torch.zeros(mem_size + 1, return_space)
        self.options = torch.zeros(mem_size, 1)
        self.action_log_probs = torch.zeros(mem_size, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(mem_size, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(mem_size + 1, 1)
        self.mem_size = mem_size
        self.step = 0
        self.trajectory_start = 0

    def to_device(self, device):
        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.options = self.options.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, option, action, action_log_prob, value_pred, reward, mask):
        """
        Description
        -------
        If the buffer is full, the oldest data will be removed and the newest one will come at the end of the buffer. 
        But `self.step` will increase until we reset the buffer.

        Args
        -------
        mask:
            = 1 - done
        """
        # Check if all args are Tensor. If not, then turn them into Tensor.
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        if not isinstance(option, torch.Tensor):
            option = torch.Tensor(option)
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action)
        if not isinstance(action_log_prob, torch.Tensor):
            action_log_prob = torch.Tensor(action_log_prob)
        if not isinstance(value_pred, torch.Tensor):
            value_pred = torch.Tensor(value_pred)
        if not isinstance(reward, torch.Tensor):
            reward = torch.Tensor(reward)
        if not isinstance(mask, torch.Tensor):
            mask = torch.Tensor(mask)
        
        if self.step >= self.mem_size:
            for i in range(self.mem_size - 1):
                self.observations[i].copy_(self.observations[i + 1])
                self.options[i].copy_(self.options[i + 1])
                self.actions[i].copy_(self.actions[i + 1])
                self.action_log_probs[i].copy_(self.action_log_probs[i + 1])
                self.value_preds[i].copy_(self.value_preds[i + 1])
                self.rewards[i].copy_(self.rewards[i + 1])
                self.masks[i].copy_(self.masks[i + 1])
            self.observations[self.mem_size - 1] = self.observations[self.mem_size]
            self.masks[self.mem_size - 1] = self.masks[self.mem_size]

            self.observations[self.mem_size].copy_(obs)
            self.options[self.mem_size - 1].copy_(option)
            self.actions[self.mem_size - 1].copy_(action)
            self.action_log_probs[self.mem_size - 1].copy_(action_log_prob)
            self.value_preds[self.mem_size - 1].copy_(value_pred)
            self.rewards[self.mem_size - 1].copy_(reward)
            self.masks[self.mem_size].copy_(mask)
        else:
            self.observations[self.step + 1].copy_(obs)
            self.options[self.step].copy_(option)
            self.actions[self.step].copy_(action)
            self.action_log_probs[self.step].copy_(action_log_prob)
            self.value_preds[self.step].copy_(value_pred.view(self.value_space))
            self.rewards[self.step].copy_(reward)
            self.masks[self.step + 1].copy_(mask)

        # self.step = (self.step + 1) % self.mem_size
        self.step = self.step + 1
    
    def normalize_rewards(self):
        with torch.no_grad():
            reward = self.rewards[:min(self.step,self.mem_size)]
            norm_reward = reward - reward.mean()
            self.rewards[:min(self.step, self.mem_size)] = norm_reward.clone()

    def get_buffer(self):
        max_idx = min(self.step, self.mem_size)
        states = self.observations[:max_idx+1].long()
        options = self.options[:max_idx].long()
        actions = self.actions[:max_idx].long()
        action_log_probs = self.action_log_probs[:max_idx]
        value_preds = self.value_preds[:max_idx+1]
        rewards = self.rewards[:max_idx]
        returns = self.returns[:max_idx+1]
        masks = self.masks[:max_idx+1].long()
        return states, options, actions, action_log_probs, value_preds, rewards, returns, masks

    def compute_returns(self, last_value, gamma, use_gae=False, tau=0.95, norm_return=False):
        """
        Description
        -------
        Please call function `normalize_rewards` before calling `compute_returns`.
        Args
        -------
        last_value:
            Can be Q value or state value.
        """
        if use_gae:
            self.value_preds[-1] = last_value
            gae = 0
            for step in reversed(range(self.trajectory_start, min(self.step,self.mem_size))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = last_value
            for step in reversed(range(self.trajectory_start, min(self.step,self.mem_size))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]
        
        if self.step > self.mem_size:
            self.trajectory_start = self.mem_size - self.step + self.trajectory_start + 1
            self.trajectory_start = max(self.trajectory_start, 0 )
        else:
            self.trajectory_start = self.step
    
    def normalize_returns(self):
        return_ = self.returns[:min(self.step, self.mem_size)].clone()
        norm_return = (return_ - return_.mean())/return_.std()
        self.returns[:min(self.step, self.mem_size)] = norm_return.clone()
    
    def sample(self, batch_size, prioritized = False):
        """
        Description:
            Sample a batch of data from the buffer.
        """
        if prioritized:
            pass
        else:
            indices = torch.randint(min(self.step, self.mem_size), batch_size, dtype=torch.int64)
            current_states = self.observations[indices]
            next_states = self.observations[indices + 1]
            options = self.options[indices]
            actions = self.actions[indices]
            action_log_probs = self.action_log_probs[indices]
            value_preds = self.value_preds[indices]
            rewards = self.rewards[indices]
            masks = self.masks[indices + 1]
            returns = self.returns[indices]

        return [indices, current_states, options, actions, action_log_probs, value_preds, rewards, masks, returns, next_states]

    def reset_buffer(self):
        """
        Description:
            Reset the whole buffer.
        """
        self.observations = torch.zeros_like(self.observations)
        self.options = torch.zeros_like(self.options)
        self.actions = torch.zeros_like(self.actions)
        self.action_log_probs = torch.zeros_like(self.action_log_probs)
        self.value_preds = torch.zeros_like(self.value_preds)
        self.rewards = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.returns)
        self.masks = torch.ones_like(self.masks)
        self.step = 0
        self.trajectory_start = 0
