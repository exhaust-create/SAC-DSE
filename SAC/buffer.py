import numpy as np
import torch
from torch.distributions import Categorical

class ReplayBuffer:
    def __init__(self, state_dim, max_size, batch_size, beta=0.1):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0
        self._init_beta = beta
        self.beta = beta
 
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.option_memory = np.zeros((self.mem_size, ))
        self.action_memory = np.zeros((self.mem_size, ))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.return_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=bool)

        self.td_error_memory = np.zeros((self.mem_size, ))
        self.probs_memory = np.zeros((self.mem_size, ))         # Probabilities for all transitions.
        self.IS_weight_memory = np.ones((self.mem_size, ))     # For importance sampling. Used in value loss function.
 
    def store_transition(self, state, option, action, reward, cumsum_return, state_, done, td_error):
        mem_idx = self.mem_cnt % self.mem_size
 
        self.state_memory[mem_idx] = state
        self.option_memory[mem_idx] = option
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.return_memory[mem_idx] = cumsum_return
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        self.td_error_memory[mem_idx] = td_error
 
        self.mem_cnt += 1

    def compute_return_mean_std(self):
        self._return_mean = self.return_memory.mean()
        self._return_std = self.return_memory.std()
 
    def sample_buffer(self, priority=False, batch_size=None):
        """
        Description
        -------
        If `priority=True`, please implement the functioin `compute_probs` first.

        Parameters
        -------
        priority: bool
            Whether to use rank-based priority sampliing. 
            If False, then no indices of samples returned. If True, return the indices of samples for further TD error update.

        Returns
        -------
        batch_idx: np.array
            Indices of samples if `priority=True`.
        """
        batch_size = self.batch_size if batch_size is None else batch_size

        mem_len = min(self.mem_size, self.mem_cnt)
        if not priority:
            batch_idx = np.random.choice(mem_len, batch_size)
    
            # states = self.state_memory[batch_idx]
            # options = self.option_memory[batch_idx]
            # actions = self.action_memory[batch_idx]
            # rewards = self.reward_memory[batch_idx]
            # returns = self.return_memory[batch_idx]
            # states_ = self.next_state_memory[batch_idx]
            # terminals = self.terminal_memory[batch_idx]
        else:
            dist = Categorical(torch.tensor(self.probs_memory)[:mem_len])
            indices = dist.sample((batch_size,))
            batch_idx = indices.numpy()

        states = self.state_memory[batch_idx]
        options = self.option_memory[batch_idx]
        actions = self.action_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        returns = self.return_memory[batch_idx]
        states_ = self.next_state_memory[batch_idx]
        terminals = self.terminal_memory[batch_idx]
        loss_weight = self.IS_weight_memory[batch_idx]
        return batch_idx, states, options, actions, rewards, returns, states_, terminals, loss_weight
    
    def compute_probs(self, beta=None):
        """
        Description
        -------
        1. Use rank-based priority replay buffer is True, otherwise use uniform sampling.
        2. Compute the probability of each sample for importance sampling and the IS weights.
        """
        if beta is None:
            beta = self.beta
        mem_len = min(self.mem_size, self.mem_cnt)
        indices = np.argsort(-np.abs(self.td_error_memory[:mem_len]))
        rank = np.arange(mem_len) + 1
        self.probs_memory[indices] = (1/rank)/((1/rank).sum())
        IS_weight = np.power(mem_len*self.probs_memory + 1e-16, -beta)
        self.IS_weight_memory[:mem_len] = IS_weight[:mem_len]/IS_weight[:mem_len].max()  # Normalize the IS weights.
    
    def increase_beta(self, step):
        beta = np.tanh(step/200)
        self.beta = max(self._init_beta, beta)
        # print("IS weight beta:", self.beta)
    
    def update_td_error(self, batch_idx, td_error):
        """
        Parameters
        -------
        batch_idx: int
            The indices given by the function `sample_buffer`.
        """
        self.td_error_memory[batch_idx] = td_error
 
    def ready(self):
        return self.mem_cnt > self.batch_size
