import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os, sys

sys.path.append('..')
from networks.Networks import Actor_NN_no_option, Q_Critic_NN
from buffer.storage import RolloutStorage
from .buffer import ReplayBuffer

"""
Best reward_scale: 10
Best tau: 
"""

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.01
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, config, env, device='cpu', embedding=True, buffer_TD_error='max', priority=True, k_step_update=None, gamma=None, tau=None,
                 batch_size=None, alpha=None):
        self.config = config
        self.env = env
        self.device = device
        self.embedding = embedding
        self.k_step_update = config["k_step_update"] if k_step_update is None else k_step_update  # If set it to a very large num, then an episode only stop when `terminated/truncated=True`.
        self.frame_skipped = config["frame_skipped"]
        self.batch_size = config["batch_size"] if batch_size is None else batch_size
        self.train_iter = config["num_epochs"]  # The num of training epochs of a minibatch.
        self.lr_gamma = 0.5
        self.gamma = config["gamma"] if gamma is None else gamma
        self.alpha = config["alpha"] if alpha is None else alpha
        self.tau = config["tau"] if tau is None else tau
        self.reward_scale = config["reward_scale"]  # For SAC, it's better to multiply the rewards with a coef `reward_scale`.
        self.mem_size = config["mem_size"]
        self.lr = config["lr"]
        self.buffer_TD_error = buffer_TD_error
        self.priority = priority
        self._epsilon_greedy = 0.3
        if embedding:
            self._state_embed_dim = env.dataset.microarch_embedding.embedding_dim * env.observation_space.n  # We use the embedding of the state as the input of networks.
            # 策略网络
            self.actor = Actor_NN_no_option(state_dim = self._state_embed_dim, output_dim = self.env.action_space.n, fc_dim=128, device=device)
            # self.actor.apply(init_weights)
            # The outputs of Q_1 & Q_2 should be multiplied with action_masks outside the networks. 
            # 第一个Q网络
            self.critic_1 = Q_Critic_NN(input_dim = self._state_embed_dim,
                                    output_dim = self.env.action_space.n,
                                    hidden_dim=256,
                                    device=device)
            # self.critic_1.apply(init_weights)
            # 第二个Q网络
            self.critic_2 = Q_Critic_NN(input_dim = self._state_embed_dim,
                                    output_dim = self.env.action_space.n,
                                    hidden_dim=256,
                                    device=device)
            # self.critic_2.apply(init_weights)
            self.target_critic_1 = Q_Critic_NN(input_dim = self._state_embed_dim,
                                    output_dim = self.env.action_space.n,
                                    hidden_dim=256,
                                    device=device)  # 第一个目标Q网络
            self.target_critic_2 = Q_Critic_NN(input_dim = self._state_embed_dim,
                                    output_dim = self.env.action_space.n,
                                    hidden_dim=256,
                                    device=device)  # 第二个目标Q网络
        else:
            self.actor = Actor_NN_no_option(state_dim = env.observation_space.n, output_dim = self.env.action_space.n, fc_dim=128, device=device)
            self.critic_1 = Q_Critic_NN(input_dim=env.observation_space.n, output_dim = self.env.action_space.n, hidden_dim=256, device=device)
            self.critic_2 = Q_Critic_NN(input_dim=env.observation_space.n, output_dim = self.env.action_space.n, hidden_dim=256, device=device)
            self.target_critic_1 = Q_Critic_NN(input_dim=env.observation_space.n, output_dim = self.env.action_space.n, hidden_dim=256, device=device)
            self.target_critic_2 = Q_Critic_NN(input_dim=env.observation_space.n, output_dim = self.env.action_space.n, hidden_dim=256, device=device)
        
        # # # # # # # # # 
        #   N-step buffer: This buffer only used to store the n-step transitions.
        #   RolloutStorage.rewards: Only means the reward of each step.
        #   RolloutStorage.returns: Stores the cumulative return of each step.
        # # # # # # # # #
        self.n_step_buffer = RolloutStorage(mem_size=self.k_step_update, obs_shape=env._first_microarch_comp.shape, action_space=env.action_space)
        
        # # # # # # # # # 
        #   Replay Buffer: This buffer is used to store all transitions that was once stored in the `self.n_step_buffer`. 
        #  # # # # # # # # 
        self.all_transitions = ReplayBuffer(state_dim=env.observation_space.n, max_size=self.mem_size, batch_size=self.batch_size)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(params = self.actor.parameters(), lr=self.lr)
        # self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=self.lr)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=40, gamma=0.7)

        self.critic_1_optimizer = optim.Adam(params = self.critic_1.parameters(), lr=2*self.lr)
        # self.critic_1_optimizer = optim.SGD(self.critic_1.parameters(), lr=2*self.lr)
        self.critic_1_scheduler = optim.lr_scheduler.StepLR(self.critic_1_optimizer, step_size=40, gamma=self.lr_gamma)

        self.critic_2_optimizer = optim.Adam(params = self.critic_2.parameters(), lr=2*self.lr)
        # self.critic_2_optimizer = optim.SGD(self.critic_2.parameters(), lr=2*self.lr)
        self.critic_2_scheduler = optim.lr_scheduler.StepLR(self.critic_2_optimizer, step_size=40, gamma=self.lr_gamma)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.log_alpha_optimizer = optim.Adam(params = [self.log_alpha], lr=1e-4)
        # self.log_alpha_optimizer = optim.SGD([self.log_alpha], lr=1e-4)
        self.log_alpha_scheduler = optim.lr_scheduler.StepLR(self.log_alpha_optimizer, step_size=5, gamma=0.8)

        if embedding:
            self.state_embed_optimizer = optim.Adam(env.dataset.microarch_embedding.parameters(), lr=self.lr)
            # self.state_embed_optimizer = optim.SGD(env.dataset.microarch_embedding.parameters(), lr=self.lr)
            self.state_embed_scheduler = optim.lr_scheduler.StepLR(self.state_embed_optimizer, step_size=5, gamma=0.8)

        self.target_entropy = -torch.tensor(0.98*np.log(1/self.env.action_space.n))

        self.total_num_explored_designs = 0
        self.update_count = 0
        self.indices = []
        self.critic_1_loss, self.critic_2_loss, self.actor_loss = [], [], []
        self.target_value, self.entropy = [], []
    
    def save(self):
        torch.save(self.actor.state_dict(),'net.pdparams')
    
    def run_an_episode(self):
        done = False
        terminated = False
        truncated = False
        current_state, info = self.env.reset()  
        while not done:
            step_counts = 0
            self.n_step_buffer.observations[0].copy_(torch.tensor(current_state).to(self.device))
            with torch.no_grad():
                # Collect k-step transitions.
                while step_counts < self.k_step_update:
                    # Get the action mask for the current state
                    action_mask = self.env.generate_action_mask(current_state.astype(int))
                    if self.embedding:
                        current_state_embed = self.env.dataset.microidx_2_microembedding(torch.tensor(current_state).to(self.device))
                        action = self.select_action(current_state_embed, action_mask)
                    else:
                        action = self.select_action(current_state, action_mask)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)

                    # For calculation of the num of simulated designs.
                    micro_idx = self.env.locate_microarch_idx(next_state)
                    self.indices.append(micro_idx)

                    # Skip several frames to reduce data collection.
                    if (step_counts + 1) % self.frame_skipped > 0 and not (terminated or truncated):
                        reward = torch.Tensor([0])
                    # NOTE: Here the reward is multiplied with a coef, so be careful if you use the reward stored to compute the final episode reward.
                    reward = self.reward_scale * reward
                    
                    self.n_step_buffer.insert(next_state, np.array([0]), action, np.array([0]), np.array([0]), reward, np.array([1-terminated]))
                    current_state = next_state.copy()
                    step_counts += 1
                    if terminated or truncated:
                        done = True
                        self.total_num_explored_designs = len(set(self.indices))
                        print("Done! Total num of simulated designs: {}".format(self.total_num_explored_designs))
                        break

                # Record entropy
                if self.embedding:
                    actor_entropy = self.compute_actor_entropy(current_state_embed, action_mask)
                else:
                    actor_entropy = self.compute_actor_entropy(current_state, action_mask)
                # If done, then record the actor's entropy.
                if done:
                    self.entropy.append(actor_entropy.item())

                # Compute cumulative returns, TD errors and store the k-step transitions.
                state_value = self.calc_state_value(next_state)
                self.target_value.append(state_value.item() + self.gamma*reward)
                # self.n_step_buffer.normalize_rewards()
                self.n_step_buffer.compute_returns(state_value, self.gamma, use_gae=False)

                td_error_1, td_error_2 = self.compute_td_error(states=self.n_step_buffer.observations[:step_counts], 
                                                            actions=self.n_step_buffer.actions[:step_counts], 
                                                            returns=self.n_step_buffer.returns[:step_counts])
                # NOTE: I don't know whether to choose the MIN, MAX or MEAN of these two TD errors.
                if self.buffer_TD_error=='max':
                    td_error = torch.max(td_error_1, td_error_2)
                elif self.buffer_TD_error=='mean':
                    td_error = (td_error_1 + td_error_2)/2
                elif self.buffer_TD_error=='min':
                    td_error = torch.min(td_error_1, td_error_2)
                else:
                    raise Exception("Please enter \"max\", \"mean\" or \"min\" for the hyperparameter \"buffer_TD_error\"!")
                # td_error = np.repeat(np.abs(self.all_transitions.td_error_memory).max(),step_counts).reshape(-1,1)
                # td_error = torch.tensor(td_error).to(self.device)

            # Copy the transitions from the n-step buffer to the replay buffer. 
            # `min(self.k_step_update,step_counts)` in case that `terminal` occurs before the kth step comes.
            for i in range(min(self.k_step_update,step_counts)):
                self.all_transitions.store_transition(
                    state   = self.n_step_buffer.observations[i].cpu().numpy(),
                    option  = 0,
                    action  = self.n_step_buffer.actions[i].cpu().numpy(),
                    reward  = self.n_step_buffer.rewards[i].cpu().numpy(),
                    cumsum_return=self.n_step_buffer.returns[i].cpu().numpy(),
                    state_  = self.n_step_buffer.observations[i+1].cpu().numpy(),
                    done = 1 - self.n_step_buffer.masks[i].cpu().numpy(),
                    td_error = td_error[i].cpu().numpy()
                    )
            self.n_step_buffer.reset_buffer()
            self.all_transitions.compute_probs()
            if self.all_transitions.ready():
                self.update_model()
                # lr = self.critic_1_scheduler.get_lr()
                # print("lr: {}".format(lr))
                self.all_transitions.increase_beta(step=self.update_count)
                self.update_count += 1

    def select_action(self, state, action_mask):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask).to(self.device)
        state = state.flatten(-2, -1)
        # self.noise_std = self.noise_std*0.95
        probs = self.actor(state, action_mask)
        if (probs<0).any():
            raise Exception("At least one of the Probs is less than 0 !")

        action_dist = Categorical(probs)
        action = action_dist.sample()
        action = action.numpy()

        # rand = np.random.uniform(0,1)
        # num = np.arange(self.env.action_space.n)
        # num = num[action_mask[0].cpu().numpy()]
        # if rand < self._epsilon_greedy:
        #     action = np.random.choice(num).reshape(-1)

        return action

    # 计算state value,直接用策略网络的输出概率进行期望计算
    def calc_state_value(self, next_states):
        if self.embedding:
            next_states_embed = self.env.dataset.microidx_2_microembedding(torch.tensor(next_states).to(self.device))
            next_states_embed = next_states_embed.flatten(-2, -1)
        action_mask = self.env.generate_action_mask(next_states.astype(int))
        action_mask = torch.tensor(action_mask).to(self.device)
        if self.embedding:
            next_probs = self.actor(next_states_embed, action_mask)
        else:
            next_probs = self.actor(torch.tensor(next_states).to(self.device), action_mask)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, axis=1, keepdim=True)

        if self.embedding:
            q1_value = self.target_critic_1(next_states_embed)
            q2_value = self.target_critic_2(next_states_embed)
        else:
            q1_value = self.target_critic_1(torch.tensor(next_states).to(self.device))
            q2_value = self.target_critic_2(torch.tensor(next_states).to(self.device))
        q1_value = q1_value * action_mask
        q2_value = q2_value * action_mask

        min_q_value = torch.min(q1_value, q2_value)
        # critic_idx = torch.argmin(torch.concat([q1_value, q2_value]))
        min_state_value = torch.sum(next_probs * min_q_value,
                               axis=1,
                               keepdim=True)
        next_value = min_state_value + self.log_alpha.exp() * entropy    # This is the state value.
        # td_target = rewards + self.gamma * next_value * (1 - dones)
        return next_value
    
    def compute_td_error(self, states, actions, returns):
        """
        Returns
        -------
        td_error_1:
            The TD error of the first Q network.
        td_error_2: 
            The TD error of the second Q network.
        """
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions).long().to(self.device)
        if not isinstance(returns, torch.Tensor):
            returns = torch.tensor(returns).to(self.device)
        if self.embedding:
            states_embed = self.env.dataset.microidx_2_microembedding(states)
            critic_1_q_values = torch.gather(self.critic_1(states_embed), 1, actions)
            critic_2_q_values = torch.gather(self.critic_2(states_embed), 1, actions)
        else:
            critic_1_q_values = torch.gather(self.critic_1(states), 1, actions)
            critic_2_q_values = torch.gather(self.critic_2(states), 1, actions)
        td_error_1 = returns - critic_1_q_values + 1e-5
        td_error_2 = returns - critic_2_q_values + 1e-5
        
        return td_error_1, td_error_2

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target * (1.0 - self.tau) + param * self.tau)

    # def update_returns(self):
    #     """
    #     Description
    #     -------
    #     To devoid the inaccuracy of the old target Q functions, the returns of the existing returns should be updated after several times of "k-step update".
    #     """
    #     for i in range(self.all_transitions.mem_cnt//self.k_step_update+1):
    #         self.n_step_buffer.observations[0].copy_(torch.tensor(self.all_transitions.state_memory[i*self.k_step_update]).to(self.device))
    #         for j in range(self.k_step_update):
    #             # Copy k_step_update transitions to `n_step_buffer`.
    #             self.n_step_buffer.insert(obs=self.all_transitions.next_state_memory[i*self.k_step_update+j], 
    #                                       option=0, 
    #                                       action=self.all_transitions.action_memory[i*self.k_step_update+j], 
    #                                       action_log_prob=0, 
    #                                       value_pred=0, 
    #                                       reward=self.all_transitions.return_memory[i*self.k_step_update+j], 
    #                                       mask=1-self.all_transitions.terminal_memory[i*self.k_step_update+j]
    #                                       )
    #             if self.all_transitions.terminal_memory[i*self.k_step_update+j] == 1:
    #                 break
    #         last_state = self.n_step_buffer.observations[-1].cpu().numpy()
    #         last_state_value = self.calc_state_value(last_state)
    #         self.n_step_buffer.compute_returns(last_state_value)
    #         # Copy the updated returns from `n_step_buffer` to `all_transitions`.
    #         for j in range(self.k_step_update):
    #             self.all_transitions.return_memory[i*self.k_step_update+j] = self.n_step_buffer.returns[j].cpu().numpy().copy()
    #         self.n_step_buffer.reset_buffer()

    def update_model(self):
        k =  1 + self.all_transitions.mem_cnt/400
        train_iter = int(k * self.train_iter)
        batch_size = int(k * self.batch_size)
        # print("batch_size: ", batch_size)
        self.all_transitions.compute_return_mean_std()
        
        for _ in range(train_iter):
            # Sampling
            indices, states, options, actions, rewards, returns, next_states, terminals, IS_weights = self.all_transitions.sample_buffer(priority=self.priority, batch_size=batch_size)
            actions = actions.reshape(actions.shape[-1],1)
            returns = returns.reshape(returns.shape[-1], 1)
            returns = (returns-self.all_transitions._return_mean)/self.all_transitions._return_std*2
            IS_weights = IS_weights.reshape(IS_weights.shape[-1], 1)
            action_masks = self.env.generate_action_mask(states.astype(int))
            action_masks = torch.tensor(action_masks).to(self.device)
            if self.embedding:
            # Get state embeddings and action masks.
                states_embed = self.env.dataset.microidx_2_microembedding(torch.tensor(states).to(self.device))
                states_embed = states_embed.flatten(-2, -1)

            # Compute the TD errors.
            td_error_1, td_error_2 = self.compute_td_error(states, actions, returns)
            # Get the MEAN of the TD errors.
            critic_1_loss = torch.mean(torch.pow((td_error_1), 2) * torch.tensor(IS_weights).to(self.device))
            critic_2_loss = torch.mean(torch.pow((td_error_2), 2) * torch.tensor(IS_weights).to(self.device))
            self.critic_1_loss.append(critic_1_loss.item())
            self.critic_2_loss.append(critic_2_loss.item())
            # print("critic_loss_1: {}, critic_loss_2: {}".format(critic_1_loss, critic_2_loss))

            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            if self.embedding:
                self.state_embed_optimizer.zero_grad()

            critic_1_loss.backward(retain_graph=True)
            critic_2_loss.backward(retain_graph=True)

            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

            if self.embedding:
                # 更新策略网络
                probs = self.actor(states_embed, action_masks)
            else:
                probs = self.actor(torch.tensor(states).to(self.device), action_masks)
            log_probs = torch.log(probs + 1e-8)
            # 直接根据概率计算熵
            entropy = -torch.sum(probs * log_probs, axis=1, keepdim=True)
            if self.embedding:
                q1_value = self.critic_1(states_embed)
                q2_value = self.critic_2(states_embed)
            else:
                q1_value = self.critic_1(torch.tensor(states).to(self.device))
                q2_value = self.critic_2(torch.tensor(states).to(self.device))
            q1_value = q1_value * action_masks
            q2_value = q2_value * action_masks
            min_qvalue = torch.sum(probs * torch.minimum(q1_value, q2_value), axis=1, keepdim=True)  # 直接根据概率计算期望
            # Get the actor's loss.
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
            self.actor_loss.append(actor_loss.item())
            # print("actor_loss: ", actor_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.embedding:
                self.state_embed_optimizer.step()

            # 更新alpha值
            alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # Updata the TD error in the replay buffer.
        # Only update the TD errors of the last sampled transitions.
        with torch.no_grad():
            td_error_1, td_error_2 = self.compute_td_error(states, actions, returns)
            # NOTE: I don't know whether to choose the MIN, MAX or MEAN of these two TD errors.
            if self.buffer_TD_error=='max':
                td_error = torch.max(td_error_1, td_error_2)
            elif self.buffer_TD_error=='mean':
                td_error = (td_error_1 + td_error_2)/2
            elif self.buffer_TD_error=='min':
                td_error = torch.min(td_error_1, td_error_2)
            else:
                raise Exception("Please enter \"max\", \"mean\" or \"min\" for the hyperparameter \"buffer_TD_error\"!")
            self.all_transitions.update_td_error(indices, td_error.flatten().numpy())

        # self.actor_scheduler.step()
        # self.critic_1_scheduler.step()
        # self.critic_2_scheduler.step()
        # if self.embedding:
        #     self.state_embed_scheduler.step()
        # self.log_alpha_scheduler.step()

    def compute_actor_entropy(self, state, action_mask):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask).to(self.device)
        state = state.flatten(-2, -1)
        probs = self.actor(state, action_mask)
        actor_distribution = Categorical(probs)
        return actor_distribution.entropy()