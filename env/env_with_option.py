import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import os, sys

sys.path.append('..')
from dataset.dataset import Dataset

class BOOMENV(gym.Env):
   '''
   Description
   -------
   This class is to generate the environment for BOOM. 
   Parameter `component_space` stores all options for all components. 
   Paremeter `microarch_comp` is used for locating the index of the current state.

   Args
   -------
   config: 
      The yaml file.
   '''
   def __init__(self, config, embed_device='cpu'):
      #  Read configurations
      dataset_path = config['dataset_path']
      design_space_path = config['design_space_path']
      self.preference = np.array(config['preference'])
      self._first_microarch = np.array(config['first_microarch'])  # Set by the user.
      self.max_steps = config['max_steps']
      self.reward_coef = config['reward_coef']
      
      # Load dataset.
      self.dataset = Dataset(dataset_path=dataset_path, design_space_path=design_space_path, embed_device=embed_device)
      self._microarch, self._ppa, self._sim_time = self.dataset.microparam.copy(), self.dataset.ppa.copy(), self.dataset.time.copy()
      self.component_space = self.dataset.component_space.copy()

      # Pre-process data.
      self.microarch_comp = self.dataset.microparam_2_microidx(self._microarch)  # Change all microarch parameter combinations into component combinations.
      self._ppa[:,1:]=-self._ppa[:,1:]   # Take the negative of the "performance".
      self._first_microarch_comp = self.dataset.microparam_2_microidx(self._first_microarch)
      first_microarch_idx = (self.microarch_comp==self._first_microarch_comp).all(axis=1).argmax() # Find the index of the first_microarch in the dataset.
      self._first_microarch_ppa = self._ppa[first_microarch_idx].copy()
      self._ppa_std = self._ppa.std(axis=0)
      self.normalized_ppa = self.normalize_ppa()
      self._norm_first_microarch_ppa = self.normalized_ppa[first_microarch_idx].copy()
      self._action_mask_space = self.__generate_action_mask_space()
      
      
      # Set up hyper-parameters
      self.observation_space = spaces.MultiDiscrete(self._first_microarch_comp.shape)
      self.action_space = spaces.Discrete(np.max(np.sum(~np.isnan(self.component_space),axis=1)))  # The max num of options of all the components.
      self.option_space = spaces.Discrete(len(self.component_space)) # The num of components.
      self.reward_space = 1
      self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)

      # Others
      self._step_counts = 0
      self.num_microarch_range = np.arange(len(self._microarch))
      self._init_reward = 0
   
   def step(self, component_idx, action_idx):
      '''
      Description
      -------
      Outputs the next state and the reward of the current state & action.

      Args
      -------
      component_idx:
         It's the current option. Indicating the idx of the changing component.
      component_value:
         It's the selected action within the current option. The value of the changing component.

      Returns
      -------
      state: 
         Microarch.
      reward: 
         r_current = (PPA_current · preference)/|preference| - alpha*|PPA_current X preference|/|preference| - cumulative_reward
      terminated:
         If a state arrives the best point, then terminate the episode. Only it can take charge of `done`. But here we assume that there isn't a "best point" in the env.
      truncated:
         If the number of episodes arrives at the max number, or the difference between the last reward and the current reward is smaller than `epsilon`, then `truncated` is True.
         It doesn't take charge of `done`.
      info: dict
      '''
      info = {}
      truncated = False
      terminated = False
      # Change the state with `component_idx` & `action_idx`.
      self.state[:,component_idx] = self.component_space[component_idx, action_idx].copy()

      exist_zero = (self.state==np.array(0)).any()
      if exist_zero:
         reward = np.array([0])
      else:
         norm_ppa, reward = self.get_state_reward(self.state)
         
      state = self.state.copy()
      if exist_zero:
         self._explored_designs.append([state, np.array([0,0,0]), reward])
      else:
         self._explored_designs.append([state, norm_ppa, reward])

      self._step_counts += 1
      self._option_selected[component_idx] = True

      # # Check final `proj_with_punishment` every 7 steps. If `proj_with_punishment<0`, then end the episode.
      # if self._step_counts % 7 == 0:
      #    if proj_with_punishment < 0:
      #       terminated = True
      # When all options are selected at least once, or the step counts exceed the `max_steps`, the episode terminates.
      if self._option_selected[2:].all() or self._step_counts > self.max_steps:
      # if self._step_counts >= self.max_steps:
         truncated = True
         terminated = True # Only for PPO_v0
      else:
         truncated = False

      return self.state, reward, terminated, truncated, info
   
   def get_state_reward(self, state):
      """
      Description
      -------
      Calculate the reward of the current state.

      Returns
      -------
      norm_ppa:
         The normalized PPA of the current state.
      reward:
         r_current = (PPA_current · preference)/|preference| - alpha*|PPA_current X preference|/|preference| - cumulative_reward
      """
      state_idx = (self.microarch_comp==state).all(axis=1)   # Find out the state index in the dataset.
      if ~state_idx.any():
         raise Exception("Can't find the state in the dataset!")
      else:
         state_idx = state_idx.argmax()
      # proj_with_punishment = np.dot(self.normalized_ppa[state_idx], self.preference)/np.linalg.norm(self.preference) \
      #    - self.reward_coef*np.linalg.norm(np.cross(self.normalized_ppa[state_idx], self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
      
      proj= np.dot(self.normalized_ppa[state_idx], self.preference)/np.linalg.norm(self.preference)
      proj = np.array([proj])

      # proj_with_punishment = self.normalized_ppa[state_idx, 0].reshape(1).copy()

      # Use the total aggregated reward as the baseline.
      if not self._explored_designs:
         reward = proj - self._init_reward
      else:
         current_sum_reward = np.array([row[2] for i, row in enumerate(self._explored_designs)]).sum()
         reward = proj - current_sum_reward - self._init_reward    # Culculate step reward.
      norm_ppa = self.normalized_ppa[state_idx]
      return norm_ppa, reward
    
   def reset(self,seed=None):
      """
      Description
      -------
      Reset the environment.

      Returns
      -------
      state:
         The initial state.
      info: dict
      """
      super().reset(seed=seed)
      info = {}
      # Store the explored designs. A 2 dims list: [microarch_component, ppa, reward]
      self._explored_designs = []

      # # Randomly sample a data from `self.microarch_comp`.
      # state_mask = self.microarch_comp[:,1]==self._first_microarch_comp[:,1]
      # idx_scale = np.arange(0, len(self.microarch_comp))
      # state_idx = np.random.choice(idx_scale[state_mask])
      # self.state = self.microarch_comp[state_idx].reshape(1, -1).copy()
      # proj_with_punishment = np.dot(self.normalized_ppa[state_idx], self.preference)/np.linalg.norm(self.preference) \
      #       - self.reward_coef*np.linalg.norm(np.cross(self.normalized_ppa[state_idx], self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
      # self._init_reward = proj_with_punishment.copy()

      # # Only select the first microarch as the initial state.
      self.state = self._first_microarch_comp.copy()

      self._step_counts = 0
      self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)
      print("Initial State: ", self.state)
      return self.state,info
   
   def zero_reset(self):
      """
      Description
      -------
      Reset all elements of the initial state to all zeros, except for the first 2 elements.
      """
      info = {}
      # Store the explored designs. A 2 dims list: [microarch_component, ppa, reward]
      self._explored_designs = []

      state = np.zeros_like(self._first_microarch_comp)
      # # Randomly sample a data from `self.microarch_comp`.
      # rand_idx = np.random.randint(0, len(self.microarch_comp))
      # state[:,:2] = self.microarch_comp[rand_idx].reshape(1, -1)[:,:2]

      state[:,:2] = self._first_microarch_comp[:,:2].copy()
      norm_ppa = self._norm_first_microarch_ppa

      # Get the initial reward. Since we want to find a design that is better than the given first design, 
      # we use the PPA of the given design as the initial reward.
      proj_with_punishment = np.dot(norm_ppa, self.preference)/np.linalg.norm(self.preference)
      proj_with_punishment = np.array([proj_with_punishment])
      self._init_reward = proj_with_punishment.copy()

      self.state = state.copy()

      self._step_counts = 0
      self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)
      print("Initial State: ", self.state)
      return self.state,info
   
   def test_reset(self, init_state = None):
      """
      Description
      -------
      When "zero_reset", `init_state` should NOT be overlooked. That means user should input `init_state` on their own.
      """
      info = {}
      # Store the explored designs. A 2 dims list: [microarch_component, ppa, reward]
      self._explored_designs = []

      if init_state is None:
         rand_idx = np.random.randint(0, len(self.microarch_comp))
         state = self.microarch_comp[rand_idx].reshape(1, -1).copy()
      else:
         state = init_state.reshape(1, -1).copy()
      # rand_microarch_ppa = self.normalized_ppa[rand_idx]
      # proj_with_punishment = np.dot(rand_microarch_ppa, self.preference)/np.linalg.norm(self.preference) \
      #                            - self.reward_coef*np.linalg.norm(np.cross(rand_microarch_ppa, self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
      # self._explored_designs.append([state, rand_microarch_ppa, proj_with_punishment])

      self.state = state.copy()
      self._step_counts = 0
      self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)
      
   
   def __generate_action_mask_space(self) -> np.ndarray:
      """
      Returns
      -------
      action_mask_space: <num_sub_design, num_component, max_num_action>
         Mask for all components in all sub-design.
      """
      action_mask_space = ~np.isnan(self.component_space)
      
      # Action mask space for sub-design-1
      action_mask_space_1 = action_mask_space.copy()
      action_mask_space_1[0,1:] = False
      action_mask_space_1[1,1:] = False
      action_mask_space_1[2,3:] = False
      action_mask_space_1[3,3:] = False
      action_mask_space_1[4,4:] = False
      action_mask_space_1[5,3:] = False
      action_mask_space_1[6,3:] = False
      action_mask_space_1[7,4:] = False
      action_mask_space_1[8,4:] = False

      # Action mask space for sub-design-2
      action_mask_space_2 = action_mask_space.copy()
      action_mask_space_2[0,1:] = False
      action_mask_space_2[1,0] = False
      action_mask_space_2[1,2:] = False
      action_mask_space_2[2,:3] = False
      action_mask_space_2[2,6:] = False
      action_mask_space_2[3,:3] = False
      action_mask_space_2[3,6:] = False
      action_mask_space_2[4,:4] = False
      action_mask_space_2[4,7:] = False
      action_mask_space_2[5,:3] = False
      action_mask_space_2[5,6:] = False
      action_mask_space_2[6,:2] = False
      action_mask_space_2[6,5:] = False
      action_mask_space_2[7,4:] = False
      action_mask_space_2[8,4:] = False
      
      # Action mask space for sub-design-3
      action_mask_space_3 = action_mask_space.copy()
      action_mask_space_3[0,:1] = False
      action_mask_space_3[1,:2] = False
      action_mask_space_3[1,3:] = False
      action_mask_space_3[2,:6] = False
      action_mask_space_3[2,9:] = False
      action_mask_space_3[3,:6] = False
      action_mask_space_3[3,9:] = False
      action_mask_space_3[4,:7] = False
      action_mask_space_3[4,10:] = False
      action_mask_space_3[5,:6] = False
      action_mask_space_3[5,9:] = False
      action_mask_space_3[6,:4] = False
      action_mask_space_3[6,7:] = False
      action_mask_space_3[7,:4] = False
      action_mask_space_3[8,:4] = False
      action_mask_space_3[8,7:] = False

      # Action mask space for sub-design-4
      action_mask_space_4 = action_mask_space.copy()
      action_mask_space_4[0,:1] = False
      action_mask_space_4[1,:3] = False
      action_mask_space_4[1,4:] = False
      action_mask_space_4[2,:9] = False
      action_mask_space_4[2,12:] = False
      action_mask_space_4[3,:9] = False
      action_mask_space_4[3,12:] = False
      action_mask_space_4[4,:10] = False
      action_mask_space_4[4,13:] = False
      action_mask_space_4[5,:9] = False
      action_mask_space_4[5,12:] = False
      action_mask_space_4[6,:6] = False
      action_mask_space_4[7,:4] = False
      action_mask_space_4[8,:7] = False

      # Action mask space for sub-design-5
      action_mask_space_5 = action_mask_space.copy()
      action_mask_space_5[0,:1] = False
      action_mask_space_5[1,:4] = False
      action_mask_space_5[2,:12] = False
      action_mask_space_5[3,:11] = False
      action_mask_space_5[4,:13] = False
      action_mask_space_5[5,:10] = False
      action_mask_space_5[6,:6] = False
      action_mask_space_5[7,:4] = False
      action_mask_space_5[8,:7] = False

      # Concatenate all sub mask space
      action_mask_space = np.vstack((np.expand_dims(action_mask_space_1, axis=0),np.expand_dims(action_mask_space_2,axis=0)))
      action_mask_space = np.vstack((action_mask_space, np.expand_dims(action_mask_space_3,axis=0)))
      action_mask_space = np.vstack((action_mask_space, np.expand_dims(action_mask_space_4, axis=0)))
      action_mask_space = np.vstack((action_mask_space, np.expand_dims(action_mask_space_5, axis=0)))

      return action_mask_space

   def generate_action_mask(self, state, component_idx):
      """
      Args
      -------
      state:
         Used to indicate which sub-design the state belongs to. 
         `state[1]` shows the sub-design.
      component_idx:
         The option.

      Returns
      -------
      mask: <1, max_num_action>
         The action mask corresponding to the input `option` and the sub-design where the state is in.
      """
      mask = self._action_mask_space[state[:,1]-1, component_idx.flatten()]
      return mask
   
   def get_cum_reward(self):
      """
      Description
      -------
      Calculate the cumulative reward of the explored designs with `_init_reward`.
      """
      if not self._explored_designs:
         return self._init_reward
      else:
         cum_reward = np.array([row[2] for i, row in enumerate(self._explored_designs)]).sum()
         cum_reward += self._init_reward
         return cum_reward
   
   def normalize_ppa(self):
      normalized_ppa = (self._ppa - self._first_microarch_ppa)/self._ppa_std
      return normalized_ppa

   def renormalize_ppa(self, state):
      return state*self._ppa_std + self._first_microarch_ppa
   
   def get_best_point(self, final_microarchs, normalized_ppas, episode_rewards):
      """
      Description
      -------
      Look for a point which is in the contraint and has the logest projection on the pref vector.

      Parameters
      -------
      final_microarchs:
         All final states of all episodes.
      """
      projs = np.dot(normalized_ppas, self.preference)/np.linalg.norm(self.preference)
      punishment = self.reward_coef*np.linalg.norm(np.cross(normalized_ppas, self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
      proj_with_punishment = projs.reshape(-1) - punishment
      # Check if there exists at least a point of which the `proj_with_punishment` > 0.
      over_zero = proj_with_punishment > 0
      if not over_zero.any():
         print("No design is in the constraint! All `proj_with_punishment` < 0.")
         idx = proj_with_punishment.argmax()
         norm_ppa = normalized_ppas[idx].reshape(-1, normalized_ppas.shape[-1])
         ppa = self.renormalize_ppa(norm_ppa)
         ppa[:,1:] = -ppa[:,1:]
         return idx, final_microarchs[idx], ppa, proj_with_punishment[idx].reshape(-1)
      else:
         # Get rid of the points out of the constraint
         selected_designs = final_microarchs[over_zero]
         selected_norm_ppas = normalized_ppas[over_zero]
         selected_rewards = projs[over_zero]

         best_point_idx = selected_rewards.argmax()
         best_design = selected_designs[best_point_idx]
         best_design_norm_ppa = selected_norm_ppas[best_point_idx].reshape(-1, selected_norm_ppas.shape[-1])
         best_design_ppa = self.renormalize_ppa(best_design_norm_ppa)
         best_design_ppa[:,1:] = -best_design_ppa[:,1:]
         proj = selected_rewards[best_point_idx]
         idx = np.arange(len(final_microarchs))[over_zero][best_point_idx]
         return idx, best_design, best_design_ppa, proj