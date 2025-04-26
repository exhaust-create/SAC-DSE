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
   def __init__(self, config, embed_device='cpu', delta_proj_coef=0.1):
      #  Read configurations
      dataset_path = config['dataset_path']
      design_space_path = config['design_space_path']
      self.preference = np.array(config['preference'])
      self._first_microarch = np.array(config['first_microarch'])  # Set by the user.
      self.max_steps = config['max_steps']
      self.reward_coef = config['reward_coef']
      self._delta_proj_coef = delta_proj_coef
      self._min_punishment = config['punishment']/config["max_episodes"]
      self._punishment = config['punishment']
      self._max_punishment = config['punishment']
      
      # Load dataset.
      self.dataset = Dataset(dataset_path=dataset_path, design_space_path=design_space_path, embed_device=embed_device)
      self._microarch, self._ppa, self._sim_time = self.dataset.microparam.copy(), self.dataset.ppa.copy(), self.dataset.time.copy()
      self.component_space = self.dataset.component_space.copy()
      self._component_range = (~np.isnan(self.component_space)).sum(axis=-1).flatten().cumsum()   # Scale of each component in `self.component_space_flatten`. 
      self.component_space_flatten = self.dataset.component_space_flatten.copy()

      # Pre-process data.
      self.microarch_comp = self.dataset.microparam_2_microidx(self._microarch)  # Change all microarch parameter combinations into component combinations.
      self._ppa[:,1:]=-self._ppa[:,1:]   # Take the negative of the "performance".
      self._first_microarch_comp = self.dataset.microparam_2_microidx(self._first_microarch)
      first_microarch_idx = (self.microarch_comp==self._first_microarch_comp).all(axis=1).argmax() # Find the index of the first_microarch in the dataset.
      self._first_microarch_ppa = self._ppa[first_microarch_idx].copy()
      self._ppa_std = self._ppa.std(axis=0)
      self.normalized_ppa = self.normalize_ppa()
      self._norm_first_microarch_ppa = self.normalized_ppa[first_microarch_idx]
      self._action_mask_space = self.__generate_action_mask_space()
      
      
      # Set up hyper-parameters
      self.observation_space = spaces.Discrete(self._first_microarch_comp.shape[-1])
      self.action_space = spaces.Discrete(len(self.component_space_flatten))  # The max num of options of all the components.
      self.reward_space = 1
      self._option_selected = np.zeros(self.microarch_comp.shape[0]).astype(np.bool_)

      # Others
      self._step_counts = 0
      self._episode_counts = 0
      self._last_state = None
      self._last_action = None
      self._init_reward = 0
      self._explored_designs = []
      self._stall_counts = 0
   
   def step(self, action_idx, test=False):
      '''
      Description
      -------
      Outputs the next state and the reward of the current state & action.

      Args
      -------
      test:
         Indicates if the `env` is running test. If `test=True`, then dont't calculate `self.total_num_explored_designs`.

      Parameters
      -------
      component_idx:
         It's the current option. Indicating the idx of the changing component.

      Returns
      -------
      state: 
         Microarch.
      reward: 
         r_current = (PPA_current · preference)/|preference| - alpha*|PPA_current X preference|/|preference| - cumulative_reward
      terminated:
         If a state doesn't change from the current step, then terminate the episode. But here we assume that there isn't a "best point" in the env.
      truncated:
         If the number of episodes arrives at the max number, or the difference between the last reward and the current reward is smaller than `epsilon`, then `truncated` is True.
      info: dict
      '''
      info = {}
      truncated = False
      terminated = False   
      # Change the state with `component_idx` & `action_idx`.
      component_idx = self.search_changed_option(action_idx)
      self._last_state = self.state.copy()
      self.state[:,component_idx] = self.component_space_flatten[action_idx].copy()

      # if self._last_action == action_idx:
      if (self._last_state == self.state).all():
         self._stall_counts += 1
         reward_stall_punishment = np.array([self._punishment*self._stall_counts])
         if self._stall_counts > 4:
            self._stall_counts = 0
            terminated = True
      else:
         self._last_action = action_idx
         reward_stall_punishment = 0
         self._stall_counts = 0

      # # Only if the current state is the same as the last state, then the episode terminates.
      # same = self._last_state == self.state
      # if same.all():
      #    terminated = True
      # else:
      #    self._last_state = self.state.copy()

      # # For debug
      # comp_option = self.component_space_flatten[action_idx].copy()
      # print("component: {}, comp_option: {}, current_state: {}".format(component_idx, comp_option, self.state))

      exist_zero = (self.state==np.array(0)).any()
      if exist_zero:
         reward = np.array([0])
      else:
         norm_ppa, reward = self.get_state_reward(self.state)
         reward = reward + reward_stall_punishment
         
         # if proj_with_punishment < -1:
         #    terminated = True
         
      state = self.state.copy()
      if exist_zero:
         self._explored_designs.append([state, np.array([0,0,0]), reward])
      else:
         self._explored_designs.append([state, norm_ppa, reward])
         # print("Num of explored_designs: {}, total num of explored_design: {}".format(len(self._explored_designs), self.total_num_explored_designs))

      self._step_counts += 1
      self._option_selected[component_idx] = True
      # When all options are selected at least once, or the step counts exceed the `max_steps`, the episode terminates.
      # if self._option_selected[2:].all() or self._step_counts > self.max_steps:
      if self._step_counts == self.max_steps:
         truncated = True
      else:
         truncated = False

      # # Punishment increase.
      # if terminated or truncated:
      #    self._episode_counts += 1
      #    self._punishment = (1-np.exp(-2*self._episode_counts))*self._max_punishment

      return self.state, reward, terminated, truncated, info
    
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
      # # Store the explored designs. A 2 dims list: [microarch_component, ppa, reward]
      # if len(self._explored_designs) != 0:
      #    # Extract the best design from the explored designs for the beginning of an episode.
      #    last_best_design_idx = np.array([row[2] for i, row in enumerate(self._explored_designs)]).argmax()
      #    self.state = self._explored_designs[last_best_design_idx][0]
      #    norm_ppa = self._explored_designs[last_best_design_idx][1]

      #    # # Has a probability to choose the randomly searched design for the beginning of an episode.
      #    # rand = np.random.uniform(0,1)
      #    # if rand > 0.5:
      #    #    last_best_design_idx = np.array([row[2] for i, row in enumerate(self._explored_designs)]).argmax()
      #    #    self.state = self._explored_designs[last_best_design_idx][0]
      #    #    norm_ppa = self._explored_designs[last_best_design_idx][1]
      #    # else:
      #    #    state_mask = self.microarch_comp[:,1]==self._first_microarch_comp[:,1]
      #    #    idx_scale = np.arange(0, len(self.microarch_comp))
      #    #    state_idx = np.random.choice(idx_scale[state_mask])
      #    #    self.state = self.microarch_comp[state_idx].reshape(1, -1).copy()
      #    #    norm_ppa = self.normalized_ppa[state_idx].copy()
      # else:
      #    # # # # # # # # # # # # # # #
      #    # #   Randomly sample a data from `self.microarch_comp`.
      #    # #   This "reset" method may cause something problems, for example, for different initial states, the rewards may be the same, 
      #    # # but the final cumulative rewards may differ from each other because of the initial rewards of the initial states.
      #    # # # # # # # # # # # # # # #
      #    # state_mask = self.microarch_comp[:,1]==self._first_microarch_comp[:,1]
      #    # idx_scale = np.arange(0, len(self.microarch_comp))
      #    # state_idx = np.random.choice(idx_scale[state_mask])
      #    # self.state = self.microarch_comp[state_idx].reshape(1, -1).copy()
      #    # norm_ppa = self.normalized_ppa[state_idx].copy()
      #    self.state = self._first_microarch_comp.copy()
      #    norm_ppa = self._norm_first_microarch_ppa.copy()

      state_mask = self.microarch_comp[:,1]==self._first_microarch_comp[:,1]
      idx_scale = np.arange(0, len(self.microarch_comp))
      state_idx = np.random.choice(idx_scale[state_mask])
      self.state = self.microarch_comp[state_idx].reshape(1, -1).copy()
      norm_ppa = self.normalized_ppa[state_idx].copy()
      
      proj= np.dot(norm_ppa, self.preference)/np.linalg.norm(self.preference)
      proj = np.array([proj])
      self._init_reward = proj.copy()

      # self.state = self._first_microarch_comp.copy()
      # self._init_reward = 0
      self._explored_designs = []

      self._step_counts = 0
      self._last_action = None
      self._option_selected = np.zeros(self.microarch_comp.shape[1]).astype(np.bool_)
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
      # state[:,:2] = self.microarch_comp[rand_idx].reshape(1, -1)[:,:2].copy()

      # state = np.zeros_like(self._first_microarch_comp)
      state[:,:2] = self._first_microarch_comp[:,:2].copy()

      self.state = state.copy()

      self._step_counts = 0
      self._option_selected = np.zeros(self.microarch_comp.shape[1]).astype(np.bool_)
      print("Initial State: ", self.state)
      return self.state,info
   
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
      
      # # Projection with distance punishment.
      # proj_with_punishment = np.dot(self.normalized_ppa[state_idx], self.preference)/np.linalg.norm(self.preference) \
      #    - self.reward_coef*np.linalg.norm(np.cross(self.normalized_ppa[state_idx], self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
      
      # Only projection.
      norm_ppa = self.normalized_ppa[state_idx].copy()
      proj = np.dot(norm_ppa, self.preference)/np.linalg.norm(self.preference)
      proj = np.array([proj])

      # Use the total aggregated reward as the baseline.
      if not self._explored_designs:
         delta_proj = proj - self._init_reward
      else:
         # current_sum_reward = np.array([row[2] for i, row in enumerate(self._explored_designs)]).sum()
         # reward = proj_with_punishment - current_sum_reward - self._init_reward    # Culculate step reward.
         last_proj = np.dot(self._explored_designs[-1][1], self.preference)/np.linalg.norm(self.preference)
         delta_proj = proj - last_proj

      reward = self._delta_proj_coef*delta_proj + (1-self._delta_proj_coef)*proj
      return norm_ppa, reward
   
   def get_projection(self, norm_ppa):
      """
      Description
      -------
      Calculate the projection of the current state.
      """
      proj = np.dot(norm_ppa, self.preference)/np.linalg.norm(self.preference)
      return proj
   
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
   
   # def test_reset(self, init_state = None):
   #    """
   #    Description
   #    -------
   #    When "zero_reset", `init_state` should NOT be overlooked. That means user should input `init_state` on their own.
   #    """
   #    info = {}
   #    # Store the explored designs. A 2 dims list: [microarch_component, ppa, reward]
   #    self._explored_designs = []

   #    if init_state is None:
   #       rand_idx = np.random.randint(0, len(self.microarch_comp))
   #       state = self.microarch_comp[rand_idx].reshape(1, -1).copy()
   #    else:
   #       state = init_state.reshape(1, -1).copy()

   #    self.state = state.copy()
   #    self._step_counts = 0
   #    self._option_selected = np.zeros(self.microarch_comp.shape[-1]).astype(np.bool_)

   #    return self.state
   
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
      # action_mask_space_1[0,1:] = False
      action_mask_space_1[0,:] = False
      # action_mask_space_1[1,1:] = False
      action_mask_space_1[1,:] = False
      action_mask_space_1[2,3:] = False
      action_mask_space_1[3,3:] = False
      action_mask_space_1[4,4:] = False
      action_mask_space_1[5,3:] = False
      action_mask_space_1[6,3:] = False
      action_mask_space_1[7,4:] = False
      action_mask_space_1[8,4:] = False

      # Action mask space for sub-design-2
      action_mask_space_2 = action_mask_space.copy()
      # action_mask_space_2[0,1:] = False
      # action_mask_space_2[1,0] = False
      action_mask_space_2[0,:] = False
      action_mask_space_2[1,:] = False
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
      # action_mask_space_3[0,:1] = False
      # action_mask_space_3[1,:2] = False
      action_mask_space_3[0,:] = False
      action_mask_space_3[1,:] = False
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
      # action_mask_space_4[0,:1] = False
      # action_mask_space_4[1,:3] = False
      action_mask_space_4[0,:] = False
      action_mask_space_4[1,:] = False
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
      # action_mask_space_5[0,:1] = False
      # action_mask_space_5[1,:4] = False
      action_mask_space_5[0,:] = False
      action_mask_space_5[1,:] = False
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

   def generate_action_mask(self, state):
      """
      Parameters
      -------
      state:
         Used to indicate which sub-design the state belongs to. 
         `state[1]` shows the sub-design.
      component_idx:
         The option.

      Returns
      -------
      mask: <1, num_all_actions>
         The action mask corresponding to the input `option` and the sub-design where the state is in.
      """
      mask = self._action_mask_space[state[:,1]-1].reshape(state.shape[0],-1)
      mask = mask[:,~np.isnan(self.component_space.flatten())]
      return mask
   
   def search_changed_option(self, actions):
      """
      Parameters
      -------
      actions:
         Shape is like (batch_size, 1)
      """
      diff = actions - self._component_range
      # `diff` is like (batch_size, num_option)
      # Get the indicex of the first element that is less than 0 in each row.
      changed_option_idx = (diff<0).argmax()
      return changed_option_idx
   
   def locate_microarch_idx(self, state):
      """
      Description
      -------
      Find the index of the input state in the dataset. Can only input a state at once.
      """
      state_idx = (self.microarch_comp==state).all(axis=1)
      if ~state_idx.any():
         raise Exception("Can't find the state in the dataset!")
      else:
         state_idx = state_idx.argmax()
      return state_idx
   
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