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
        self._first_microarch = np.array(config['first_microarch'])  # Set by the user.
        self.max_steps = config['max_steps']
        
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
        self.reward_space = 3
        self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)

        # Others
        self._step_counts = 0
        self.num_microarch_range = np.arange(len(self._microarch))

    def step(self, component_idx, action_idx):
        '''
        Description
        -------
        Outputs the next state and the reward of the current state & action.

        Args
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
            norm_ppa = np.array([0,0,0])
        else:
            norm_ppa = self.get_state_reward(self.state)
            
        state = self.state.copy()
        self._explored_designs.append([state, norm_ppa])

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

        return self.state, norm_ppa, terminated, truncated, info

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

        norm_ppa = self.normalized_ppa[state_idx]
        return norm_ppa

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

        self.state = state.copy()

        self._step_counts = 0
        self._option_selected = np.zeros(self.option_space.n).astype(np.bool_)
        print("Initial State: ", self.state)
        return self.state,info

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

    def normalize_ppa(self):
        normalized_ppa = (self._ppa - self._first_microarch_ppa)/self._ppa_std
        return normalized_ppa

    def renormalize_ppa(self, norm_ppa):
        return norm_ppa*self._ppa_std + self._first_microarch_ppa
    
    def compute_projection(self, norm_ppa, preference):
        proj = np.dot(norm_ppa, preference)/np.linalg.norm(preference)
        return proj
