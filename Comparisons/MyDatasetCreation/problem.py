# Author: baichen318@gmail.com
# Modified By: mingjun0521@qq.com

import torch
import numpy as np
from typing import List, Optional, Tuple, NoReturn
from .dataset import load_dataset, ndarray_to_tensor, tensor_to_ndarray
from dataset.dataset import Dataset

class DesignSpaceProblem:
    """
    Construct the dataset used to train and test the model. 

    Including original data, training data and the test data.
    """
    def __init__(self, configs: dict):
        self.configs = configs
        self.load_dataset()
        self._first_microarch_ppa = self.evaluate_true(torch.tensor(self._first_microarch_comp))
        self.norm_ppa = self.normalize_ppa()
        super().__init__()

    def load_dataset(self) -> NoReturn:
        """
        Randomly pick N points from the original dataset to construct a new dataset.

        Pick (N+2) points from the new dataset to construct the Labeled dataset,
        and the rest of N points make up the Unlabeled dataset. 
        
        Transductive rmse will be computed with these N points.
        """
        self.dataset = Dataset(dataset_path=self.configs["dataset_path"], design_space_path=self.configs["design_space_path"], embed_device='cpu')
        self._first_microarch = torch.tensor(self.configs["first_microarch"])
        self._first_microarch_comp = self.dataset.microparam_2_microidx(self._first_microarch)
        # Load data from the document
        _microarch, y, time= self.dataset.microparam.copy(), self.dataset.ppa.copy(), self.dataset.time.copy()
        x = self.dataset.microparam_2_microidx(_microarch)

        y[:,1:] = -y[:,1:]

        # Only obtain designs with the particular "DecodeWidth".
        state_mask = x[:,1]==self._first_microarch_comp[:,1]
        idx_scale = np.arange(0, len(x))
        state_idx = idx_scale[state_mask]
        self._x = x[state_idx].copy()
        self._y = y[state_idx].copy()
        self._ppa_std = y.std(axis=0)
        self.time = time[state_idx].copy()

        # All data in the file
        self._x = ndarray_to_tensor(self._x)
        self._y = ndarray_to_tensor(self._y)
        self.time = ndarray_to_tensor(self.time)

        self.n_dim = self._x.shape[-1]

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the performance 'y' of feature 'x' from the original dataset.
        """
        state_idx = (self._x==x).all(dim=1)   # Find out the state index in the dataset.
        if ~state_idx.any():
            raise Exception("Can't find the state in the dataset!")
        else:
            state_idx = state_idx.int().argmax()
        return self._y[state_idx].to(torch.float32).squeeze()
    
    def normalize_ppa(self):
      normalized_ppa = (self._y - self._first_microarch_ppa)/self._ppa_std
      return normalized_ppa
    
    def renormalize_ppa(self, state):
      ppa = state*self._ppa_std + self._first_microarch_ppa
      return ppa

    def remove_dim(self, input:torch.Tensor, input_idx:int):
        '''
        Remove the 'input_idx'th dim from the input.
        '''
        input_idx, _ = torch.topk(input_idx, k=len(input_idx)) # Sort the input_idx from large to small
        input = input.t()
        for idx in input_idx:
            input = torch.cat((input[:idx], input[idx+1:]), dim=0)
        input = input.t()
        return input

def rescale_dataset(input):
    # input[:,1:] = -input[:,1:]
    return -input

def my_create_problem(configs: dict) -> DesignSpaceProblem:
    return DesignSpaceProblem(configs)
