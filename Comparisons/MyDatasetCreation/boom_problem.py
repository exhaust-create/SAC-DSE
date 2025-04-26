# Author: baichen318@gmail.com
# Modified By: mingjun0521@qq.com

import torch
import numpy as np
from torch import Tensor
from abc import ABC, abstractmethod
from .vlsi_flow.manager import vlsi_flow
from .vlsi_flow.vlsi_report import get_report
from typing import List, Optional, Tuple, NoReturn
from .dataset import load_dataset, ndarray_to_tensor
from .design_space.boom_design_space import parse_boom_design_space
from sklearn.preprocessing import StandardScaler

def neg_data(input):
    return -input

class BaseProblem(torch.nn.Module, ABC):
    """
        base class for construction a problem.
    """

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base class for construction a problem.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
            evaluate the function on a set of points.

        args:
            X: a `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: if `True`, add observation noise as specified by `noise_std`.

        returns:
            a `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        """
            evaluate the function (w/o observation noise) on a set of points.
        """
        raise NotImplementedError


class MultiObjectiveProblem(BaseProblem):
    """
        base class for a multi-objective problem.
    """

    num_objectives: int
    _ref_point: List[float]
    _max_hv: float

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base constructor for multi-objective test functions.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.float)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        try:
            return self._max_hv
        except AttributeError:
            raise NotImplementedError(
                error_message("problem {} does not specify maximal hypervolume".format(
                    self.__class__.__name__)
                )
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        """
            generate `n` pareto optimal points.
        """
        raise NotImplementedError


class DesignSpaceProblem(MultiObjectiveProblem):
    """
    Construct the dataset used to train and test the model. 

    Including original data, training data and the test data.
    """
    def __init__(self, configs: dict):
        self.configs = configs
        self.load_dataset()

        self._ref_point = torch.tensor([-3.0, -3.0, -3.0])
        self._bounds = torch.tensor([(3.0, 3.0, 3.0)])
        super().__init__()

    def load_dataset(self) -> NoReturn:
        """
        Randomly pick N points from the original dataset to construct a new dataset.

        Pick (N+2) points from the new dataset to construct the Labeled dataset,
        and the rest of N points make up the Unlabeled dataset. 
        
        Transductive rmse will be computed with these N points.
        """
        # Coefficient defined
        p = 0.8 # Percentage of the original dataset to be used to construct the training dataset

        # Load data from the document
        x, y, time = load_dataset(self.configs["dataset"]["path"])
        #####################################
        # 下面可以加入特征值处理的代码
        #####################################
        
        # Standardize y
        self.stand_scaler = StandardScaler()
        y = self.stand_scaler.fit_transform(y)
        
        # In the contest dataset, the larger the perf value is, the better it is, so here we need to 
        # reverse the sign of the first column of y.
        # But if the dataset is from BOOM-Explorer, then the sing of the 1st column 
        # isn't need to be reversed.
        y[:,0] = -y[:,0]
        
        # The less the original y is, the better the input is. But the hypervolume calculation functions
        # here assume maximizing y, so we need to reverse the sign of the original y.
        y = -y

        # All data in the file
        self.original_x = ndarray_to_tensor(x)
        self.original_y = ndarray_to_tensor(y)
        self.time = ndarray_to_tensor(time)
        
        # Normalize the Original Data
        self.original_x_mean = torch.mean(self.original_x, dim=0) # Mean of the original inputs
        self.original_x_std = torch.std(self.original_x, dim=0)   # standard deviation of the original inputs
        # Find where the standard deviation is 0, and remove the corresponding dimension
        self.zero_idx = torch.where(self.original_x_std == 0)
        if self.zero_idx[0].shape[0] != 0:
            self.zero_idx = torch.as_tensor(self.zero_idx)
            self.original_x = self.remove_dim(self.original_x, self.zero_idx)   # 删去了 std=0 的那一维度的 original_x
            self.original_x_mean = self.remove_dim(self.original_x_mean, self.zero_idx)
            self.original_x_std = self.remove_dim(self.original_x_std, self.zero_idx)

        self.original_x_norm = torch.div(self.original_x - self.original_x_mean, self.original_x_std)  # normalize x by every dim
        
        # When test boom exp, do not annotate this line 
        self.original_x_norm = self.original_x.clone().detach()
        
        # Randomly pick 80% of the normalized original data to construct the training data
        idx = np.random.choice(len(self.original_x_norm), int(len(self.original_x_norm) * p), replace=False)  # No duplicate elements
        self.x_train = self.original_x_norm[idx]
        self.y_train = self.original_y[idx]
        self.time_train = self.time[idx]
        # The rest of the normalized original data is used to construct the test data
        rest_idx = np.setdiff1d(np.arange(len(self.original_x)), idx)
        self.x_test = self.original_x_norm[rest_idx]
        self.y_test = self.original_y[rest_idx]
        self.time_test = self.time[rest_idx]

        self.n_dim = self.x_train.shape[-1]

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the performance 'y' of feature 'x' from the original dataset.
        """
        if self.configs["mode"] == "offline":
            _, indices = torch.topk(
                ((self.original_x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
                1,
                1
            )
            return self.original_y[indices].to(torch.float32).squeeze()
        else:
            idx = [self.design_space.vec_to_idx(_x.numpy().tolist()) for _x in x]
            self.design_space.generate_chisel_codes(idx)
            vlsi_flow(self.design_space, idx)
            perf, power, _ = get_report(self.design_space)
            y = torch.cat(
                (ndarray_to_tensor(perf).unsqueeze(1), ndarray_to_tensor(power).unsqueeze(1)),
                dim=1
            )
            return y
    
    def featur_process(self, x):
        pass

    def restandard_x(self, x):
        x = x * self.original_x_std + self.original_x_mean
        len_x = len(x)
        x1 = torch.cat((x[:,:self.zero_idx], self.original_x[:len_x,self.zero_idx]), dim=1)
        x = torch.cat((x1, x[:,self.zero_idx:]), dim=1)
        return x

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

def boom_create_problem(configs: dict) -> DesignSpaceProblem:
    return DesignSpaceProblem(configs)
