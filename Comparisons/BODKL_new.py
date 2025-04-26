import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from botorch.models import SingleTaskGP
import gpytorch
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional, Dict, Tuple, NoReturn, List, Any
# from thop import profile  # 引入 thop 库来计算 FLOPs

from Comparisons.MyDatasetCreation.initialize import micro_al_new

class MLP(nn.Sequential):
    """
        MLP as preprocessor of DKLGP
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        # NOTICE: we do not add spectral normalization
        self.add_module("linear-1", nn.Linear(input_dim, 256))
        self.add_module("relu-1", nn.ReLU())
        self.add_module("linear-2", nn.Linear(256, 256))
        self.add_module("relu-2", nn.ReLU())
        self.add_module("linear-3", nn.Linear(256, output_dim))

class BODKL(object):
    def __init__(self, problem: object) -> None:
        self.problem = problem
        self.preference = np.array(problem.configs["preference"])
        self.reward_coef = problem.configs["reward_coef"]
        # Get data
        _x_labeled, x_unlabeled, \
        _y_labeled, y_unlabeled , \
        time_labeled, time_unlabeled\
            = micro_al_new(self.problem)
        self._x_labeled = _x_labeled.detach().numpy()
        self.x_unlabeled = x_unlabeled.detach().numpy()
        self._y_labeled = _y_labeled.detach().numpy()
        self.y_unlabeled = y_unlabeled.detach().numpy()
        self.time_labeled = time_labeled.detach().numpy()
        self.time_unlabeled = time_unlabeled.detach().numpy()
        self.dim_y = self._y_labeled.shape[-1]

        self._proj_labeled = self.get_projections(self._y_labeled)
        self._proj_unlabeled = self.get_projections(self.y_unlabeled)
        self.mlp = MLP(self._x_labeled.shape[-1], 6)

        self.found_designs = []
        self.found_designs_ppa = []
        self.found_designs_proj = []
        self.train_mse = []
        self.test_mse = []
        self.FLOPs_list = []

        # 拿到数据集中True Optimal的proj
        _all_proj = np.concat((self._proj_labeled, self._proj_unlabeled),axis=0)
        _all_ppa = np.concatenate((self._y_labeled, self.y_unlabeled), axis=0)
        idx_true_best = _all_proj.argmax()
        proj_true_best = _all_proj[idx_true_best]
        ppa_true_best = _all_ppa[idx_true_best]
        ppa_true_best = self.problem.renormalize_ppa(torch.tensor(ppa_true_best.reshape(-1, self.dim_y)))
        ppa_true_best[:,1:] = -ppa_true_best[:,1:]
        print("True Optimal PPA: ", ppa_true_best.detach().numpy())
        print("True Optimal Proj: ", proj_true_best)

    def train(self, max_iter):
        exploration_round = tqdm(range(max_iter))
        for step in exploration_round:
            # x = self.mlp(torch.tensor(self._x_labeled))
            x = self.mlp_forward(self._x_labeled)
            gp = SingleTaskGP(x, torch.tensor(self._proj_labeled, dtype=torch.float32).view(-1,1))
            self.mlp, gp = self.train_model(self.mlp, gp, torch.tensor(self._x_labeled), torch.tensor(self._proj_labeled, dtype=torch.float32))
            # mlp_FLOPs, _ = profile(self.mlp, inputs=(torch.tensor(self._x_labeled, dtype=torch.float32),))     # Calculate FLOPs

            mlp_output = self.mlp_forward(self.x_unlabeled)
            acq_func = LogExpectedImprovement(model=gp, best_f=torch.tensor(self._proj_labeled, dtype=torch.float32).max())
            acqv = acq_func(mlp_output.unsqueeze(1))
            # acqv_FLOPs, _ = profile(acq_func, inputs=(mlp_output.unsqueeze(1), ))
            # FLOPs = mlp_FLOPs + acqv_FLOPs
            # self.FLOPs_list.append(FLOPs)

            max_idx = torch.argmax(acqv)
            x_star = self.x_unlabeled[max_idx]
            y_star = self.y_unlabeled[max_idx] # Simulate x_start to obtain PPA
            proj_star = self._proj_unlabeled[max_idx]

            # Updata dataset
            self._x_labeled = np.concatenate([self._x_labeled,x_star.reshape(-1,self._x_labeled.shape[-1])],axis=0)
            self._y_labeled = np.concatenate([self._y_labeled, y_star.reshape(-1, self.dim_y)], axis=0)
            self._proj_labeled = np.concatenate([self._proj_labeled, proj_star.reshape(-1)], axis=0)
            self.time_labeled = np.concatenate([self.time_labeled,self.time_unlabeled[max_idx].reshape(1)],axis=0)
            self.x_unlabeled = np.delete(self.x_unlabeled,max_idx,axis=0)
            self.y_unlabeled = np.delete(self.y_unlabeled,max_idx,axis=0)
            self._proj_unlabeled = np.delete(self._proj_unlabeled, max_idx, axis=0)
            self.time_unlabeled = np.delete(self.time_unlabeled,max_idx,axis=0)

            self.found_designs.append(x_star)
            self.found_designs_ppa.append(y_star)
            self.found_designs_proj.append(proj_star)

        return self.found_designs, self.found_designs_ppa, self.found_designs_proj

    def train_model(self, mlp, gp, train_x, train_y, epochs=20, lr=1e-4):
        criterion = ExactMarginalLogLikelihood(gp.likelihood, gp)
        optimizer = optim.Adam([{'params': mlp.parameters(), 'lr': lr},
                                {'params': gp.parameters(), 'lr': lr}],)
        for epoch in range(epochs):
            optimizer.zero_grad()
            mlp_output = mlp(train_x)
            _x = (mlp_output - mlp_output.min(0)[0])/(mlp_output.max(0)[0] - mlp_output.min(0)[0])
            gp.set_train_data(_x, train_y, strict=False)
            outputs = gp(_x)
            loss = -criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
        return mlp, gp
    
    def mlp_forward(self, x):
        """
        Forward pass of the MLP.
        """
        x = torch.tensor(x, dtype=torch.float32)
        mlp_output = self.mlp(x)
        mlp_output = (mlp_output - mlp_output.min(0)[0])/(mlp_output.max(0)[0] - mlp_output.min(0)[0])
        return mlp_output

    def get_projections(self, y:np.ndarray):
        """
        Get the projection of the input y on the preference vector.
        """
        proj = np.dot(y, self.preference)/np.linalg.norm(self.preference)
        return proj