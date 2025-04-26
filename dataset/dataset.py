# Author: mingjun0521@qq.com
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from scipy import stats
from collections import Counter
from typing import Dict, Tuple, Sequence

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Dataset:
    def __init__(self, dataset_path: str, design_space_path: str, embed_device = 'cpu'):
        self.device = embed_device
        self.microparam, self.ppa, self.time = self.load_dataset(dataset_path)  # self.dataset = [microarch, ppa, simulation_time]
        self.design_space, self.component_position, self.component_space = self.load_design_space(design_space_path) # self.design_space = [design_space, component_position]
        self.num_components = self.component_position.shape[0]
        self.component_space_flatten = self._flatten_component_space()
        self.microarch_embedding = nn.Embedding(len(self.component_space_flatten) + 1, 4, padding_idx=0).to(embed_device)   # Embed all actions.
        self.component_indices_embedding = nn.Embedding(self.num_components, 2).to(embed_device)   # Embed component_indices. Align with `self.microarch_embedding`.

    def load_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ----------将contest数据集转换成二维数组,并进行特征工程----------
        data = pd.read_csv(path, header=None)
        data = data.values.tolist()
        size = len(data)                    # 要在降维代码前面
        data = list(np.array(data).ravel())
        data_list = [[]for i in range(0, size)]                #定义空二维数组
        for i in range(0,size):
            temp = data[i]
            temp = temp.split()
            data_list[i] = temp
        data_list = np.array(data_list)                         # 还是字符列表

        # Extract data
        # PPA: Performance, Power, Area in order
        data_ppa_str = data_list[:,-4:-1]
        data_embedding_str = data_list[:,0:-4]
        data_embedding_int = [[]for i in range(0, size)]
        data_ppa_flt = [[]for i in range(0, size)] 
        for i in range(0, size):
            temp1 = data_ppa_str[i].ravel()
            temp2 = data_embedding_str[i].ravel()
            temp1 = list((map(float, temp1)))
            temp2 = list((map(int, temp2)))
            data_ppa_flt[i] = temp1
            data_embedding_int[i] = temp2 
        # data_ppa_flt = stats.zscore(data_ppa_flt)             # To normalize data.
        data_ppa_flt = np.array(data_ppa_flt)
        data_embedding_int = np.array(data_embedding_int)

        data_time_str = data_list[:,-1]
        data_time_flt = list((map(float, data_time_str)))
        time = np.array(data_time_flt)

        x = data_embedding_int
        y = data_ppa_flt

        return x, y, time

    def load_design_space(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """       
        Returns
        -------
        design_space:
            All possible parameter combinations of all components. Doesn't include component's idx.
            It is like: 0 | [parameter_combination_1 for component_0]
                        0 | [parameter_combination_2 for component_0]
                        1 | [parameter_combination_1 for component_1]
                        1 | [parameter_combination_2 for component_1]
                        1 | [parameter_combination_3 for component_1]
                        ...

        component_position:
            The range of each component's parameter position. The component is like: ISU, IFU, PRF, ...
            Like:   component_0: [0,1]
                    component_1: [1,3]
                    component_2: [3:4]
        
        component_space:
            Options for all components.
            Like:   component_0: [possible options]
                    component_1: [possible options]
                    component_2: [possible options]
        """
        data = pd.read_excel(path, sheet_name='Components', header=0)
        data = data.values.tolist()
        indices = []
        for i, x in enumerate(data[0]):
            if x == 'idx':
                indices.append(i)
        indices = np.array(indices)
        data = np.array(data[1:])
        component_space = np.transpose(data[:,indices])
        design_space = np.delete(data,indices,axis=1)
        design_space = np.transpose(design_space)   # After transposition, each row's id means a component index.
        # component_position： The range of each component's position. 
        # 格式："component_idx: [start, end]" (左闭右开).
        # Like: component_0: [0,1]
        #       component_1: [1,3]
        #       component_2: [3:4]
        component_position = np.zeros([len(indices),2]).astype(int)
        j = 0
        for i in range(len(indices)):
            if i != len(indices)-1:
                temp = indices[i+1] - indices[i]-1
                component_position[i] = np.array([j, j+temp])
                j=j+temp
            else:
                component_position[i] = np.array([j,design_space.shape[0]])
        return design_space, component_position, component_space
    
    def microparam_2_microidx(self, microparam: torch.Tensor) -> np.ndarray:
        """
        Description
        -------
            Turn the microarch parameter combination into the idx combination.

        Args
        -------
        microparam:
            A microarch parametre combination.

        Return
        -------
        microidx:
            The microarch index combination extracted from the design space.
        """
        # Check if `microparam` is torch.Tensor
        if not isinstance(microparam, torch.Tensor):
            microparam = torch.tensor(microparam)
        design_space = torch.Tensor(self.design_space)
        microparam = microparam.view(-1,self.design_space.shape[0])
        microidx = torch.zeros([len(microparam), self.num_components]).int()
        for i in range(len(microparam)):
            for j in range(self.num_components):
                [a,b] = self.component_position[j]
                design_space_truncated = design_space[a:b].transpose(0,1)
                design_space_truncated = design_space_truncated[~design_space_truncated.isnan()].view(-1,b-a)
                mask = (design_space_truncated == microparam[i,a:b]).view(len(design_space_truncated),-1).all(dim=1)
                if mask.sum() == 0:
                    raise ValueError("The microarch parameter combination is not in the design space.")
                else:
                    _, idx = torch.topk(mask.int(), k=1)
                microidx[i,j] = idx+1   # 因为原生 idx 是从0开始的，所以需要 +1 变成和 design-space 中的 idx 数值一样。
        # for i in range(self.num_components):
        #     [a,b] = self.component_position[i]
        #     _, idx = torch.topk(
        #         ((design_space[a:b] == microparam.view(-1,self.design_space.shape[0])[:,a:b])).int(), 
        #         k=1)
        #     microidx[i] = idx+1    # 因为原生 idx 是从0开始的，所以需要 +1 变成和 design-space 中的 idx 数值一样。
        return microidx.detach().numpy()
    
    def microidx_2_microparam(self, microidx:torch.Tensor) -> np.ndarray:
        """
        Turn the microarch idx combination into the parameter combination.

        Args
        -------
        microidx:
            The microarch idx combination

        Return
        -------
        microparam:
            A microarch parametre combination.
        """
        # Check if `microidx` is torch.Tensor
        if not isinstance(microidx, torch.Tensor):
            microidx = torch.tensor(microidx)
        microparam = torch.zeros([self.microparam.shape[-1]])
        design_space = torch.Tensor(self.design_space)
        for i in range(self.num_components):
            [start,end] = self.component_position[i]
            microparam[start:end] = design_space[start:end,microidx[i]-1]
        return microparam.int().detach().numpy()
    
    def _flatten_component_space(self):
        component_space_flatten = self.component_space.flatten()
        return component_space_flatten[~np.isnan(component_space_flatten)] # Get rid of `NaN`
    
    def microidx_2_microembedding(self, microidx:torch.Tensor) -> np.ndarray:
        """
        Description
        -------
        Turn the microarch idx combination into the embedding combination.
        For embedding, we need to regard each option of each component as independent. 
        For example:   component idx | component options         component idx | flatten component options
                        0            |   [1, 2]            -->      0          |      [1, 2]
                        1            |   [1, 2, 3]         -->      1          |      [3, 4, 5]
                        2            |   [1, 2, 3, 4]      -->      2          |      [6, 7, 8, 9]
        We call the changed `microidx` as `microidx_flatten`. We can only use embedding after flattening the `microidx`.

        Parameters
        -------
        microidx:
            The microarch idx combination

        Return
        -------
        microembedding:
            A microarch embedding combination. 
        """
        # Check if `microidx` is torch.Tensor
        if not isinstance(microidx, torch.Tensor):
            microidx = torch.tensor(microidx).to(self.device)
        microidx = microidx.view(-1, self.num_components)
        num_NotNaN = np.sum(~np.isnan(self.component_space),axis=1)
        num_NotNaN = np.insert(num_NotNaN, 0, 0)
        bias = torch.tensor(num_NotNaN.cumsum()).to(self.device).flatten()
        microidx_flatten = microidx + bias[:self.num_components]    # The action is 1-based, so we use 0 as the `padding_idx`.`
        microembedding = self.microarch_embedding(microidx_flatten.long())
        return microembedding

def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.Tensor(array)
