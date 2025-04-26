# Author: mingjun0521@qq.com


import csv
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from scipy import stats
from collections import Counter


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.Tensor(array)
