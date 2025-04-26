# Author: baichen318@gmail.com


import math
import numpy as np
from typing import List, NoReturn
from .dataset import ndarray_to_tensor
from .problem import DesignSpaceProblem
from sklearn.cluster import KMeans
import torch

# from models.Visualization import plot_pictures

class RandomizedTED(object):

    def __init__(self, kwargs: dict):
        super(RandomizedTED, self).__init__()
        self.Nrted = kwargs["Nrted"]
        self.mu = kwargs["mu"]
        self.sig = kwargs["sig"]

    def f(self, u, v):
        u = u[:-2]
        v = v[:-2]
        return pow(
            math.e,
            -np.linalg.norm(
                np.array(u, dtype=np.float64) - np.array(v, dtype=np.float64)
            ) ** 2 / (2 * self.sig ** 2)
        )

    def f_same(self, K: List[List[int]]) -> np.ndarray:
        n = len(K)
        F = []
        for i in range(n):
            t = []
            for j in range(n):
                t.append(self.f(K[i], K[j]))
            F.append(t)
        return np.array(F)

    def update_f(self, F: List[List[int]], K: List[int]) -> NoReturn:
        n = F.shape[0]
        for i in range(len(K)):
            denom = self.f(K[i], K[i]) + self.mu
            for j in range(n):
                for k in range(n):
                    F[j][k] -= (F[j][i] * F[k][i]) / denom

    def select_mi(self, K: List[List[int]], F: List[List[int]]) -> List[List[int]]:
        idx = np.argmax(
                [np.linalg.norm(F[i]) ** 2 / (self.f(K[i], K[i]) + self.mu) \
                    for i in range(len(K))]
            )
        return idx, K[idx]

    def rted(self, vec: np.ndarray, m: int) -> List[List[int]]:
        """
            vec: the dataset
            m: the number of samples initialized from `vec`
        """
        K_ = []
        K_index = []
        for i in range(m):
            M_index = np.random.choice(len(vec), self.Nrted, replace=False)
            M_ = list(vec[M_index])

            M_ = M_ + K_
            M_index = M_index.tolist() + K_index
            F = self.f_same(M_)     # Calculate coveriance metric
            self.update_f(F, M_)    # Update F
            k_little_index, k_little = self.select_mi(M_, F)
            K_.append(k_little)
            K_index.append(M_index[k_little_index])
        return K_index, K_


class MicroAL(RandomizedTED):
    """
    Dataset constructed after cluster w.r.t. DecodeWidth
    """
    _cluster_dataset = None
   
    def __init__(self, problem: DesignSpaceProblem):
        self.problem = problem
        self.configs = problem.configs["initialize"]
        self.num_per_cluster = self.configs["batch"] // self.configs["cluster"]
        self.decoder_threshold = self.configs["decoder-threshold"]
        # feature dimension
        self.n_dim = problem.n_dim
        super(MicroAL, self).__init__(self.configs)

    @property
    def cluster_dataset(self):
        return self._cluster_dataset

    @cluster_dataset.setter
    def cluster_dataset(self, dataset):
        self._cluster_dataset = dataset

    def set_weight(self, pre_v=None):
        # if `pre_v` is specified, then `weights` will be assigned accordingly
        if pre_v:
            assert isinstance(pre_v, list) and len(pre_v) == self.n_dim, \
                assert_error("unsupported pre_v {}".format(pre_v))
            weights = pre_v
        else:
            # NOTICE: `decodeWidth` should be assignd with larger weights
            weights = [1 for i in range(self.n_dim)]
            weights[1] *= self.decoder_threshold
        return weights

    def gather_groups(self, dataset, cluster):
        new_dataset = [[] for i in range(self.configs["cluster"])]
        idx_in_dataset = [[] for i in range(self.configs["cluster"])]

        for i in range(len(dataset)):
            new_dataset[cluster[i]].append(dataset[i])
            idx_in_dataset[cluster[i]].append(i)
        for i in range(len(new_dataset)):
            new_dataset[i] = np.array(new_dataset[i])
        return idx_in_dataset, new_dataset

    def initialize(self, dataset: np.ndarray) -> List[List[int]]:
        """
        Pick points from training data as labeled data, and the rest of the training data is as unlabeled data.
        """
        # NOTICE: `rted` may select duplicated points,
        # in order to avoid this problem, we delete 80%
        # some points randomly
        def _delete_duplicate(vec):
            """
                `vec`: <list>
            """
            return [list(v) for v in set([tuple(v) for v in vec])]
            
        # Cluster dataset w.r.t. DecodeWidth, using KMeans
        weights = self.set_weight()
        # cluster = KMeans(n_clusters=self.configs["cluster"], max_iter=self.configs["clustering-iteration"]).fit(dataset*weights)
        cluster = KMeans(n_clusters=self.configs["cluster"], 
                         max_iter=self.configs["clustering-iteration"]).fit(dataset*weights)
        centroids = cluster.cluster_centers_
        new_assignment = cluster.labels_

        idx_in_dataset, self.cluster_dataset = self.gather_groups(dataset, new_assignment)

        # Sample points from each cluster using RTED
        sampled_data = []
        data_idx_in_dataset = []
        for i in range(len(idx_in_dataset)):
            c = self.cluster_dataset[i]
            x = []
            x_idx = []
            while len(x) < min(self.num_per_cluster, len(c)):
                if len(c) > (self.num_per_cluster - len(x)) and \
                    len(c) > self.configs["Nrted"]:
                    # 按照给出的算法流程，确实有可能会 rted 出重复的元素
                    candidates_idx, candidates = self.rted(
                        c,
                        self.num_per_cluster - len(x)
                    )
                    candidates_idx = list(np.array(idx_in_dataset[i])[candidates_idx])
                else:
                    candidates = c
                    candidates_idx = idx_in_dataset[i]
                for _c in candidates:
                    x.append(_c)
                for _c_idx in candidates_idx:
                    x_idx.append(_c_idx)
                x = _delete_duplicate(x)
                x_idx = list(set(x_idx))
            for _x in x:
                sampled_data.append(_x)
            for _x_idx in x_idx:
                data_idx_in_dataset.append(_x_idx)
            
        '''
        # Find the best point and the worst point
        a_max_idx = np.argmax(centroids,axis=0)[1]
        a_min_idx = np.argmin(centroids,axis=0)[1]
        dis_best = np.sum((self.cluster_dataset[a_max_idx] - centroids[a_max_idx])**2,axis=1)
        dis_worst = np.sum((self.cluster_dataset[a_min_idx] - centroids[a_min_idx])**2,axis=1)
        best_point = self.cluster_dataset[a_max_idx][np.argmax(dis_best)]
        worst_point = self.cluster_dataset[a_min_idx][np.argmax(dis_worst)]
        sampled_data.append(best_point)
        sampled_data.append(worst_point)
        '''    
        return data_idx_in_dataset, sampled_data # This is only the labeled data's features


def micro_al(problem: DesignSpaceProblem, return_time=False):
    """
    Returns:
        :param1 x_labeled: <tensor>
        :param2 x_unlabeled: <tensor>
        :param3 y_labeled: <tensor>
        :param4 time_labeled: <tensor>
    """
    initializer = MicroAL(problem)
    x_labeled_idx, x_labeled = initializer.initialize(problem._x.detach().numpy())
    x_labeled = ndarray_to_tensor(x_labeled)          # x is a normalized tensor
    y_labeled = problem.norm_ppa[x_labeled_idx]    # Get the labeled x's true value
    time_labeled = problem.time[x_labeled_idx]

    rest_idx = np.setdiff1d(np.arange(len(problem._x)), x_labeled_idx)
    x_unlabeled = ndarray_to_tensor(problem._x[rest_idx])

    # # Visualization
    # plot_pictures(gt=np.array([[0,0,0]]), pred=-y_labeled, original_y=-problem.norm_ppa, path="micro_al.pdf")

    if not return_time:
        return x_labeled, x_unlabeled, y_labeled
    else:
        return x_labeled, x_unlabeled, y_labeled, time_labeled
    
def micro_al_new(problem: DesignSpaceProblem):
    """
    The same as micro_al, but return the unlabeled x's y and time.
    Returns:
        :param x_labeled: <tensor>
        :param x_unlabeled: <tensor>
        :param y_labeled: <tensor>
        :param y_unlabeled: <tensor>
        :param time_labeled: <tensor>
        :param time_unlabeled: <tensor>
    """
    initializer = MicroAL(problem)
    x_labeled_idx, x_labeled = initializer.initialize(problem._x.detach().numpy())
    x_labeled = ndarray_to_tensor(x_labeled)
    y_labeled = problem.norm_ppa[x_labeled_idx]    # Get the labeled x's true value
    time_labeled = problem.time[x_labeled_idx]

    rest_idx = np.setdiff1d(np.arange(len(problem._x)), x_labeled_idx)
    x_unlabeled = problem._x[rest_idx]
    x_unlabeled = ndarray_to_tensor(x_unlabeled)
    y_unlabeled = problem.norm_ppa[rest_idx]  
    y_unlabeled = ndarray_to_tensor(y_unlabeled)  # Can only used in model tes
    time_unlabeled = problem.time[rest_idx]    
    time_unlabeled = ndarray_to_tensor(time_unlabeled)  # Can only used in model test

    return x_labeled, x_unlabeled, y_labeled, y_unlabeled, time_labeled, time_unlabeled