import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

from Comparisons.MyDatasetCreation.initialize import micro_al_new

class BagGBRT(object):
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
        
        # Sum the time for simulation. 
        # Randomly pick 5 points from self.time to get the max time as the final simulation time, with no duplication.
        self.num_parallel = 5    # number of points in a parallel simulation
        self.simulate_time = 0
        idx_sim = torch.randperm(len(self.time_labeled))
        num_sim = int(len(self.time_labeled)/self.num_parallel)
        for i in range(num_sim):
            self.simulate_time += self.time_labeled[idx_sim[i*self.num_parallel:(i+1)*self.num_parallel]].max()
        # Instantiate model
        HBO_params = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.5}
        HBO_params_ada = {'n_estimators': 20, 'n_jobs': 1, 'bootstrap': False}
        self.bag_gbrt = BaggingRegressor(estimator=GradientBoostingRegressor(**HBO_params),**HBO_params_ada)
        # self.bag_gbrt.fit(self._x_labeled, self._proj_labeled)        

        self.found_designs = []
        self.found_designs_ppa = []
        self.found_designs_proj = []
        self.train_mse = []
        self.test_mse = []
            
    def train(self, max_iter):
        iterator = tqdm(range(max_iter))
        for step in iterator:
            # # Use initial predictions to compute weights based on errors
            # predicted = self.bag_gbrt.predict(self._x_labeled)
            # errors = np.abs(self._proj_labeled - predicted)
            # sample_weight = errors / np.sum(errors)  # Normalize weights
            # # self.bag_gbrt.fit(self._x_labeled, self._proj_labeled)
            # for estimator in self.bag_gbrt.estimators_:
            #     estimator.fit(self._x_labeled, self._proj_labeled, sample_weight=sample_weight)
            self.bag_gbrt.fit(self._x_labeled, self._proj_labeled)  
            pred_projs = [estimator.predict(self.x_unlabeled) for estimator in self.bag_gbrt.estimators_]
            pred_projs = np.array(pred_projs).transpose()   # (num_designs, num_estimators)
            mean = np.mean(pred_projs, axis=1)
            var=np.var(pred_projs, axis=1)

            # Compute MSE
            train_pred = self.bag_gbrt.predict(self._x_labeled)
            self.train_mse.append(np.mean((self._proj_labeled - train_pred)**2)) 
            if (step+1)%15 == 0:
                self.test_mse.append(np.mean((self._proj_unlabeled - mean)**2))

            # Compute UCB
            ucb = self.ucb(mean,var,step)
            max_idx = np.argmax(ucb)
            x_star = self.x_unlabeled[max_idx]
            y_star = self.y_unlabeled[max_idx] # Simulate x_start to obtain PPA
            proj_star = self._proj_unlabeled[max_idx]
            self.simulate_time += self.time_unlabeled[max_idx]
            # Updata dataset
            self._x_labeled = np.concatenate([self._x_labeled,x_star.reshape(-1,self._x_labeled.shape[-1])],axis=0)
            self._y_labeled = np.concatenate([self._y_labeled, y_star.reshape(-1, self.dim_y)], axis=0)
            self._proj_labeled = np.concatenate([self._proj_labeled, proj_star.reshape(-1)], axis=0)
            self.time_labeled = np.concatenate([self.time_labeled,self.time_unlabeled[max_idx].reshape(1)],axis=0)
            self.x_unlabeled = np.delete(self.x_unlabeled,max_idx,axis=0)
            self.y_unlabeled = np.delete(self.y_unlabeled,max_idx,axis=0)
            self._proj_unlabeled = np.delete(self._proj_unlabeled, max_idx, axis=0)
            self.time_unlabeled = np.delete(self.time_unlabeled,max_idx,axis=0)

            # Test
            all_x = np.concatenate([self._x_labeled, self.x_unlabeled], axis=0)
            all_y = np.concatenate([self._y_labeled, self.y_unlabeled], axis=0)
            all_proj = np.concatenate([self._proj_labeled, self._proj_unlabeled], axis=0)
            pred_proj = self.bag_gbrt.predict(all_x)
            idx = np.argmax(pred_proj)
            found_design = all_x[idx].copy()
            found_design_ppa = all_y[idx].copy()
            found_design_proj = all_proj[idx].copy()
            self.found_designs.append(found_design)
            self.found_designs_ppa.append(found_design_ppa)
            self.found_designs_proj.append(found_design_proj)

        return self.found_designs, self.found_designs_ppa, self.found_designs_proj
        # return self._x_labeled, self._y_labeled, self._proj_labeled
            
    def ucb(self, mean:np.ndarray, var:np.ndarray, iter):
        """
        Inputs:
        -------
        mean: shape like np.array(n_sample,dim_y)
            The predictive obj mean values.
        var: shape like np.array(n_sample,dim_y)
            The predictive obj variance values.

        Outputs:
        -------
        ucb: shape like np.array(n_sample,)
        """
        beta = np.sqrt(2*np.log(iter+1)/(iter+1))
        ucb = mean + beta * np.sqrt(var)
        return ucb
    
    def get_projections(self, y:np.ndarray):
        """
        Get the projection of the input y on the preference vector.
        """
        proj = np.dot(y, self.preference)/np.linalg.norm(self.preference)
        return proj
    
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
        punishment = self.reward_coef*np.linalg.norm(np.cross(normalized_ppas, self.preference.reshape(1,-1)), axis=1)/np.linalg.norm(self.preference.reshape(1,-1))
        proj_with_punishment = episode_rewards - punishment
        # Check if there exists at least a point of which the `proj_with_punishment` > 0.
        over_zero = proj_with_punishment > 0
        if not over_zero.any():
            print("No design is in the constraint! All `proj_with_punishment` < 0.")
            idx = proj_with_punishment.argmax()
            norm_ppa = normalized_ppas[idx].reshape(-1, normalized_ppas.shape[-1])
            ppa = self.problem.renormalize_ppa(torch.tensor(norm_ppa))
            ppa[:,1:] = -ppa[:,1:]
            return idx, final_microarchs[idx], ppa.numpy(), proj_with_punishment[idx]
        else:
            # Get rid of the points out of the constraint
            selected_designs = final_microarchs[over_zero]
            selected_norm_ppas = normalized_ppas[over_zero]
            selected_rewards = episode_rewards[over_zero]

            best_point_idx = selected_rewards.argmax()
            best_design = selected_designs[best_point_idx]
            best_design_norm_ppa = selected_norm_ppas[best_point_idx].reshape(-1, selected_norm_ppas.shape[-1])
            best_design_ppa = self.problem.renormalize_ppa(torch.tensor(best_design_norm_ppa))
            best_design_ppa[:,1:] = -best_design_ppa[:,1:]
            proj = selected_rewards[best_point_idx]
            idx = np.arange(len(final_microarchs))[over_zero][best_point_idx]
            return idx, best_design, best_design_ppa.numpy(), proj
    
    # def report(self):
    #     gt = get_pareto_frontier(self.problem.original_y)
    #     gt = rescale_dataset(gt)
    #     pred = rescale_dataset(self.pareto_y)
    #     original_y = rescale_dataset(self.problem.original_y)
    #     p = self.problem.configs["report"]["path"]
    #     mkdir(p)
    #     plot_pictures(gt, pred, original_y,
    #                   os.path.join(p, "BagGBRT_report.pdf"))
    #     write_txt(
    #         os.path.join(
    #             p,
    #             "BagGBRT_adrs.rpt"
    #         ),
    #         np.array(self.adrs),
    #         fmt="%f"
    #     )
    #     write_txt(
    #         os.path.join(
    #             p,
    #             "BagGBRT_hypervolume.rpt"
    #         ),
    #         np.array(self.final_phv),
    #         fmt="%f"
    #     )
    #     write_txt(
    #         os.path.join(
    #             p,
    #             "BagGBRT_pareto-frontier.rpt"
    #         ),
    #         np.array(pred),
    #         fmt="%f"
    #     )
    #     write_txt(
    #         os.path.join(
    #             p,
    #             "BagGBRT_pareto-optimal-solutions.rpt"
    #         ),
    #         np.array(self.pareto_x),
    #         fmt="%f"
    #     )