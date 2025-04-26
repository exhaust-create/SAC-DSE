import argparse
# coding=utf-8
import random
import matplotlib.pyplot as plt
import torch
import yaml
from typing import Dict
import os, sys, time
import numpy as np
import multiprocessing
import pandas as pd
from functools import partial

os.chdir(sys.path[0])
from Comparisons.MyDatasetCreation.problem import DesignSpaceProblem
from Comparisons.BODKL_new import BODKL
from drawer.PlotLearningCurve import plot_result, ema_plotting

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)


# Random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Read the config.yml file
def get_configs(fyaml: str) -> Dict:
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs

def bodkl_run_once(config, seed=300):
    max_episodes = config["max_episodes"]
    times = 5
    train_iter = int(times * max_episodes)
    all_rewards = np.zeros((max_episodes, 1))  # (num_episode, num_seeds)
    # indices = np.arange(0, train_iter, times)

    setup_seed(seed)
    problem = DesignSpaceProblem(config)
    agent = BODKL(problem)
    found_designs, found_designs_ppa, found_designs_proj = agent.train(train_iter)  # "found_designs_ppa" is normalized PPA.
    
    # train_mse = np.array(agent.train_mse).reshape(1,-1)
    test_mse = np.array(agent.test_mse)
    FLOPs = np.array(agent.FLOPs_list)

    all_rewards = np.array(found_designs_proj).reshape(max_episodes,times).max(axis=1)

    # best_design_episode, best_design, best_design_ppa, proj = agent.get_best_point(np.array(found_designs), np.array(found_designs_ppa),
    #                                                           np.array(found_designs_proj))
    best_design_episode = np.array(found_designs_proj).argmax()
    best_design = found_designs[best_design_episode]
    best_design_ppa = torch.tensor(found_designs_ppa)[best_design_episode]
    best_design_ppa = problem.renormalize_ppa(best_design_ppa)  # Tensor
    best_design_ppa[1:] = -best_design_ppa[1:]
    proj = found_designs_proj[best_design_episode]

    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa.numpy(), proj))
    return np.array(found_designs_proj), all_rewards, test_mse, best_design, best_design_ppa.detach().numpy(), proj, FLOPs

if __name__ == "__main__":
    seed = 300
    num_seed = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--width_pref', default = None)
    args = parser.parse_args()
    width_pref = "2W_721" if args.width_pref is None else args.width_pref
    config_sac = get_configs("config/config_sac_"+width_pref+".yml")
    mean, std, bodkl_best_proj, all_sim_rounds = [], [], [], []
    best_design, best_design_ppa = [], []
    seeds = [(seed + i) for i in range(num_seed)]
    
    start_time = time.time()
    for seed in seeds:
        found_designs_projs, all_rewards, test_mse, bodkl_best_design, bodkl_best_design_ppa, proj, FLOPs = bodkl_run_once(config_sac, seed=seed)
        mean.append(all_rewards)
        bodkl_best_proj.append(proj)
        best_design.append(bodkl_best_design)
        best_design_ppa.append(bodkl_best_design_ppa)
    end_time = time.time()
    bodkl_running_time = end_time - start_time
    bodkl_running_time = bodkl_running_time/len(seeds)
    mean = np.array(mean).mean(axis=0)

    episodes = np.arange(len(mean))
    label=['bodkl']
    ema_plotting(episodes, mean, 15, 'Avg_Reward', 'avg_reward', os.path.join(config_sac["reports_folder_path"], "BODKL/bodkl_avg_reward_"+width_pref+".pdf"), label=label)

    data={'bodkl_mean': mean}
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "BODKL/bodkl_mean_"+width_pref+".csv"), index=False)

    # Save best_proj.
    data={
        'bodkl_best_proj': bodkl_best_proj,
        "bodkl_run_time": bodkl_running_time
        }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "BODKL/bodkl_best_proj_"+width_pref+".csv"), index=False)

    data={
        'bodkl_best_design': best_design,
        'bodkl_best_design_ppa': best_design_ppa
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "BODKL/best_design_and_ppa_"+width_pref+".csv"), index=False)

    # # For FLOPs. Running only a random seed is enough.
    # data={'BoDKL_FLOPs': FLOPs}
    # pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "BODKL/bodkl_FLOPs.csv"), index=False)