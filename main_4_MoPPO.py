# coding=utf-8
import random
import torch
import yaml
from typing import Dict
import os, sys, time
import numpy as np
import multiprocessing
import pandas as pd
from functools import partial
import argparse

os.chdir(sys.path[0])

from env.Env_4_MultiObj import BOOMENV as MultiObj_Env
from PPO.MultiObjective_PPO2 import MultiObjective_PPO

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


####################### MultiObj PPO2 Start ####################
def multiobj_ppo_run_once(config, seed=300):
    times = 15
    max_episodes = config["max_episodes"] * times
    all_test_states, all_test_ppas, all_test_projs = [], [], []

    setup_seed(seed)
    env = MultiObj_Env(config, device)
    agent = MultiObjective_PPO(config, env, max_episodes, device)

    for i in range(max_episodes):
        agent.run_an_episode()
        agent.schedule_lr(i)    # Deduce the learning rete.

        if (i+1) % times == 0:
            test_states, test_ppas, test_projs = agent.test()    # Return projections for all preferences
            all_test_states.append(test_states)
            all_test_ppas.append(test_ppas)
            all_test_projs.append(test_projs)
    
    all_test_states = np.stack(all_test_states)
    all_test_ppas = np.stack(all_test_ppas)
    all_test_projs = np.stack(all_test_projs).transpose()     # shape(num_of_prefs,max_episodes/times)

    best_idx = all_test_projs.argmax(axis=1)
    idx = [0,1,2]
    best_design = all_test_states[best_idx,idx]       # shape(num_of_prefs,n)
    best_design_ppa = all_test_ppas[best_idx,idx]     # shape(num_of_prefs,m)
    best_design_proj = all_test_projs[idx,best_idx]     # shape(num_of_prefs,)

    return all_test_projs, best_design, best_design_ppa, best_design_proj

def multiobj_ppo_run_multiprocess(config, seed):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]
    # seeds = [305,306,307,308]

    func = partial(multiobj_ppo_run_once, config)
    with multiprocessing.Pool(3) as pool:
        results = pool.map(func, seeds)
    all_projs, best_design, best_design_ppa, best_design_proj = zip(*results)       # list(num_of_seeds,)
    # for i in range(num_seed):
    #     all_projs, best_design, best_design_ppa, best_design_proj = multiobj_ppo_run_once(config, seeds[i])

    all_projs = np.stack(all_projs, axis=1)      # shape(num_of_prefs,num_of_seeds,max_episodes/times)
    mean = all_projs.mean(axis=1)          # shape(num_of_prefs, max_episodes/times)

    best_design = np.stack(best_design, axis=1)      # shape(num_of_prefs, num_of_seeds, n)
    best_design_ppa = np.stack(best_design_ppa, axis=1)      # shape(num_of_prefs, num_of_seeds, m)
    best_design_proj = np.stack(best_design_proj, axis=1)   # shape(num_of_prefs, num_of_seeds)

    return mean, best_design, best_design_ppa, best_design_proj
    
####################### MultiObj PPO2 End ####################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', default = None)
    args = parser.parse_args()

    seed = 300
    input_width = args.width
    width = "5W" if input_width is None else input_width
    width_pref = width+"_721"  # Only the "xW" makes sense, where the "x" can be the number 1~5.
    config_ppo = get_configs("config/config_ppo_"+width_pref+".yml")
    config_sac = get_configs("config/config_sac_"+width_pref+".yml")

    start_time = time.time()
    mean, best_design, best_design_ppa, best_design_proj = multiobj_ppo_run_multiprocess(config_ppo, seed)
    end_time = time.time()
    multiobj_ppo_running_time = end_time - start_time

    # Save all mean values.
    data={
        'MoPPO_mean'+width+'_217': mean[0],
        'MoPPO_mean'+width+'_361': mean[1],
        'MoPPO_mean'+width+'_721': mean[2]
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "MoPPO_mean_"+width+".csv"), index=False)

    # Save best_proj.
    data={
        'MoPPO_best_proj'+width+'_217': best_design_proj[0],
        'MoPPO_best_proj'+width+'_361': best_design_proj[1],
        'MoPPO_best_proj'+width+'_721': best_design_proj[2],
        "MoPPO_run_time": multiobj_ppo_running_time
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "MoPPO_best_proj"+width+".csv"), index=False)

    # Save best_design & PPA.
    data={
        'MoPPO_best_design'+width+'_217': best_design[0].tolist(),
        'MoPPO_best_design_ppa'+width+'_217': best_design_ppa[0].tolist(),
        'MoPPO_best_design'+width+'_361': best_design[1].tolist(),
        'MoPPO_best_design_ppa'+width+'_361': best_design_ppa[1].tolist(),
        'MoPPO_best_design'+width+'_721': best_design[2].tolist(),
        'MoPPO_best_design_ppa'+width+'_721': best_design_ppa[2].tolist()
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "MoPPO_best_design_and_ppa_"+width+".csv"), index=False)