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

os.chdir(sys.path[0])

from env.env_no_option import BOOMENV as Env_new
from env.env_with_option import BOOMENV as HRLEnv
from drawer.PlotLearningCurve import plot_result, ema_plotting
from PPO.PPO2 import PPO2_v0, PPO2_v1
from SAC.SAC import SAC
from Comparisons.MoEnDSE import BagGBRT
from Comparisons.MyDatasetCreation.problem import DesignSpaceProblem

import argparse

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

####################### PPOV0 Multi-processor Start #####################
def ppov0_run_once(config, seed=300):
    times = 15
    max_episodes = config["max_episodes"] * times
    all_rewards = np.zeros((config["max_episodes"], 1))  # (num_episode, num_seeds)

    setup_seed(seed)
    env = HRLEnv(config, device)
    agent = PPO2_v0(config, env, max_episodes, device)
    total_rewards, episodes, final_microarchs, norm_ppas = [], [], [], []

    # Run a whole process with a random seed.
    for i in range(max_episodes):
        agent.run_an_episode()
        agent.schedule_lr(i)    # Deduce the learning rete.
        total_reward = env.get_cum_reward()
        total_rewards.append(total_reward)  # Can only depict the distribution of the values.
        final_microarchs.append(env._explored_designs[-1][0])
        norm_ppas.append(env._explored_designs[-1][1])
        print('EP:{} total_reward:{}'.
                format(i + 1, total_reward))
    entropy = np.array(agent.entropy)
    all_rewards = np.array(total_rewards).reshape(-1,times).max(axis=1)

    best_design_episode = np.array(total_rewards).argmax()
    best_design = final_microarchs[best_design_episode]
    best_design_ppa = norm_ppas[best_design_episode]
    best_design_ppa = env.renormalize_ppa(best_design_ppa)
    best_design_ppa[1:] = -best_design_ppa[1:]
    proj = total_rewards[best_design_episode]
    
    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))
    return np.array(total_reward), all_rewards, entropy, best_design, best_design_ppa, proj

def ppov0_run_multiprocess(config, seed):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]
    # seeds = [305,306,307,308]

    func = partial(ppov0_run_once, config)
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, seeds)
    found_designs_projs, all_rewards, entropy, best_design, best_design_ppa, proj = zip(*results)
    # for i in range(len(seeds)):
    #     found_designs_projs, all_rewards, test_mse = baggbrt_run_once(config, seeds[i])
    all_rewards = np.array(all_rewards).transpose()
    total_rewards = np.array(found_designs_projs)[-1].transpose().tolist()  # All found designs at the last trial.
    proj = np.array(proj).reshape(-1)

    # Get the means and standard deviations of episode rewards.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    return mean, std, entropy, best_design, best_design_ppa, proj
###################### PPOV0 Multi-processor End ######################

####################### PPOV1 Multi-processor Start #####################
def ppov1_run_once(config, seed=300):
    max_episodes = config["max_episodes"]
    all_best_projs = np.zeros((max_episodes, 1))  # (num_episode, num_seeds)

    setup_seed(seed)
    env = Env_new(config, device)
    agent = PPO2_v1(config, env, device, embedding=True)  # Emperiments show that embedding is essential.
    best_state_projs, episodes, final_microarchs, norm_ppas = [], [], [], []

    # Run a whole process with a random seed.
    for i in range(max_episodes):
        agent.run_an_episode()
        all_norm_ppa = np.array([row[1] for i, row in enumerate(env._explored_designs)])
        all_projs = env.get_projection(all_norm_ppa)
        best_idx_in_episode = all_projs.argmax()
        best_state_proj_in_episode = all_projs[best_idx_in_episode]
        all_best_projs[i, 0] = best_state_proj_in_episode.copy()
        best_state_projs.append(best_state_proj_in_episode)
        final_microarchs.append(env._explored_designs[best_idx_in_episode][0])
        norm_ppas.append(env._explored_designs[best_idx_in_episode][1])
        print('PPO_v1 EP:{} best_state_proj:{}'.
                format(i + 1, best_state_proj_in_episode))
        
    entropy = np.array(agent.entropy)

    best_design_episode = np.array(best_state_projs).argmax()
    best_design = final_microarchs[best_design_episode]
    best_design_ppa = np.array(norm_ppas)[best_design_episode]
    best_design_ppa = env.renormalize_ppa(best_design_ppa)
    best_design_ppa[1:] = -best_design_ppa[1:]
    proj = best_state_projs[best_design_episode].reshape(-1)
    sim_rounds = agent.total_num_explored_designs
    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))
    return all_best_projs, entropy, best_design, best_design_ppa, proj, sim_rounds

def ppov1_run_multiprocess(config, seed):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]
    # seeds = [305,306,307,308]

    func = partial(ppov1_run_once, config)
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, seeds)
    all_projs, entropy, best_design, best_design_ppa, proj, sim_rounds = zip(*results)
    # for i in range(len(seeds)):
    #     found_designs_projs, all_projs, test_mse = baggbrt_run_once(config, seeds[i])
    
    all_projs = np.array(all_projs).squeeze(-1).transpose()
    proj = np.array(proj).reshape(-1)

    # Get the means and standard deviations of episode rewards.
    mean = all_projs.mean(axis=1)
    std = all_projs.std(axis=1)
    return mean, std, entropy, best_design, best_design_ppa, proj, sim_rounds
####################### PPOV1 Multi-processor End #####################

####################### SAC Multi-processor Start #####################
def sac_run_once(config, buffer_TD_error, embedding, priority, k_step_update, batch_size, gamma, alpha, tau, seed):
    max_episodes = config["max_episodes"]
    all_best_projs = np.zeros((max_episodes, 1))  # (num_episode, num_seeds)

    setup_seed(seed)
    env = Env_new(config, device)
    agent = SAC(config, env, device,
                embedding=embedding,                # Embedding will decrease the num of training data to some extent.
                buffer_TD_error=buffer_TD_error,
                priority=priority,
                k_step_update=k_step_update,
                gamma=gamma,
                tau=tau,
                batch_size=batch_size,
                alpha=alpha)  # Emperiments show that embedding is essential.
    best_state_projs, episodes, final_microarchs, norm_ppas = [], [], [], []

    # Run a whole process with a random seed.
    for i in range(max_episodes):
        agent.run_an_episode()
        all_norm_ppa = np.array([row[1] for i, row in enumerate(env._explored_designs)])
        all_projs = env.get_projection(all_norm_ppa)
        best_idx_in_episode = all_projs.argmax()
        best_state_proj_in_episode = all_projs[best_idx_in_episode]
        best_state_projs.append(best_state_proj_in_episode)
        final_microarchs.append(env._explored_designs[best_idx_in_episode][0])
        norm_ppas.append(env._explored_designs[best_idx_in_episode][1])
        print('SAC EP:{} best_state_proj:{}'.
                format(i + 1, best_state_proj_in_episode))
        
    entropy = np.array(agent.entropy)

    best_design_episode = np.array(best_state_projs).argmax()
    best_design = final_microarchs[best_design_episode]
    best_design_ppa = np.array(norm_ppas)[best_design_episode]
    best_design_ppa = env.renormalize_ppa(best_design_ppa)
    best_design_ppa[1:] = -best_design_ppa[1:]
    proj = best_state_projs[best_design_episode].reshape(-1)
    sim_rounds = agent.total_num_explored_designs
    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))
    return all_best_projs, entropy, best_design, best_design_ppa, proj, sim_rounds

def sac_run_multiprocess(config, buffer_TD_error='min', embedding=True, priority=False, k_step_update=8, batch_size=25, gamma=0.8, alpha=0.5, tau=0.005, seed=300):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]

    func = partial(sac_run_once, config, buffer_TD_error, embedding, priority, k_step_update, batch_size, gamma, alpha, tau)
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, seeds)
    all_projs, entropy, best_design, best_design_ppa, proj, sim_rounds = zip(*results)
    # for i in range(len(seeds)):
    #     found_designs_projs, all_projs, test_mse = baggbrt_run_once(config, seeds[i])
    
    all_projs = np.array(all_projs).squeeze(-1).transpose()
    proj = np.array(proj).reshape(-1)

    # Get the means and standard deviations of episode rewards.
    mean = all_projs.mean(axis=1)
    std = all_projs.std(axis=1)
    return mean, std, entropy, best_design, best_design_ppa, proj, sim_rounds
####################### SAC Multi-processor End #####################

# ---------------- BagGBRT Multiprocessing ---------------- #
# BagGBRT runs very slowly, so I split it into two functions for multiprocessing.
def baggbrt_run_once(config, seed=300):
    max_episodes = config["max_episodes"]
    times = 15
    train_iter = times * max_episodes
    all_rewards = np.zeros((max_episodes, 1))  # [num_episode, num_seeds]

    setup_seed(seed)
    problem = DesignSpaceProblem(config)
    agent = BagGBRT(problem)
    found_designs, found_designs_ppa, found_designs_proj = agent.train(
        train_iter)  # "found_designs_ppa" is normalized PPA.
    
    # train_mse = np.array(agent.train_mse).reshape(1,-1)
    test_mse = np.array(agent.test_mse)

    all_rewards = np.array(found_designs_proj).reshape(max_episodes,times).max(axis=1)

    best_design_episode = np.array(found_designs_proj).argmax()
    best_design = found_designs[best_design_episode]
    best_design_ppa = torch.tensor(found_designs_ppa)[best_design_episode]
    best_design_ppa = problem.renormalize_ppa(best_design_ppa)  # Tensor
    best_design_ppa[1:] = -best_design_ppa[1:]
    proj = found_designs_proj[best_design_episode]

    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa.numpy(), proj))
    return np.array(found_designs_proj), all_rewards, test_mse, best_design, best_design_ppa.numpy(), proj


def baggbrt_run_multiprocess(config, seed):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]
    mean_episode_rewards, std_episode_rewards = [], []

    func = partial(baggbrt_run_once, config)
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, seeds)
    found_designs_projs, all_rewards, test_mse, best_design, best_design_ppa, proj = zip(*results)
    # for i in range(len(seeds)):
    #     found_designs_projs, all_rewards, test_mse = baggbrt_run_once(config, seeds[i])
    all_rewards = np.array(all_rewards).transpose()
    total_rewards = np.array(found_designs_projs)[-1].transpose().tolist()  # All found designs at the last trial.
    proj = np.array(proj)

    # Get the means and standard deviations of episode rewards.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    return mean, std, test_mse, best_design, best_design_ppa, proj
# ---------------- BagGBRT Multiprocessing END---------------- #

if __name__ == '__main__':
    # Read Width and Preference
    parser = argparse.ArgumentParser()
    parser.add_argument('--width_pref', default = None)
    args = parser.parse_args()

    seed = 300
    width_pref = "2W_721" if args.width_pref is None else args.width_pref  # Change this to modify the names of "config" files and csv file.
    config_sac = get_configs("config/config_sac_"+width_pref+".yml")
    config_ppo = get_configs("config/config_ppo_"+width_pref+".yml")

    # Count the running time
    start_time = time.time()
    sac_mean, sac_std, sac_entropy, sac_best_design, sac_best_design_ppa, sac_best_proj, sac_sim_rounds = sac_run_multiprocess(config_sac, seed=seed)
    end_time = time.time()
    sac_running_time = end_time - start_time
    start_time = time.time()   
    baggbrt_mean, baggbrt_std, test_mse, baggbrt_best_design, baggbrt_best_design_ppa, baggbrt_best_proj = baggbrt_run_multiprocess(config_sac, seed)
    end_time = time.time()
    baggbrt_running_time = end_time - start_time
    start_time = time.time()
    ppov0_mean, ppov0_std, ppov0_entropy, ppov0_best_design, ppov0_best_design_ppa, ppov0_best_proj = ppov0_run_multiprocess(config_ppo, seed)
    end_time = time.time()
    ppov0_running_time = end_time - start_time
    print("PPO_v0 time: {}".format(ppov0_running_time))
    start_time = time.time()     # The UNIT is second.
    ppov1_mean, ppov1_std, ppov1_entropy, ppov1_best_design, ppov1_best_design_ppa, ppov1_best_proj, ppov1_sim_rounds = ppov1_run_multiprocess(config_ppo, seed)
    end_time = time.time()
    ppov1_running_time = end_time - start_time

    # Plot Rewards
    episodes = np.array([i for i in range(len(ppov0_mean))])
    mean = np.concatenate((
                        ppov0_mean.reshape(1, -1), 
                        ppov1_mean.reshape(1, -1), 
                        sac_mean.reshape(1, -1), 
                        baggbrt_mean.reshape(1, -1)
                        ), 
                        axis=0)
    std = np.concatenate((
                        ppov0_std.reshape(1, -1), 
                        ppov1_std.reshape(1, -1), 
                        sac_std.reshape(1, -1), 
                        baggbrt_std.reshape(1, -1)
                        ), axis=0)
    label=['ppov0',
           'ppov1',
           'sac',
           'baggbrt'
           ]
    ema_plotting(episodes, mean, 15, 'Avg_Reward', 'avg_reward', os.path.join(config_sac["reports_folder_path"], "avg_reward_"+width_pref+".pdf"), label=label)
    

    # Save best_proj.
    data={
        'ppov0_best_proj': ppov0_best_proj,
        'ppov1_best_proj': ppov1_best_proj,
        'sac_best_proj': sac_best_proj,
        'baggbrt_best_proj': baggbrt_best_proj
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "best_proj_"+width_pref+".csv"), index=False)

    # Save best_design & PPA.
    data={
        'ppov0_best_design': ppov0_best_design,
        'ppov0_best_design_ppa': ppov0_best_design_ppa,
        'ppov1_best_design': ppov1_best_design,
        'ppov1_best_design_ppa': ppov1_best_design_ppa,
        'sac_best_design': sac_best_design,
        'sac_best_design_ppa': sac_best_design_ppa,
        'baggbrt_best_design': baggbrt_best_design,
        'baggbrt_best_design_ppa': baggbrt_best_design_ppa
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "best_design_and_ppa_"+width_pref+".csv"), index=False)

    # For simulation rounds.
    data={
        'ppov1_sim_rounds': ppov1_sim_rounds,
        'sac_sim_rounds': sac_sim_rounds,
        'baggbrt_sim_time': baggbrt_running_time,
        'ppov0_sim_time': ppov0_running_time,
        'ppov1_sim_time': ppov1_running_time,
        'sac_sim_time': sac_running_time
    }
    pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "sim_rounds_"+width_pref+".csv"), index=False)

    # Plot MSE
    # mse_label = ["trial 1", "trial 2", "trial 3"]
    mse_label = None
    mse_length = np.array([i for i in range(test_mse[0].shape[-1])])
    test_mse_mean = np.array(test_mse).mean(axis=0).reshape(-1)
    plot_result(mse_length, test_mse_mean, None, 'mse', os.path.join(config_sac["reports_folder_path"],"Entropy/MoEnDSE_"+str(width_pref)+".pdf"), label=mse_label, xlabel='exploration rounds', dot=False)

