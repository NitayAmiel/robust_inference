#!/usr/bin/env python
# coding: utf-8

# IPython magic commands removed for script execution
import pandas as pd
import numpy as np
from scipy.stats import norm, bernoulli
import json
import warnings; warnings.simplefilter('ignore')
from tqdm import tqdm
from utils import opt_mean_tuning, make_ess_coverage_plot_intro
from training import train_tree, train_tree_2
from utils_optimization import MinMaxOptimizer_l2, constraint_cross_validation_bias
import torch
import xgboost as xgb


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading and preprocessing
data = pd.read_csv('data/bias_dataset.csv')
data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
leaning = 'right'  # 'left' or 'right'
Yhat_string = data["label_gpt4o"].to_numpy()
Y_string = data["bias_text"].to_numpy()
confidence = data["confidence_in_prediction_gpt-4o"].to_numpy()
nan_indices = list(np.where(pd.isna(confidence))[0]) + list(np.where(pd.isna(Yhat_string))[0])
good_indices = list(set(range(len(data))) - set(nan_indices))
confidence = confidence[good_indices]
Yhat_string = Yhat_string[good_indices]
Y_string = Y_string[good_indices]
n = len(Yhat_string)
if leaning == 'left':
    dict_map = {"A": 1, "B": 0, "C": 0, "left": 1, "center": 0, "right": 0}
elif leaning == 'right':
    dict_map = {"A": 0, "B": 0, "C": 1, "left": 0, "center": 0, "right": 1}
Yhat = np.array([dict_map[Yhat_string[i]] for i in range(n)])
Y = np.array([dict_map[Y_string[i]] for i in range(n)])
confidence = confidence.reshape(len(confidence), 1)


# Burn-in and uncertainty estimation
bi = 200
n_rem = n - bi
tree = train_tree_2(confidence[:bi], np.abs((Y - Yhat)[:bi]))
error = tree.predict(confidence[bi:])
uncertainties = (1 - confidence[bi:]).reshape(-1)
uncertainties = (1 - 2 * np.abs(confidence[bi:] - 0.5)).reshape(-1)


# ============================================================================
# Experiment comparing Regular Robust vs Path Robust across configurations
# ============================================================================
alpha = 0.1  # desired error level for confidence interval
num_trials = 500

true_prevalence = np.mean(Y)

budgets = np.linspace(0.35, 0.6, 10)
cv_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]
k = 5
c_list = constraint_cross_validation_bias(confidence[:bi], Y[:bi], Yhat[:bi], cv_list, k, budgets, device)

# Store optimizers to access optimal values later
optimizers_path = []  # use_path=True (2D optimization)
optimizers_regular = []  # use_path=False (1D optimization)

for i in range(len(budgets)):
    constraint_sum = c_list[i] * np.sqrt(n_rem)
    bg = budgets[i]
    eta = bg / np.mean(uncertainties)
    pi = np.clip(eta * uncertainties, 0.0, 1.0)
    nb = bg * n_rem
    
    # Use the unified interface with use_path=True (2D optimization)
    optimizer_path = MinMaxOptimizer_l2(error, pi, nb, n_rem, constraint_sum, use_path=True)
    optimizer_path.optimize()
    optimizers_path.append(optimizer_path)
    
    # Regular robust with use_path=False (1D optimization)
    optimizer_regular = MinMaxOptimizer_l2(error, pi, nb, n_rem, constraint_sum, use_path=False)
    optimizer_regular.optimize()
    optimizers_regular.append(optimizer_regular)
    
    # Print optimal parameters for this budget
    print(f"Budget {bg:.2f}: path_r={optimizer_path.optimal_r:.3f}, path_gamma={optimizer_path.optimal_gamma:.3f}, regular_r={optimizer_regular.optimal_r:.3f}")

results = []
columns = ["lb", "ub", "interval width", "coverage", "estimator", "$n_b$"]
temp_df = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)  # 4 estimators now

label = Y[:bi]

for j in range(len(budgets)):
    bg = budgets[j]
    eta = bg / np.mean(uncertainties)
    
    # Standard active (fixed small tau)
    tau_fixed = 0.01
    sampling_prob = uncertainties / np.mean(uncertainties) * bg
    sampling_prob = np.clip(sampling_prob, 0, 1)
    probs_active = np.clip((1 - tau_fixed) * sampling_prob + tau_fixed * bg, 0, 1)
    
    # Robust active with path (2D optimization)
    probs_robust_path = optimizers_path[j].get_optimal_probs()
    
    # Regular robust (1D optimization, use_path=False)
    probs_robust_regular = optimizers_regular[j].get_optimal_probs()

    for i in range(num_trials):
        # 1. Uniform Sampling
        xi_unif = bernoulli.rvs([bg] * n_rem)
        unif_label = Yhat[bi:] + (Y[bi:] - Yhat[bi:]) * xi_unif / bg
        unif_label = np.concatenate([label, unif_label])
        pointest = np.mean(unif_label)
        varhat = np.var(unif_label)
        l = pointest - norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        u = pointest + norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        coverage = (true_prevalence >= l) * (true_prevalence <= u)
        temp_df.loc[0] = l, u, u - l, coverage, "uniform", int(n_rem * bg + bi)

        # 2. Standard Active Sampling
        weights_active = np.zeros(n)
        weights_active[:bi] = 1
        weights_active[bi:] = bernoulli.rvs(probs_active) / probs_active
        
        sampling_ratio = (1 - probs_active) / probs_active
        lam = opt_mean_tuning(Y[bi:], Yhat[bi:], weights_active[bi:], sampling_ratio)
        Yhat_ = lam * Yhat
        active_label = Yhat_[bi:] + (Y[bi:] - Yhat_[bi:]) * weights_active[bi:]
        active_label = np.concatenate([label, active_label])
        pointest = np.mean(active_label)
        varhat = np.var(active_label)
        l = pointest - norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        u = pointest + norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        coverage = (true_prevalence >= l) * (true_prevalence <= u)
        temp_df.loc[1] = l, u, u - l, coverage, "active", int(n_rem * bg + bi)

        # 3. Robust Active Sampling with Path (2D optimization)
        xi = bernoulli.rvs(probs_robust_path)
        robust_path_label = Yhat[bi:] + (Y[bi:] - Yhat[bi:]) * xi / probs_robust_path
        robust_path_label = np.concatenate([label, robust_path_label])
        pointest = np.mean(robust_path_label)
        varhat = np.var(robust_path_label)
        l = pointest - norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        u = pointest + norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        coverage = (true_prevalence >= l) * (true_prevalence <= u)
        temp_df.loc[2] = l, u, u - l, coverage, "robust active (path)", int(n_rem * bg + bi)

        # 4. Regular Robust Active Sampling (1D optimization)
        xi = bernoulli.rvs(probs_robust_regular)
        robust_label = Yhat[bi:] + (Y[bi:] - Yhat[bi:]) * xi / probs_robust_regular
        robust_label = np.concatenate([label, robust_label])
        pointest = np.mean(robust_label)
        varhat = np.var(robust_label)
        l = pointest - norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        u = pointest + norm.ppf(1 - alpha / 2) * np.sqrt(varhat / n)
        coverage = (true_prevalence >= l) * (true_prevalence <= u)
        temp_df.loc[3] = l, u, u - l, coverage, "robust active", int(n_rem * bg + bi)

        results += [temp_df.copy()]

df = pd.concat(results, ignore_index=True)
df["coverage"] = df["coverage"].astype(bool)

make_ess_coverage_plot_intro(df, "prevalence $p_{\\mathrm{" + leaning + "}}$", f"llm_bias_ess_coverage_{leaning}_use_path.pdf", true_prevalence)

