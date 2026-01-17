import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import xgboost as xgb
from training import SimpleNN, train_nn, train_tree, train_tree_2
from scipy.stats import bernoulli
from scipy.optimize import minimize, differential_evolution
from skopt import gp_minimize
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader, TensorDataset
from utils import ols, active_logistic_pointestimate, logistic_cov


class GeneralizedPathOptimizer_Torch:
    """
    PyTorch-based 2D optimizer for (rho, gamma) using Adam.
    Handles the robust variance objective with automatic differentiation.
    """
    def __init__(self, e_hat_burn, pi, nb, n, constraint_sum, device='cuda', gamma_max=1.0):
        # e_hat_burn is the residual |Y - f(X)|
        self.e_hat_burn = torch.tensor(e_hat_burn, dtype=torch.float32).to(device)
        self.pi = torch.tensor(pi, dtype=torch.float32).to(device)
        self.nb = nb
        self.n = n
        self.constraint_sum = torch.tensor(constraint_sum, dtype=torch.float32).to(device)
        self.ratio = nb / n
        self.device = device
        self.gamma_max = gamma_max
        
        self.mask = self.pi > 0
        self.n_nonzero = torch.sum(self.mask)

    def compute_path(self, r, gamma):
        """
        Computes the Power-Mean path. Normalizes to maintain budget feasibility.
        """
        eps = 1e-8
        if gamma < 1e-4:
            # Limit case: Geometric path (log-linear)
            raw = (self.pi ** (1 - r)) * (self.ratio ** r)
        else:
            # Power-Mean Geodesic path
            raw = ((1 - r) * (self.pi ** gamma) + r * (self.ratio ** gamma)) ** (1 / gamma)
        
        # Strict normalization to preserve budget E[pi] = nb/n
        probs = (raw / torch.sum(raw)) * self.nb
        return torch.clamp(probs, 1e-10, 1.0)

    def optimize(self, epochs=250, lr=0.05):
        # Sigmoid reparameterization to enforce [0, 1] bounds
        r_param = torch.tensor(0.0, requires_grad=True, device=self.device)
        gamma_param = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([r_param, gamma_param], lr=lr)
        best_loss = float('inf')
        best_r, best_gamma = 0.5, 0.5

        for epoch in range(epochs):
            optimizer.zero_grad()
            r = torch.sigmoid(r_param)
            gamma = torch.sigmoid(gamma_param) * self.gamma_max
            
            p = self.compute_path(r, gamma)
            
            # FIXED: Squared error used to reflect variance
            empirical_term = torch.sum((self.e_hat_burn[self.mask] ** 2) / p[self.mask])
            
            # Robustness penalty based on L2 uncertainty set
            robust_term = self.constraint_sum * torch.sqrt(torch.sum(1.0 / (p[self.mask] ** 2)))
            
            loss = (empirical_term + robust_term) / self.n_nonzero
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_r = r.item()
                best_gamma = gamma.item()

        return best_loss, best_r, best_gamma

class PathCurvatureOptimizer_l2:
    """
    Logic for 2D optimization over (rho, gamma) using multiple search methods.
    """
    def __init__(self, e_hat_burn, pi, nb, n, constraint_sum, gamma_max=1.0):
        self.e_hat_burn = e_hat_burn
        self.pi = pi
        self.nb = nb
        self.n = n
        self.constraint_sum = constraint_sum
        self.ratio = nb / n
        self.gamma_max = gamma_max
        self.nonzero_mask = self.pi > 0
        self.n_nonzero = np.sum(self.nonzero_mask)
        
        self.optimal_r = None
        self.optimal_gamma = None
        self.optimal_probs = None

    def compute_path(self, r: float, gamma: float) -> np.ndarray:
        if gamma < 1e-5:
            raw = (self.pi ** (1 - r)) * (self.ratio ** r)
        else:
            raw = ((1 - r) * (self.pi ** gamma) + r * (self.ratio ** gamma)) ** (1 / gamma)
        probs = (raw / np.sum(raw)) * self.nb
        return np.clip(probs, 1e-10, 1.0)

    def objective_function(self, r: float, gamma: float) -> float:
        denominator = self.compute_path(r, gamma)
        # FIXED: Squared error (e^2/p) as per original variance derivation
        terms = (self.e_hat_burn[self.nonzero_mask] ** 2) / denominator[self.nonzero_mask]
        penalty = self.constraint_sum * np.sqrt(np.sum(1 / (denominator[self.nonzero_mask] ** 2)))
        return (np.sum(terms) + penalty) / self.n_nonzero if self.n_nonzero > 0 else 0

    def _optimize_grid(self, r_steps, gamma_steps):
        r_vals = np.linspace(0, 1, r_steps)
        gamma_vals = np.linspace(0, self.gamma_max, gamma_steps)
        best_v, best_r, best_g = float('inf'), 1.0, 0.0
        for g in gamma_vals:
            for r in r_vals:
                v = self.objective_function(r, g)
                if v < best_v: best_v, best_r, best_g = v, r, g
        return best_v, best_r, best_g

    def _optimize_scipy(self):
        bounds = [(0, 1), (0, self.gamma_max)]
        best_v, best_r, best_g = float('inf'), 1.0, 0.0
        def func(x): return self.objective_function(x[0], x[1])
        for x0 in [[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]]:
            res = minimize(func, x0=x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_v: best_v, best_r, best_g = res.fun, res.x[0], res.x[1]
        res_de = differential_evolution(func, bounds=bounds, seed=42)
        if res_de.fun < best_v: best_v, best_r, best_g = res_de.fun, res_de.x[0], res_de.x[1]
        return best_v, float(best_r), float(best_g)

    def _optimize_bayesian(self):
        def func(x): return self.objective_function(x[0], x[1])
        res = gp_minimize(func, [(0.0, 1.0), (0.0, self.gamma_max)], n_calls=50, random_state=42)
        return res.fun, float(res.x[0]), float(res.x[1])

    def _optimize_torch(self, device):
        torch_opt = GeneralizedPathOptimizer_Torch(self.e_hat_burn, self.pi, self.nb, self.n, self.constraint_sum, device=device, gamma_max=self.gamma_max)
        return torch_opt.optimize()

    def optimize(self, r_steps=50, gamma_steps=50, device='cuda'):
        results = [
            self._optimize_grid(r_steps, gamma_steps),
            self._optimize_scipy(),
            self._optimize_bayesian(),
            self._optimize_torch(device=device)
        ]
        best_idx = min(range(len(results)), key=lambda i: results[i][0])
        val, r, g = results[best_idx]
        self.optimal_r, self.optimal_gamma = r, g
        self.optimal_probs = self.compute_path(r, g)
        return val, r, g

class MinMaxOptimizer_l2:
    """
    Robust sampling optimizer. 
    Supports 1D geometric path or 2D power-mean path extension.
    """
    def __init__(self, e_hat: np.ndarray, pi: np.ndarray, nb: float, n: int, constraint_sum: float, use_path: bool = False, gamma_max: float = 1.0):
        self.e_hat = e_hat
        self.pi = pi
        self.nb = nb
        self.n = n
        self.constraint_sum = constraint_sum
        self.use_path = use_path
        self.gamma_max = gamma_max
        self.nonzero_mask = self.pi != 0
        self.n_nonzero = np.sum(self.nonzero_mask)
        
        self.optimal_r = None
        self.optimal_gamma = None
        self.optimal_probs = None

        if self.use_path:
            self._path_optimizer = PathCurvatureOptimizer_l2(e_hat, pi, nb, n, constraint_sum, gamma_max)

    def compute_probs(self, r: float) -> np.ndarray:
        # Standard geometric path interpolation
        ratio = self.nb / self.n
        raw = (self.pi ** (1 - r)) * (ratio ** r)
        probs = (raw / np.sum(raw)) * self.nb
        return np.clip(probs, 0.0, 1.0)

    def objective_function(self, r: float) -> float:
        denominator = self.compute_probs(r)
        # FIXED: Square the error term to minimize variance
        terms = (self.e_hat[self.nonzero_mask] ** 2) / denominator[self.nonzero_mask]
        penalty = self.constraint_sum * np.sqrt(np.sum(1 / (denominator[self.nonzero_mask] ** 2)))
        return (np.sum(terms) + penalty) / self.n_nonzero if self.n_nonzero > 0 else 0

    def optimize(self, r_bounds: Tuple[float, float] = (0, 1), r_steps: int = 100, device='cuda') -> Tuple[float, float]:
        if self.use_path:
            # 2D Optimization over r and gamma
            val, r, g = self._path_optimizer.optimize(r_steps=r_steps, device=device)
            self.optimal_r, self.optimal_gamma = r, g
            self.optimal_probs = self._path_optimizer.optimal_probs
            return val, r
        
        # Original 1D grid search over r
        r_values = np.linspace(r_bounds[0], r_bounds[1], r_steps)
        best_v, best_r = float('inf'), 1.0
        for r in r_values:
            v = self.objective_function(r)
            if v < best_v: best_v, best_r = v, r
        
        self.optimal_r = best_r
        self.optimal_gamma = 0.0 # Standard Geometric path endpoint
        self.optimal_probs = self.compute_probs(best_r)
        return best_v, best_r

    def get_optimal_probs(self) -> np.ndarray:
        if self.optimal_probs is None:
            raise ValueError("Call optimize() first.")
        return self.optimal_probs

def process_fold(args):
    """Process a single fold of cross-validation"""
    X_train, Y_train, X_test, Y_test, model, bg, c, device = args
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Get predictions
    Yhat = model.predict(xgb.DMatrix(X_test))
    error_train = np.abs(Y_train - model.predict(xgb.DMatrix(X_train)))
    error_train_tensor = torch.tensor(error_train, dtype=torch.float32).view(-1,1).to(device)
    
    dataset = TensorDataset(X_train_tensor, error_train_tensor)
    loader = DataLoader(dataset, batch_size=min(256, len(X_train)), shuffle=True)
    
    # Train neural network (reduced epochs for CV speed)
    nn_ = SimpleNN(X_train_tensor.shape[1]).to(device)
    train_nn(nn_, loader, epochs=100, lr=0.05)
    
    # Get uncertainty predictions
    with torch.no_grad():
        uncertainty = np.abs(nn_(X_test_tensor).cpu().numpy().flatten())
    
    uncertainty = np.clip(uncertainty, 0, 1)
    n = len(Y_test)
    nb = bg * n
    error = uncertainty.copy()
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimizer.optimize()
    
    # Get optimal probs from optimizer
    probs = optimizer.get_optimal_probs()
    
    num_trials = 50  # Reduced for speed
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat + (Y_test - Yhat) * (xi / probs_)
    std = np.mean(np.std(active_robust_labels, axis=1))
    
    return std

def constraint_cross_validation(X_bi: np.ndarray, Y_bi: np.ndarray, model, cv_list: np.ndarray, k: int, budgets: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(X_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            # Prepare arguments for parallel processing
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                process_args.append((X_train, Y_train, X_test, Y_test, model, bg, c, device))
            
            # Process folds sequentially (faster than multiprocessing for NN training)
            fold_results = [process_fold(args) for args in process_args]
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list

def process_fold_regression(args):
    """Process a single fold of cross-validation"""
    X_train, income_features_unlabeled_train, Y_train, X_test, income_features_unlabeled_test, Y_test, model, bg, c, Hessian_inv, enc, device = args

    Yhat = model.predict(xgb.DMatrix(income_features_unlabeled_test))
    # Get predictions

    error_train = np.abs(Y_train - model.predict(xgb.DMatrix(income_features_unlabeled_train)))
    xgb_err = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=20)
    xgb_err.fit(X_train, error_train)
    predicted_errs = np.clip(np.abs(xgb_err.predict(X_test)), 0, np.inf)
    h = Hessian_inv[:,0]
    uncertainty = np.abs(h.dot(X_test.T)) * predicted_errs
    
    uncertainty = np.clip(uncertainty, 0, 1)
    n = len(Y_test)
    nb = bg * n
    error = uncertainty.copy()
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimizer.optimize()
    
    # Get optimal probs from optimizer
    probs = optimizer.get_optimal_probs()
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat + (Y_test - Yhat) * (xi / probs)
    
    # Compute OLS for each trial
    pointest_active_robust = np.array([ols(X_test, active_robust_labels[i]) for i in range(num_trials)])
    grads = np.array([(np.dot(X_test, pointest_active_robust[i]) - active_robust_labels[i])[:, np.newaxis] * X_test for i in range(num_trials)])

    # Compute covariance for each trial
    V = np.array([np.cov(grads[i].T) for i in range(num_trials)])
    Sigma_active_robust = np.array([Hessian_inv @ V[i] @ Hessian_inv for i in range(num_trials)])
    std = np.mean(np.sqrt(Sigma_active_robust[:, 0, 0]))
    
    return std



def constraint_cross_validation_regression(X_bi, income_features_unlabeled_bi: np.ndarray, Y_bi: np.ndarray, model, cv_list: np.ndarray, k: int, budgets: np.ndarray, Hessian_inv: np.ndarray, enc, device: torch.device) -> np.ndarray:

    optimal_c_list = []
    n_samples = len(Y_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            # Prepare arguments for parallel processing
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                income_features_unlabeled_train = income_features_unlabeled_bi[train_idx]
                Y_train = Y_bi[train_idx]
                X_test = X_bi[test_idx]
                income_features_unlabeled_test = income_features_unlabeled_bi[test_idx]
                Y_test = Y_bi[test_idx]
                process_args.append((X_train, income_features_unlabeled_train, Y_train, X_test, income_features_unlabeled_test, Y_test, model, bg, c, Hessian_inv, enc, device))
            
            # Process folds in parallel
            with Pool(processes=k) as pool:
                fold_results = pool.map(process_fold_regression, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list


def process_fold_bias(args):
    """Process a single fold of cross-validation"""
    X_train, Y_train, Yhat_train, X_test, Y_test, Yhat_test, bg, c, device = args
    
    # Get predictions
    error_train = np.abs(Y_train - Yhat_train)
    tree = train_tree_2(X_train, error_train)
    
    error = tree.predict(X_test)
    uncertainty = (1 - 2 * np.abs(X_test - 0.5)).reshape(-1)

    n = len(Y_test)
    nb = bg * n
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimizer.optimize()
    
    # Get optimal probs from optimizer
    probs = optimizer.get_optimal_probs()
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat_test + (Y_test - Yhat_test) * (xi / probs_)
    std = np.mean(np.std(active_robust_labels, axis=1))
    
    return std

def constraint_cross_validation_bias(X_bi: np.ndarray, Y_bi: np.ndarray, Yhat_bi: np.ndarray, cv_list: np.ndarray, k: int, budgets: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(X_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                Yhat_train = Yhat_bi[train_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                Yhat_test = Yhat_bi[test_idx]
                process_args.append((X_train, Y_train, Yhat_train, X_test, Y_test, Yhat_test, bg, c, device))
            
            # Process folds in parallel
            with Pool(processes=k) as pool:
                fold_results = pool.map(process_fold_bias, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list


def process_fold_politeness(args):
    """Process a single fold of cross-validation"""
    confidence_train, X_train, Y_train, Yhat_train, confidence_test, X_test, Y_test, Yhat_test, bg, c, h, device = args
    
    # Get predictions
    error_train = np.abs(Y_train - Yhat_train)
    tree = train_tree_2(confidence_train, error_train**2)
    
    error = np.sqrt(tree.predict(confidence_test)) * np.abs(X_test.dot(h))
    uncertainty = (1 - 2 * np.abs(confidence_test - 0.5)).reshape(-1)

    n = len(Y_test)
    nb = bg * n
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimizer.optimize()
    
    # Get optimal probs from optimizer
    probs = optimizer.get_optimal_probs()
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    
    # Compute std across trials
    stds = []
    for i in range(num_trials):
        # Use single trial weights
        weights = xi[i] / probs_
        pointest = active_logistic_pointestimate(X_test, Y_test, Yhat_test, weights, 1)
        Sigmahat = logistic_cov(pointest, X_test, Y_test, Yhat_test, weights, 1)
        stds.append(np.sqrt(Sigmahat[0, 0]))
    
    std = np.mean(stds)
    
    return std

def constraint_cross_validation_politeness(confidence_bi: np.ndarray, X_bi: np.ndarray, Y_bi: np.ndarray, Yhat_bi: np.ndarray, cv_list: np.ndarray, k: int, budgets: np.ndarray, h: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(confidence_bi)
    fold_size = n_samples // k

    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            process_args = []
            for train_idx, test_idx in fold_indices:
                confidence_train = confidence_bi[train_idx]
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                Yhat_train = Yhat_bi[train_idx]
                confidence_test = confidence_bi[test_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                Yhat_test = Yhat_bi[test_idx]
                process_args.append((confidence_train, X_train, Y_train, Yhat_train, confidence_test, X_test, Y_test, Yhat_test, bg, c, h, device))
            
            # Process folds in parallel
            with Pool(processes=k) as pool:
                fold_results = pool.map(process_fold_politeness, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list