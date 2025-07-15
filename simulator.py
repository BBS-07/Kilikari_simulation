import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from tqdm import tqdm
import os
import pickle



def generate_kilkari_style_reward_matrix(n, K=7, seed=42):
    """
    Simulates Kilkari-style reward matrix.
    """
    np.random.seed(seed)
    mu_matrix = np.zeros((n, K))
    tier1_frac, tier3_frac = 0.4059, 0.0699
    tier1_cutoff = int(tier1_frac * n)
    tier3_cutoff = int(tier3_frac * n)
    perm = np.random.permutation(n)
    tier1_ids = perm[:tier1_cutoff]
    tier3_ids = perm[-tier3_cutoff:]
    tier2_ids = perm[tier1_cutoff:-tier3_cutoff]
    mu_matrix[tier1_ids, :] = 1.0
    mu_matrix[tier3_ids, :] = 0.0
    n_t2 = len(tier2_ids)
    base_probs = np.array([0.3584, 0.3510, 0.3908, 0.3841, 0.3753, 0.3598, 0.4197])
    rank = 2
    U = np.random.normal(0, 1, size=(n_t2, rank))
    V = np.random.normal(0, 1, size=(K, rank))
    raw = U @ V.T
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    norm = sigmoid(raw)
    scale = base_probs / norm.mean(axis=0)
    mu_t2 = norm * scale[np.newaxis, :]
    mu_t2 += np.random.normal(0, 0.01, size=mu_t2.shape)
    mu_t2 = np.clip(mu_t2, 0.01, 0.99)
    mu_matrix[tier2_ids, :] = mu_t2
    return mu_matrix

def matrix_completion_softimpute(M_obs, rank=2, lambda_=10, max_iters=10):
    """
    A simplified version of the SoftImpute algorithm, which solves
    the nuclear norm minimization problem from the paper.
    """
    M_filled = M_obs.copy()
    nan_mask = np.isnan(M_obs)
    global_mean = np.nanmean(M_obs)
    if np.isnan(global_mean): global_mean = 0.5
    M_filled[nan_mask] = global_mean

    for _ in range(max_iters):
        U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)
        s_thresh = np.maximum(s - lambda_, 0)
        M_reconstructed = U[:, :len(s_thresh)] @ np.diag(s_thresh) @ Vt
        M_filled[nan_mask] = M_reconstructed[nan_mask]
        
    return M_filled
# lambda=10 
def mc_rme(u_target, M_obs, K_groups=5, rank=2, lambda_=10):
    """Algorithm 3: Matrix Completion with Robust Median Estimates. """
    n, K = M_obs.shape
    other_users = [i for i in range(n) if i != u_target]
    user_estimates = []
    min_group_size = K + 1

    for _ in range(K_groups):
        if len(other_users) >= min_group_size - 1:
            random_indices = np.random.choice(other_users, size=min_group_size - 1, replace=False)
            group_indices = np.append(random_indices, u_target)
        else:
            group_indices = np.arange(n)
        
        target_pos_in_group = np.where(group_indices == u_target)[0][0]
        M_group_obs = M_obs[group_indices, :]
        M_group_completed = matrix_completion_softimpute(M_group_obs, rank=rank, lambda_=lambda_)
        user_estimates.append(M_group_completed[target_pos_in_group, :])
    
    return np.median(np.array(user_estimates), axis=0)

class BanditEnvironment:
    def __init__(self, mu_matrix):
        self.mu_matrix, self.n, self.K = mu_matrix, mu_matrix.shape[0], mu_matrix.shape[1]
    def get_reward(self, actions):
        probs = self.mu_matrix[np.arange(self.n), actions]
        return np.random.binomial(1, probs)

# ========== Bandit Policies ==========
class BanditPolicy:
    def __init__(self, **kwargs):
        self.n, self.K, self.name = kwargs.get('n'), kwargs.get('K'), "BanditPolicy"
    def select_arms(self, t, history): raise NotImplementedError
    def update(self, t, actions, rewards, history): pass

class EpsGreedyUCBPolicy(BanditPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.name = f"EpsGreedyUCB"
        self.counts, self.rewards_sum = np.zeros((self.n, self.K)), np.zeros((self.n, self.K))
    def select_arms(self, t, history):
        actions = np.zeros(self.n, dtype=int)
        total_pulls = t + 1
        for i in range(self.n):
            if np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(self.K)
            else:
                if np.sum(self.counts[i]) == 0:
                    actions[i] = np.random.randint(self.K)
                else:
                    means = np.divide(self.rewards_sum[i], self.counts[i], out=np.zeros(self.K), where=self.counts[i] > 0)
                    bonus = np.sqrt((2 * np.log(total_pulls)) / np.maximum(1, self.counts[i]))
                    actions[i] = np.argmax(means + bonus)
        return actions
    def update(self, t, actions, rewards, history):
        for i in range(self.n):
            self.counts[i, actions[i]] += 1
            self.rewards_sum[i, actions[i]] += rewards[i]

class UCBPolicy(BanditPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta', 0.01)
        self.name = "UCB"
        self.counts, self.rewards_sum = np.zeros((self.n, self.K)), np.zeros((self.n, self.K))
    def select_arms(self, t, history):
        if t < self.K: return np.full(self.n, t, dtype=int)
        actions = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            means = np.divide(self.rewards_sum[i], self.counts[i], out=np.zeros(self.K), where=self.counts[i] > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                bonus = np.sqrt((2 * np.log(1 / self.delta)) / self.counts[i])
            bonus[self.counts[i] == 0] = 1e6 
            actions[i] = np.argmax(means + bonus)
        return actions
    def update(self, t, actions, rewards, history):
        for i in range(self.n):
            self.counts[i, actions[i]] += 1
            self.rewards_sum[i, actions[i]] += rewards[i]

class ThompsonSamplingPolicy(BanditPolicy):
    """ Thompson Sampling with Beta-Bernoulli model. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Thompson Sampling"
        self.alpha, self.beta = np.ones((self.n, self.K)), np.ones((self.n, self.K))
    def select_arms(self, t, history):
        return np.argmax(np.random.beta(self.alpha, self.beta), axis=1)
    def update(self, t, actions, rewards, history):
        for i in range(self.n):
            if rewards[i] == 1: self.alpha[i, actions[i]] += 1
            else: self.beta[i, actions[i]] += 1

class GreedyMCPolicy(BanditPolicy):
    """ Algorithm 1: Greedy Matrix Completion Policy. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.T_explore = kwargs.get('T_explore', 10)
        self.lambda_ = kwargs.get('lambda_', 10)
        self.name, self.best_arms = f"Greedy MC ({self.T_explore})", None
        
        self.M_obs = np.full((self.n, self.K), np.nan)
        self.counts = np.zeros((self.n, self.K))

    def select_arms(self, t, history):
        if t < self.T_explore: 
            return np.random.randint(self.K, size=self.n)
        
        if self.best_arms is None:
            print(f"\n{self.name}: End of exploration. Estimating best arms...")
            E_hat = np.zeros((self.n, self.K))
            for i in tqdm(range(self.n), desc=f"Estimating for {self.name}"):
                E_hat[i, :] = mc_rme(i, self.M_obs, lambda_=self.lambda_)
            
            self.best_arms = np.argmax(E_hat, axis=1)
            print(f"{self.name}: Estimation complete. Starting exploitation.")
            
        return self.best_arms

    def update(self, t, actions, rewards, history):
        if t < self.T_explore:
            for i in range(self.n):
                a, r = actions[i], rewards[i]
                
                if self.counts[i, a] == 0:
                    self.M_obs[i, a] = r
                else:
                    learning_rate = 1 / (t + 1)
                    old_val = self.M_obs[i, a]
                    if np.isnan(old_val): old_val = 0 
                    self.M_obs[i, a] = (learning_rate * r) + (1 - learning_rate) * old_val
                
                self.counts[i, a] += 1

class PhasedMCPolicy(BanditPolicy):
    """ Algorithm 2: Phased Matrix Completion """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase_length = kwargs.get('phase_length', 10)
        self.beta_temp = kwargs.get('beta_temp', 1.0)
        self.lambda_ = kwargs.get('lambda_', 10)
        self.name = f"Phased MC ({self.phase_length})"
        self.Q = np.full((self.n, self.K), 1/self.K)

        self.M_obs = np.full((self.n, self.K), np.nan)
        self.counts = np.zeros((self.n, self.K))

    def select_arms(self, t, history):
        actions = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            actions[i] = np.random.choice(self.K, p=self.Q[i])
        return actions
    
    def update(self, t, actions, rewards, history):
        # --- Part 1: Update of M_obs for the current round ---
        for i in range(self.n):
            a, r = actions[i], rewards[i]
            
            learning_rate = 1 / (t + 1)
            
            if self.counts[i, a] == 0:
                self.M_obs[i, a] = r
            else:
                old_val = self.M_obs[i, a]
                if np.isnan(old_val): old_val = 0
                self.M_obs[i, a] = (learning_rate * r) + (1 - learning_rate) * old_val
            
            self.counts[i, a] += 1
        
        # --- Part 2: Conditional re-estimation at the end of a phase ---
        if (t + 1) % self.phase_length == 0 and t > 0:
            print(f"\n{self.name}: End of phase at round {t+1}. Re-estimating policy...")
            
            E_hat = np.zeros((self.n, self.K))
            for i in tqdm(range(self.n), desc=f"Estimating for {self.name} (t={t+1})"):
                E_hat[i, :] = mc_rme(i, self.M_obs, lambda_=self.lambda_)
            
            exp_scores = np.exp(self.beta_temp * E_hat)
            exp_scores_sum = np.sum(exp_scores, axis=1, keepdims=True)
            self.Q = np.divide(exp_scores, exp_scores_sum, 
                               out=np.full_like(exp_scores, 1/self.K), 
                               where=exp_scores_sum > 1e-8)
            print(f"{self.name}: Policy updated.")
# ========== Simulation & Evaluation Functions ==========

def run_simulation(policy, env, T):
    history = []
    for t in tqdm(range(T), desc=f"Running {policy.name}"):
        actions = policy.select_arms(t, history)
        rewards = env.get_reward(actions)
        policy.update(t, actions, rewards, history)
        history.append((t, actions, rewards))
    return history

def calculate_final_regret(history, mu_matrix):
    n = mu_matrix.shape[0]
    optimal_rewards_per_round = np.sum(np.max(mu_matrix, axis=1))
    total_actual_expected_reward = 0
    for t, actions, _ in history:
        total_actual_expected_reward += np.sum(mu_matrix[np.arange(n), actions])
    T = len(history)
    return (optimal_rewards_per_round * T) - total_actual_expected_reward

def calculate_average_position_curve(history, mu_matrix):
    n = mu_matrix.shape[0]
    arm_ranks = np.argsort(np.argsort(-mu_matrix, axis=1), axis=1)
    avg_positions = [np.mean(arm_ranks[np.arange(n), actions]) for t, actions, _ in history]
    return avg_positions

if __name__ == "__main__":
    n, K, T = 5000, 7, 50
    main_seed = 42

    policy_configs = {
        "EpsGreedyUCB": {"class": EpsGreedyUCBPolicy, "params": {"epsilon": 0.1}},
        "UCB": {"class": UCBPolicy, "params": {"delta": 0.01}},
        "Thompson Sampling": {"class": ThompsonSamplingPolicy, "params": {}},
        "Greedy MC (5)": {"class": GreedyMCPolicy, "params": {"T_explore": 5}},
        "Greedy MC (10)": {"class": GreedyMCPolicy, "params": {"T_explore": 10}},
        "Greedy MC (15)": {"class": GreedyMCPolicy, "params": {"T_explore": 15}},
        "Phased MC (2)": {"class": PhasedMCPolicy, "params": {"phase_length": 2}},
        "Phased MC (5)": {"class": PhasedMCPolicy, "params": {"phase_length": 5}},
    }
    
    all_histories = {}
    final_regrets = {}
    all_position_curves = {}

    mu_matrix = generate_kilkari_style_reward_matrix(n, K, seed=main_seed)
    env = BanditEnvironment(mu_matrix)

    for name, config in policy_configs.items():
        policy_params = {'n': n, 'K': K, **config['params']}
        policy = config['class'](**policy_params)
        np.random.seed(main_seed) 
        
        history = run_simulation(policy, env, T)
        all_histories[name] = history
        final_regrets[name] = calculate_final_regret(history, mu_matrix)
        all_position_curves[name] = calculate_average_position_curve(history, mu_matrix)

    output_dir = "simulation_history"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, history in all_histories.items():
        filename = f"{output_dir}/history_{name.replace(' ', '')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = list(final_regrets.keys())
    regrets = list(final_regrets.values())
    
    bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.bar(names, regrets, color=bar_colors)
    
    ax.set_ylabel('Final Regret', fontsize=14)
    ax.set_title(f'Comparison of Final Regret (T={T})', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    
    for name, curve in all_position_curves.items():
        plt.plot(curve, label=name, lw=2, marker='o', markersize=4, alpha=0.8)
        
    plt.title(f'Average Position of Chosen Slot vs. Rounds (T={T})', fontsize=16)
    plt.xlabel('Rounds (t)', fontsize=14)
    plt.ylabel('Average Position (0=Best)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()