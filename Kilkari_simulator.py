import numpy as np

# ========== Reward Matrix Generation ==========
# Generate n*k reward matrix with specific tier structures
# n = number of users, K = number of arms (time periods)
def generate_kilkari_style_reward_matrix(n, K=7, seed=42):
    """
    Simulates Kilkari-style reward matrix:
    - Tier 1: always picks up (prob = 1.0)
    - Tier 3: never picks up (prob = 0.0)
    - Tier 2: low-rank structure with empirical base probabilities
    """
    np.random.seed(seed)
    mu_matrix = np.zeros((n, K))

    # Tier definitions
    tier1_frac, tier3_frac = 0.4059, 0.0699
    tier1_cutoff = int(tier1_frac * n)
    tier3_cutoff = int(tier3_frac * n)

    perm = np.random.permutation(n)
    tier1_ids = perm[:tier1_cutoff]
    tier3_ids = perm[-tier3_cutoff:]
    tier2_ids = perm[tier1_cutoff:-tier3_cutoff]

    # Tier 1: always pick up
    mu_matrix[tier1_ids, :] = 1.0

    # Tier 3: never pick up
    mu_matrix[tier3_ids, :] = 0.0

    # Tier 2: low-rank structure
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
    return mu_matrix, tier1_ids, tier2_ids, tier3_ids

# ========== Bandit Environment ==========

class BanditEnvironment:
    def __init__(self, mu_matrix):
        self.mu_matrix = mu_matrix
        self.n, self.K = mu_matrix.shape

    def get_reward(self, actions):
        probs = self.mu_matrix[np.arange(self.n), actions]
        return np.random.binomial(1, probs)

# ========== Bandit Policies ==========

class BanditPolicy:
    def __init__(self, n, K):
        self.n = n
        self.K = K

    def select_arms(self, t, history):
        raise NotImplementedError

    def update(self, actions, rewards, t, history):
        pass

    def init_from_history(self, history):
        pass

class RandomPolicy(BanditPolicy):
    def select_arms(self, t, history):
        return np.random.randint(self.K, size=self.n)

    def update(self, actions, rewards, t, history):
        pass

class EpsGreedyUCBPolicy(BanditPolicy):
    def __init__(self, n, K, epsilon=0.1):
        super().__init__(n, K)
        self.epsilon = epsilon
        self.counts = np.zeros((n, K))
        self.rewards = np.zeros((n, K))
        self.t = 0

    def select_arms(self, t, history):
        actions = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            if np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(self.K)
            else:
                means = np.divide(self.rewards[i], self.counts[i],
                                  out=np.zeros(self.K), where=self.counts[i] > 0)
                bonus = np.sqrt((2 * np.log(max(1, self.t + 1))) / np.maximum(1, self.counts[i]))
                ucb = means + bonus
                actions[i] = np.argmax(ucb)
        return actions

    def update(self, actions, rewards, t, history):
        for i in range(self.n):
            a = actions[i]
            r = rewards[i]
            self.counts[i, a] += 1
            self.rewards[i, a] += r
        self.t += 1

    def init_from_history(self, history):
        for t, actions, rewards in history:
            self.update(actions, rewards, t, history)

# ========== Simulation Runner ==========

def run_policy(policy, rounds, env, history):
    for t in rounds:
        actions = policy.select_arms(t, history)
        rewards = env.get_reward(actions)
        policy.update(actions, rewards, t, history)
        history.append((t, actions.copy(), rewards.copy()))

# ========== Run Simulation ==========

if __name__ == "__main__":
    np.random.seed(42)
    n, K, T = 5000, 7, 5

    mu_matrix, t1, t2, t3 = generate_kilkari_style_reward_matrix(n, K)
    env = BanditEnvironment(mu_matrix)
    history = []

    # Phase 1: Random policy
    random_policy = RandomPolicy(n, K)
    run_policy(random_policy, range(0, T // 2), env, history)

    # Phase 2: Eps-Greedy UCB policy (warm-started)
    ucb_policy = EpsGreedyUCBPolicy(n, K, epsilon=0.1)
    ucb_policy.init_from_history(history)
    run_policy(ucb_policy, range(T // 2, T), env, history)

    # Example: Print stats for last 2 rounds
    print("Last 2 rounds of reward rates:")
    for t, actions, rewards in history[-2:]:
        print(f"Round {t+1}: Mean reward = {rewards.mean():.4f}")
    print(history)  # Print last 2 rounds of history
    print(mu_matrix)
    print()