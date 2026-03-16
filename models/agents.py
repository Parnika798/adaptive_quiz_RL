"""
RL Agents:
  1. QLearningAgent      — tabular off-policy
  2. SARSAAgent          — tabular on-policy
  3. PolicyIterationAgent — model-based (discretised)
  4. DQNAgent            — neural network (PyTorch optional, numpy fallback)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json


# ---------------------------------------------------------------------------
# Helper: discretise continuous obs → tuple key for tabular methods
# ---------------------------------------------------------------------------

def discretise_obs(obs: np.ndarray, bins: int = 5) -> tuple:
    """Map continuous [0,1] obs to discrete bins for tabular Q-table."""
    return tuple(int(min(x * bins, bins - 1)) for x in obs)


# ---------------------------------------------------------------------------
# 1. Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    name = "Q-Learning"

    def __init__(self, n_actions: int = 3, alpha: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 bins: int = 5, seed: int = 42):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.Q: Dict[tuple, np.ndarray] = {}

    def _get_Q(self, state: tuple) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def act(self, obs: np.ndarray) -> int:
        state = discretise_obs(obs, self.bins)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._get_Q(state)))

    def update(self, obs, action, reward, next_obs, done):
        s  = discretise_obs(obs, self.bins)
        s_ = discretise_obs(next_obs, self.bins)
        Qs  = self._get_Q(s)
        Qs_ = self._get_Q(s_)
        target = reward + (0.0 if done else self.gamma * np.max(Qs_))
        Qs[action] += self.alpha * (target - Qs[action])
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        data = {"Q": {str(k): v.tolist() for k, v in self.Q.items()},
                "epsilon": self.epsilon}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.Q = {eval(k): np.array(v) for k, v in data["Q"].items()}
        self.epsilon = data["epsilon"]


# ---------------------------------------------------------------------------
# 2. SARSA Agent
# ---------------------------------------------------------------------------

class SARSAAgent(QLearningAgent):
    name = "SARSA"

    def update(self, obs, action, reward, next_obs, done, next_action=None):
        s  = discretise_obs(obs, self.bins)
        s_ = discretise_obs(next_obs, self.bins)
        Qs  = self._get_Q(s)
        Qs_ = self._get_Q(s_)
        if next_action is None or done:
            next_q = 0.0
        else:
            next_q = Qs_[next_action]
        target = reward + (0.0 if done else self.gamma * next_q)
        Qs[action] += self.alpha * (target - Qs[action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# 3. Policy Iteration Agent (model-based, discretised MDP)
# ---------------------------------------------------------------------------

class PolicyIterationAgent:
    """
    Model-based agent. Requires a discrete MDP approximation.
    We discretise the skill-level dimension and run exact DP.
    """
    name = "Policy Iteration"

    def __init__(self, n_skill_levels: int = 11, n_questions: int = 20,
                 n_actions: int = 3, gamma: float = 0.9):
        self.n_sl = n_skill_levels
        self.n_q  = n_questions
        self.n_a  = n_actions
        self.gamma = gamma
        n_states = n_skill_levels * n_questions
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
        self._fitted = False

    def _state_idx(self, sl: int, q: int) -> int:
        return sl * self.n_q + q

    def _build_model(self):
        """
        Build approximate transition P(s'|s,a) and reward R(s,a).
        Uses IRT-inspired probabilities:
          P(correct | Easy, sl)   = 0.2 + 0.07 * sl
          P(correct | Medium, sl) = 0.1 + 0.08 * sl
          P(correct | Hard, sl)   = 0.0 + 0.09 * sl
        """
        n_s = self.n_sl * self.n_q
        self.P = np.zeros((n_s, self.n_a, n_s))  # transition
        self.R = np.zeros((n_s, self.n_a))        # expected reward

        p_correct_coeff = [(0.20, 0.07), (0.10, 0.08), (0.00, 0.09)]
        r_correct = [1.0, 2.0, 3.0]

        for sl in range(self.n_sl):
            for q in range(self.n_q):
                s = self._state_idx(sl, q)
                q_next = min(q + 1, self.n_q - 1)
                for a in range(self.n_a):
                    intercept, slope = p_correct_coeff[a]
                    p_c = np.clip(intercept + slope * sl, 0.0, 1.0)
                    # Transition: skill increases by 1 if correct
                    sl_next_c = min(sl + 1, self.n_sl - 1)
                    sl_next_w = max(sl - 1, 0)
                    s_next_c = self._state_idx(sl_next_c, q_next)
                    s_next_w = self._state_idx(sl_next_w, q_next)
                    self.P[s, a, s_next_c] += p_c
                    self.P[s, a, s_next_w] += 1 - p_c
                    self.R[s, a] = p_c * r_correct[a] + (1 - p_c) * (-0.5 * r_correct[a])

    def fit(self, max_iter: int = 200, tol: float = 1e-6):
        self._build_model()
        for _ in range(max_iter):
            # Policy Evaluation
            for __ in range(500):
                V_old = self.V.copy()
                for s in range(len(self.V)):
                    a = self.policy[s]
                    self.V[s] = self.R[s, a] + self.gamma * np.dot(self.P[s, a], self.V)
                if np.max(np.abs(self.V - V_old)) < tol:
                    break
            # Policy Improvement
            policy_stable = True
            for s in range(len(self.V)):
                old_a = self.policy[s]
                q_vals = [self.R[s, a] + self.gamma * np.dot(self.P[s, a], self.V)
                          for a in range(self.n_a)]
                self.policy[s] = int(np.argmax(q_vals))
                if old_a != self.policy[s]:
                    policy_stable = False
            if policy_stable:
                break
        self._fitted = True

    def act(self, obs: np.ndarray) -> int:
        if not self._fitted:
            return int(np.random.randint(self.n_a))
        # Map continuous obs to discrete state
        skill = float(obs[:5].mean())   # mean skill level
        q_num = float(obs[-2])          # question number normalised
        sl = int(min(skill * self.n_sl, self.n_sl - 1))
        q  = int(min(q_num * self.n_q,  self.n_q  - 1))
        s  = self._state_idx(sl, q)
        return int(self.policy[s])

    # SARSA/Q-Learning style interface for compatibility
    def update(self, *args, **kwargs):
        pass  # model-based: no online updates needed


# ---------------------------------------------------------------------------
# 4. DQN Agent (numpy-only linear approximation fallback)
#    Full PyTorch DQN used if available
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random

    class _QNet(nn.Module):
        def __init__(self, obs_dim, n_actions, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        name = "DQN"

        def __init__(self, obs_dim: int, n_actions: int = 3,
                     lr: float = 1e-3, gamma: float = 0.9,
                     epsilon: float = 1.0, epsilon_min: float = 0.05,
                     epsilon_decay: float = 0.995,
                     buffer_size: int = 10_000, batch_size: int = 64,
                     target_update: int = 100, seed: int = 42):
            torch.manual_seed(seed)
            self.n_actions = n_actions
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update = target_update
            self._steps = 0

            self.online  = _QNet(obs_dim, n_actions)
            self.target  = _QNet(obs_dim, n_actions)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.optim   = optim.Adam(self.online.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
            self.buffer  = deque(maxlen=buffer_size)

        def act(self, obs: np.ndarray) -> int:
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0)
                return int(self.online(t).argmax().item())

        def store(self, obs, action, reward, next_obs, done):
            self.buffer.append((obs, action, reward, next_obs, done))

        def update(self, obs, action, reward, next_obs, done):
            self.store(obs, action, reward, next_obs, done)
            if len(self.buffer) < self.batch_size:
                return
            batch = random.sample(self.buffer, self.batch_size)
            obs_b, act_b, rew_b, nobs_b, done_b = map(np.array, zip(*batch))

            obs_t  = torch.FloatTensor(obs_b)
            nobs_t = torch.FloatTensor(nobs_b)
            act_t  = torch.LongTensor(act_b)
            rew_t  = torch.FloatTensor(rew_b)
            done_t = torch.FloatTensor(done_b)

            current_q = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target(nobs_t).max(1)[0]
                target_q = rew_t + self.gamma * next_q * (1 - done_t)

            loss = self.loss_fn(current_q, target_q)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self._steps += 1
            if self._steps % self.target_update == 0:
                self.target.load_state_dict(self.online.state_dict())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def save(self, path: str):
            torch.save(self.online.state_dict(), path)

        def load(self, path: str, obs_dim: int):
            self.online.load_state_dict(torch.load(path, map_location="cpu"))
            self.target.load_state_dict(self.online.state_dict())

    TORCH_AVAILABLE = True

except ImportError:

    class DQNAgent:  # type: ignore
        """Numpy linear-approximation fallback when PyTorch is unavailable."""
        name = "DQN (linear)"

        def __init__(self, obs_dim: int, n_actions: int = 3, lr: float = 0.01,
                     gamma: float = 0.9, epsilon: float = 1.0,
                     epsilon_min: float = 0.05, epsilon_decay: float = 0.995, **kwargs):
            self.obs_dim   = obs_dim
            self.n_actions = n_actions
            self.lr        = lr
            self.gamma     = gamma
            self.epsilon   = epsilon
            self.epsilon_min   = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.W = np.zeros((n_actions, obs_dim))  # linear weights

        def act(self, obs: np.ndarray) -> int:
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            q_vals = self.W @ obs
            return int(np.argmax(q_vals))

        def update(self, obs, action, reward, next_obs, done):
            q_curr = self.W[action] @ obs
            q_next = np.max(self.W @ next_obs) if not done else 0.0
            target = reward + self.gamma * q_next
            self.W[action] += self.lr * (target - q_curr) * obs
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def save(self, path: str):
            np.save(path, self.W)

        def load(self, path: str, **kwargs):
            self.W = np.load(path)

    TORCH_AVAILABLE = False
