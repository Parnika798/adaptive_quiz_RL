"""
Baseline Agents for comparison:
  1. RandomAgent          — uniformly random difficulty
  2. FixedEasyAgent       — always Easy
  3. FixedMediumAgent     — always Medium
  4. FixedHardAgent       — always Hard
  5. RuleBasedAgent       — threshold-based heuristic (mimics human teacher)
"""

import numpy as np
from typing import Optional


class RandomAgent:
    name = "Random"

    def __init__(self, n_actions: int = 3, seed: Optional[int] = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, *args, **kwargs):
        pass


class FixedDifficultyAgent:
    def __init__(self, difficulty: int = 1):
        assert difficulty in (0, 1, 2)
        self.difficulty = difficulty
        self.name = ["Fixed-Easy", "Fixed-Medium", "Fixed-Hard"][difficulty]

    def act(self, obs: np.ndarray) -> int:
        return self.difficulty

    def update(self, *args, **kwargs):
        pass


class RuleBasedAgent:
    """
    Threshold-based heuristic mimicking a human teacher:
    - Overall skill < 0.33  → Easy
    - 0.33 ≤ skill < 0.66  → Medium
    - skill ≥ 0.66          → Hard
    Uses recent accuracy (last element of obs) to fine-tune.
    """
    name = "Rule-Based"

    def act(self, obs: np.ndarray) -> int:
        # obs layout: [skill_1,...,skill_n, q_norm, recent_acc]
        mean_skill = float(np.mean(obs[:-2]))
        recent_acc = float(obs[-1])

        if mean_skill < 0.33:
            base = 0  # Easy
        elif mean_skill < 0.66:
            base = 1  # Medium
        else:
            base = 2  # Hard

        # Adjust: if recent accuracy too high, increase; if too low, decrease
        if recent_acc > 0.85 and base < 2:
            base += 1
        elif recent_acc < 0.40 and base > 0:
            base -= 1

        return int(base)

    def update(self, *args, **kwargs):
        pass


# Convenience list of all baselines
ALL_BASELINES = [
    RandomAgent(),
    FixedDifficultyAgent(0),
    FixedDifficultyAgent(1),
    FixedDifficultyAgent(2),
    RuleBasedAgent(),
]
