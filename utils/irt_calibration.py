"""
Item Response Theory (IRT) — 1PL / 2PL / 3PL Calibration
Estimates student ability (θ) and item parameters (a, b, c) via
Expectation-Maximisation (EM) or closed-form approximation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # σ(x) = 1/(1+e^{-x})
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IRTItem:
    question_id: int | str
    skill_id: str
    difficulty: float        # b parameter
    discrimination: float    # a parameter  (1.0 for 1PL)
    guessing: float          # c parameter  (0.0 for 1PL/2PL)
    se_difficulty: float = 0.0   # standard error


@dataclass
class StudentAbility:
    student_id: int | str
    theta: float             # latent ability estimate
    se: float = 0.0          # standard error
    n_responses: int = 0


# ---------------------------------------------------------------------------
# 3-parameter logistic model
# ---------------------------------------------------------------------------

def p_correct_3pl(theta: float, a: float, b: float, c: float) -> float:
    """P(correct | θ, a, b, c) under the 3PL IRT model."""
    return c + (1.0 - c) * expit(a * (theta - b))


def p_correct_batch(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c + (1.0 - c) * expit(a * (theta - b))


# ---------------------------------------------------------------------------
# Joint MLE calibration (simplified — suitable for demo & moderate datasets)
# ---------------------------------------------------------------------------

class IRTCalibrator:
    """
    Calibrates IRT parameters from a response matrix.
    For large datasets, use the full EM algorithm (mirt / ltm in R).
    Here we use per-item MLE with a fixed population prior θ ~ N(0,1).
    """

    def __init__(self, model: str = "2PL", n_theta_points: int = 61):
        assert model in ("1PL", "2PL", "3PL")
        self.model = model
        # Gauss-Hermite quadrature points for marginal likelihood
        self.theta_pts, self.theta_wts = np.polynomial.hermite.hermgauss(n_theta_points)
        self.theta_pts = self.theta_pts * np.sqrt(2)   # rescale to N(0,1)
        self.theta_wts = self.theta_wts / np.sqrt(np.pi)

        self.items_: Dict[str, IRTItem] = {}
        self.abilities_: Dict[str, StudentAbility] = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "IRTCalibrator":
        """
        df must have columns: student_id, question_id, skill_id, correct
        """
        print(f"[IRT] Calibrating {self.model} model on "
              f"{df['question_id'].nunique()} items × "
              f"{df['student_id'].nunique()} students ...")

        # Step 1: Estimate item parameters per question
        for qid, grp in df.groupby("question_id"):
            skill = grp["skill_id"].iloc[0]
            responses = grp["correct"].values.astype(float)
            params = self._calibrate_item(responses)
            self.items_[qid] = IRTItem(
                question_id=qid, skill_id=str(skill),
                difficulty=params[1],
                discrimination=params[0],
                guessing=params[2] if self.model == "3PL" else 0.0,
            )

        # Step 2: Estimate student abilities via MAP
        for sid, grp in df.groupby("student_id"):
            q_params = [
                self.items_[qid] for qid in grp["question_id"]
                if qid in self.items_
            ]
            responses = grp.loc[grp["question_id"].isin(self.items_), "correct"].values
            if len(responses) == 0:
                theta = 0.0
            else:
                theta = self._estimate_ability(q_params, responses.astype(float))
            self.abilities_[sid] = StudentAbility(
                student_id=sid, theta=theta, n_responses=len(responses)
            )

        print(f"[IRT] Done. Mean difficulty b̄={np.mean([i.difficulty for i in self.items_.values()]):.3f}")
        return self

    # ------------------------------------------------------------------
    def _calibrate_item(self, responses: np.ndarray) -> Tuple[float, float, float]:
        """Returns (a, b, c) for a single item."""
        n = len(responses)
        p_obs = responses.mean()

        if self.model == "1PL":
            # b = logit(p_obs) approximation
            p_obs = np.clip(p_obs, 0.01, 0.99)
            b = -np.log(p_obs / (1 - p_obs))
            return 1.0, b, 0.0

        # Marginal MLE over θ ~ N(0,1) quadrature
        def neg_log_lik(params):
            if self.model == "2PL":
                a, b = params; c = 0.0
            else:
                a, b, c = params
                c = expit(c)  # constrain c ∈ (0,1)
            a = np.exp(a)    # constrain a > 0

            # Marginal P(correct) = ∫ P(correct|θ) N(θ;0,1) dθ
            p_theta = p_correct_batch(self.theta_pts, a, b, c)
            p_marginal = float(np.dot(self.theta_wts, p_theta))
            p_marginal = np.clip(p_marginal, 1e-9, 1 - 1e-9)
            lik = p_obs * np.log(p_marginal) + (1 - p_obs) * np.log(1 - p_marginal)
            return -lik * n

        if self.model == "2PL":
            x0 = [0.0, 0.0]
            bounds = [(-2, 3), (-4, 4)]
        else:
            x0 = [0.0, 0.0, -1.1]
            bounds = [(-2, 3), (-4, 4), (-4, 1)]

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
        if self.model == "2PL":
            a = np.exp(res.x[0]); b = res.x[1]; c = 0.0
        else:
            a = np.exp(res.x[0]); b = res.x[1]; c = expit(res.x[2])
        return a, b, c

    def _estimate_ability(self, items: list, responses: np.ndarray) -> float:
        """MAP estimate of θ given item responses. Prior: N(0,1)."""
        if len(items) == 0:
            return 0.0

        def neg_map(theta_arr):
            theta = theta_arr[0]
            log_lik = 0.0
            for item, r in zip(items, responses):
                p = p_correct_3pl(theta, item.discrimination, item.difficulty, item.guessing)
                p = np.clip(p, 1e-9, 1 - 1e-9)
                log_lik += r * np.log(p) + (1 - r) * np.log(1 - p)
            log_prior = -0.5 * theta ** 2  # N(0,1) prior
            return -(log_lik + log_prior)

        res = minimize(neg_map, [0.0], method="L-BFGS-B", bounds=[(-4, 4)])
        return float(res.x[0])

    # ------------------------------------------------------------------
    def difficulty_band(self, qid) -> str:
        """Map IRT b-parameter to Easy/Medium/Hard."""
        if qid not in self.items_:
            return "Medium"
        b = self.items_[qid].difficulty
        if b < -0.5:
            return "Easy"
        elif b < 0.5:
            return "Medium"
        else:
            return "Hard"

    def get_item_df(self) -> pd.DataFrame:
        rows = [vars(i) for i in self.items_.values()]
        return pd.DataFrame(rows)

    def get_ability_df(self) -> pd.DataFrame:
        rows = [vars(a) for a in self.abilities_.values()]
        return pd.DataFrame(rows)
