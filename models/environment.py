"""
AdaptiveQuizEnv — OpenAI Gym-compatible RL environment
Incorporates: IRT difficulty calibration, multi-skill tracking,
Ebbinghaus forgetting, NLP answer evaluation, real student simulation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Any, Tuple
import time

from utils.forgetting_model import MultiSkillTracker, SKILLS
from utils.answer_evaluator import AnswerEvaluator, Question, build_sample_question_bank
from utils.irt_calibration import IRTCalibrator, p_correct_3pl


# ---------------------------------------------------------------------------
# Student simulator backed by IRT
# ---------------------------------------------------------------------------

class IRTStudentSimulator:
    """
    Simulates a student using 3PL IRT.
    theta is drawn from N(mu, sigma) at reset; updated after each response.
    """

    def __init__(self, theta_mu: float = 0.0, theta_sigma: float = 1.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.theta_mu = theta_mu
        self.theta_sigma = theta_sigma
        self.theta: float = 0.0

    def reset(self, theta: Optional[float] = None):
        if theta is not None:
            self.theta = theta
        else:
            self.theta = float(self.rng.normal(self.theta_mu, self.theta_sigma))
        return self.theta

    def respond(self, item: Question) -> Tuple[bool, int]:
        """
        Returns (correct: bool, response_time_ms: int).
        Uses 3PL: P(correct) = c + (1-c) σ(a(θ−b))
        """
        a = 1.0       # default discrimination (can be set from IRT calibration)
        b = item.difficulty_b
        c = 0.25 if item.question_type == "mcq" else 0.0
        p = p_correct_3pl(self.theta, a, b, c)
        correct = bool(self.rng.random() < p)

        # Simulate response time: faster if easier relative to θ
        gap = self.theta - b
        mean_rt = max(3000, int(20000 / (1 + np.exp(gap))))
        rt = int(self.rng.normal(mean_rt, mean_rt * 0.3))
        rt = max(1000, rt)

        # Learning: correct response slightly increases theta
        if correct:
            self.theta = min(4.0, self.theta + 0.05)
        return correct, rt


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AdaptiveQuizEnv(gym.Env):
    """
    Observation space:
        [skill_levels × n_skills,  question_number_norm,  recent_accuracy_5]
        = n_skills + 2 dimensions

    Action space: {0=Easy, 1=Medium, 2=Hard}

    Reward:
        Correct + speed_bonus + hint_penalty − difficulty_mismatch_penalty
    """

    metadata = {"render_modes": ["human"]}
    ACTIONS = {0: "Easy", 1: "Medium", 2: "Hard"}
    N_QUESTIONS = 20

    def __init__(
        self,
        question_bank: Optional[List[Question]] = None,
        skills: Optional[List[str]] = None,
        irt_calibrator: Optional[IRTCalibrator] = None,
        session_gap_hours: float = 0.0,   # simulated time gap for decay
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.skills = skills or SKILLS
        self.n_skills = len(self.skills)
        self.question_bank = question_bank or build_sample_question_bank()
        self.irt = irt_calibrator
        self.session_gap_hours = session_gap_hours
        self._rng = np.random.default_rng(seed)

        # Build question bank index by (skill, difficulty)
        self._qbank_index: Dict[Tuple[str, str], List[Question]] = {}
        for q in self.question_bank:
            key = (q.skill_id, q.difficulty)
            self._qbank_index.setdefault(key, []).append(q)

        # Spaces
        n_obs = self.n_skills + 2
        self.observation_space = spaces.Box(
            low=np.zeros(n_obs, dtype=np.float32),
            high=np.ones(n_obs, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Components
        self.skill_tracker = MultiSkillTracker(skills=self.skills)
        self.evaluator = AnswerEvaluator()
        self.student = IRTStudentSimulator(seed=seed)

        # Episode state
        self._step = 0
        self._recent_correct: List[int] = []
        self._current_skill_idx = 0
        self._history: List[Dict] = []
        self._sim_time = time.time()

    # -----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.skill_tracker.reset()
        self._step = 0
        self._recent_correct = []
        self._history = []
        self._current_skill_idx = 0

        # Simulate session gap (forgetting between sessions)
        if self.session_gap_hours > 0:
            gap_time = self._sim_time - self.session_gap_hours * 3600
            self.skill_tracker.apply_all_decay(gap_time)

        # Sample student ability from distribution
        self.student.reset()

        return self._get_obs(), {}

    # -----------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        difficulty = self.ACTIONS[action]

        # Pick skill to target (round-robin, weighted toward weakest)
        skill = self._select_skill()

        # Sample question
        question = self._sample_question(skill, difficulty)

        # Student responds (IRT-based simulation)
        correct, rt_ms = self.student.respond(question)

        # Update multi-skill tracker with simulated time
        self._sim_time += 300  # 5 min per question
        self.skill_tracker.update(skill, correct, difficulty, self._sim_time)

        # Compute reward
        reward = self._compute_reward(correct, action, rt_ms, skill)

        # Track history
        self._recent_correct.append(int(correct))
        if len(self._recent_correct) > 5:
            self._recent_correct.pop(0)

        self._history.append({
            "step": self._step,
            "skill": skill,
            "difficulty": difficulty,
            "difficulty_action": action,
            "correct": correct,
            "reward": reward,
            "response_time_ms": rt_ms,
            "theta": self.student.theta,
            "overall_skill": self.skill_tracker.overall_level(),
        })

        self._step += 1
        terminated = self._step >= self.N_QUESTIONS
        obs = self._get_obs()
        info = {"correct": correct, "skill": skill, "difficulty": difficulty}
        return obs, reward, terminated, False, info

    # -----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        skill_vec = self.skill_tracker.get_vector()  # [0,1]^n_skills
        q_norm = self._step / self.N_QUESTIONS
        acc5 = np.mean(self._recent_correct) if self._recent_correct else 0.5
        obs = np.concatenate([skill_vec, [q_norm, acc5]]).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _select_skill(self) -> str:
        """Prefer weakest skill, with some exploration."""
        if self._rng.random() < 0.3:
            return self.skills[self._rng.integers(self.n_skills)]
        return self.skill_tracker.weakest_skill()

    def _sample_question(self, skill: str, difficulty: str) -> Question:
        key = (skill, difficulty)
        pool = self._qbank_index.get(key)
        if not pool:
            # Fallback: any question for this skill
            pool = [q for q in self.question_bank if q.skill_id == skill]
        if not pool:
            pool = self.question_bank  # global fallback
        idx = int(self._rng.integers(len(pool)))
        return pool[idx]

    def _compute_reward(self, correct: bool, action: int,
                        rt_ms: int, skill: str) -> float:
        difficulty = self.ACTIONS[action]
        skill_level = self.skill_tracker.skills[skill].level

        # Base reward
        difficulty_multiplier = {0: 1.0, 1: 2.0, 2: 3.0}[action]
        base = difficulty_multiplier if correct else (-0.5 * difficulty_multiplier)

        # Speed bonus
        speed_bonus = 0.2 if (correct and rt_ms < 10_000) else 0.0

        # Zone-of-proximal-development bonus: challenge should slightly exceed skill
        zpd_bonus = 0.0
        if correct and difficulty == "Medium" and 0.3 < skill_level < 0.7:
            zpd_bonus = 0.5
        elif correct and difficulty == "Hard" and skill_level > 0.6:
            zpd_bonus = 0.8
        elif correct and difficulty == "Easy" and skill_level < 0.3:
            zpd_bonus = 0.3

        # Mismatch penalty: hard question for novice or easy for expert
        mismatch_penalty = 0.0
        if difficulty == "Hard" and skill_level < 0.2:
            mismatch_penalty = -1.0
        elif difficulty == "Easy" and skill_level > 0.8:
            mismatch_penalty = -0.5

        return float(base + speed_bonus + zpd_bonus + mismatch_penalty)

    # -----------------------------------------------------------------------
    def get_episode_summary(self) -> Dict[str, Any]:
        if not self._history:
            return {}
        df_hist = __import__("pandas").DataFrame(self._history)
        return {
            "total_reward": df_hist["reward"].sum(),
            "accuracy": df_hist["correct"].mean(),
            "final_skill": df_hist["overall_skill"].iloc[-1],
            "skill_gain": df_hist["overall_skill"].iloc[-1] - df_hist["overall_skill"].iloc[0],
            "difficulty_distribution": df_hist["difficulty"].value_counts().to_dict(),
            "history": df_hist,
        }

    def render(self, mode="human"):
        if self._history:
            h = self._history[-1]
            print(f"Step {h['step']:2d} | {h['skill']:10s} | {h['difficulty']:6s} | "
                  f"{'✓' if h['correct'] else '✗'} | r={h['reward']:+.2f} | "
                  f"skill={h['overall_skill']:.3f}")
