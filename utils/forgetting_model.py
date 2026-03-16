"""
Skill Forgetting & Decay Model
Implements Ebbinghaus exponential decay + spaced repetition weighting.
Also handles multi-dimensional skill tracking per topic/concept.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


SKILLS = ["algebra", "geometry", "statistics", "calculus", "arithmetic"]


# ---------------------------------------------------------------------------
# Ebbinghaus Forgetting Curve
# ---------------------------------------------------------------------------

def retention(delta_t_hours: float, stability: float = 24.0) -> float:
    """
    R(t) = exp(−t / S)  where S = stability (hours until ~37% retention).
    Default S=24h → after 24h student retains ~37% of gained skill.
    """
    return float(np.exp(-delta_t_hours / stability))


# ---------------------------------------------------------------------------
# Per-skill state with decay
# ---------------------------------------------------------------------------

@dataclass
class SkillState:
    skill_id: str
    level: float = 0.0            # current mastery ∈ [0,1]
    stability: float = 24.0       # Ebbinghaus stability (hours)
    last_practiced: float = field(default_factory=time.time)
    practice_count: int = 0
    correct_count: int = 0

    def decay(self, current_time: Optional[float] = None):
        """Apply forgetting since last practice."""
        if current_time is None:
            current_time = time.time()
        delta_t = (current_time - self.last_practiced) / 3600.0  # → hours
        r = retention(delta_t, self.stability)
        self.level = self.level * r

    def update(self, correct: bool, difficulty: str,
               current_time: Optional[float] = None):
        """Update skill after answering a question."""
        if current_time is None:
            current_time = time.time()
        self.decay(current_time)

        difficulty_gain = {"Easy": 0.03, "Medium": 0.07, "Hard": 0.12}
        gain = difficulty_gain.get(difficulty, 0.05)
        if correct:
            # Correct: increase level + increase stability (spacing effect)
            self.level = min(1.0, self.level + gain)
            self.stability *= 1.2   # memory consolidation
            self.correct_count += 1
        else:
            # Wrong: small decrease, stability unaffected
            self.level = max(0.0, self.level - gain * 0.3)

        self.last_practiced = current_time
        self.practice_count += 1

    @property
    def mastery_category(self) -> str:
        if self.level < 0.33:
            return "Novice"
        elif self.level < 0.66:
            return "Developing"
        else:
            return "Proficient"


# ---------------------------------------------------------------------------
# Multi-dimensional skill tracker
# ---------------------------------------------------------------------------

class MultiSkillTracker:
    """
    Tracks a student's mastery across multiple skills/topics with
    Ebbinghaus forgetting applied between practice sessions.
    """

    def __init__(self, skills: List[str] = None, default_stability: float = 24.0):
        skills = skills or SKILLS
        self.skills: Dict[str, SkillState] = {
            s: SkillState(skill_id=s, stability=default_stability)
            for s in skills
        }

    def update(self, skill_id: str, correct: bool, difficulty: str,
               current_time: Optional[float] = None):
        if skill_id not in self.skills:
            self.skills[skill_id] = SkillState(skill_id=skill_id)
        self.skills[skill_id].update(correct, difficulty, current_time)

    def apply_all_decay(self, current_time: Optional[float] = None):
        for s in self.skills.values():
            s.decay(current_time)

    def get_vector(self) -> np.ndarray:
        """Return skill levels as a numpy array (for RL state)."""
        return np.array([self.skills[s].level for s in sorted(self.skills)])

    def overall_level(self) -> float:
        return float(np.mean([s.level for s in self.skills.values()]))

    def weakest_skill(self) -> str:
        return min(self.skills, key=lambda s: self.skills[s].level)

    def strongest_skill(self) -> str:
        return max(self.skills, key=lambda s: self.skills[s].level)

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for s in self.skills.values():
            rows.append({
                "skill": s.skill_id,
                "level": round(s.level, 4),
                "stability_h": round(s.stability, 1),
                "practice_count": s.practice_count,
                "correct_count": s.correct_count,
                "mastery": s.mastery_category,
            })
        return pd.DataFrame(rows)

    def reset(self):
        for s in self.skills.values():
            s.level = 0.0
            s.practice_count = 0
            s.correct_count = 0
            s.stability = 24.0
            s.last_practiced = time.time()
