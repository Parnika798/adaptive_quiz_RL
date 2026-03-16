"""
Answer Evaluation Module
Supports: MCQ grading, keyword-based short-answer NLP grading,
          numeric answer checking, and fuzzy string matching.
"""

import re
import math
import difflib
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Question:
    question_id: str
    text: str
    question_type: str           # "mcq" | "short_answer" | "numeric" | "fill_blank"
    skill_id: str
    difficulty: str              # "Easy" | "Medium" | "Hard"
    difficulty_b: float          # IRT b-parameter
    # MCQ
    options: Optional[List[str]] = None
    correct_option: Optional[int] = None   # 0-indexed
    # Short answer / fill-in-blank
    correct_keywords: Optional[List[str]] = None
    correct_answer_text: Optional[str] = None
    # Numeric
    correct_numeric: Optional[float] = None
    numeric_tolerance: float = 0.01

    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

class MCQGrader:
    def grade(self, question: Question, student_answer: Union[int, str]) -> bool:
        if isinstance(student_answer, str):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3,
                       "a": 0, "b": 1, "c": 2, "d": 3}
            student_answer = mapping.get(student_answer, -1)
        return int(student_answer) == question.correct_option


class NumericGrader:
    def grade(self, question: Question, student_answer: Union[str, float]) -> bool:
        try:
            val = float(str(student_answer).strip())
            return abs(val - question.correct_numeric) <= question.numeric_tolerance
        except (ValueError, TypeError):
            return False


class KeywordNLPGrader:
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t not in self._stopwords()]

    def _stopwords(self):
        return {"a", "an", "the", "is", "are", "was", "were", "be",
                "been", "being", "have", "has", "had", "do", "does",
                "did", "will", "would", "could", "should", "may",
                "might", "shall", "and", "or", "but", "in", "on",
                "at", "to", "for", "of", "with", "by"}

    def token_f1(self, pred: str, gold: str) -> float:
        pred_toks = set(self._tokenize(pred))
        gold_toks = set(self._tokenize(gold))
        if not pred_toks or not gold_toks:
            return float(pred_toks == gold_toks)
        common = pred_toks & gold_toks
        precision = len(common) / len(pred_toks)
        recall    = len(common) / len(gold_toks)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def fuzzy_similarity(self, pred: str, gold: str) -> float:
        return difflib.SequenceMatcher(None, pred.lower(), gold.lower()).ratio()

    def grade(self, question: Question, student_answer: str,
              threshold: float = 0.5) -> bool:
        if question.correct_keywords:
            answer_lower = student_answer.lower()
            hits = sum(1 for kw in question.correct_keywords if kw.lower() in answer_lower)
            if hits / len(question.correct_keywords) >= threshold:
                return True
        if question.correct_answer_text:
            f1 = self.token_f1(student_answer, question.correct_answer_text)
            if f1 >= threshold:
                return True
            fuzzy = self.fuzzy_similarity(student_answer, question.correct_answer_text)
            return fuzzy >= 0.75
        return False


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class AnswerEvaluator:
    def __init__(self):
        self._mcq     = MCQGrader()
        self._numeric = NumericGrader()
        self._nlp     = KeywordNLPGrader()

    def evaluate(self, question: Question, student_answer) -> bool:
        if question.question_type == "mcq":
            return self._mcq.grade(question, student_answer)
        elif question.question_type == "numeric":
            return self._numeric.grade(question, student_answer)
        elif question.question_type in ("short_answer", "fill_blank"):
            return self._nlp.grade(question, str(student_answer))
        return False

    def compute_response_features(self, question: Question,
                                  student_answer,
                                  response_time_ms: int,
                                  hint_used: bool = False) -> dict:
        correct = self.evaluate(question, student_answer)
        speed_bonus = 0.1 if (correct and response_time_ms < 10000) else 0.0
        hint_penalty = -0.15 if hint_used else 0.0
        return {"correct": correct, "speed_bonus": speed_bonus,
                "hint_penalty": hint_penalty}


# ---------------------------------------------------------------------------
# Question bank — 45 MCQ questions (3 per skill × difficulty)
# 5 skills × 3 difficulties × 3 questions = 45 total
# Enough for 3 non-repeating 15-question sessions
# ---------------------------------------------------------------------------

def build_sample_question_bank() -> List[Question]:
    """
    Returns 45 MCQ questions — 3 per skill × difficulty combination.
    Every question has exactly 4 options and correct_option (0-indexed).
    question_type is always 'mcq'.
    """
    bank = []

    # (skill, difficulty, b, text, [A, B, C, D], correct_idx)
    templates = [

        # ══════════════════════════════════════════════════════════
        # ALGEBRA
        # ══════════════════════════════════════════════════════════

        # Algebra · Easy
        ("algebra", "Easy", -1.2,
         "What is 2x + 3 = 7 solved for x?",
         ["x = 1", "x = 2", "x = 3", "x = 4"], 1),

        ("algebra", "Easy", -1.1,
         "If y = 3x and x = 4, what is y?",
         ["7", "10", "12", "16"], 2),

        ("algebra", "Easy", -1.0,
         "What is the value of 5x when x = 6?",
         ["11", "20", "25", "30"], 3),

        # Algebra · Medium
        ("algebra", "Medium", 0.1,
         "Solve x² − 5x + 6 = 0. What are the roots?",
         ["x = 1, 2", "x = 2, 3", "x = 3, 4", "x = −2, −3"], 1),

        ("algebra", "Medium", 0.2,
         "What is the slope of the line y = 3x − 7?",
         ["−7", "3", "7", "−3"], 1),

        ("algebra", "Medium", 0.3,
         "Simplify: (x² − 9) ÷ (x − 3)",
         ["x − 3", "x + 3", "x² − 3", "x + 9"], 1),

        # Algebra · Hard
        ("algebra", "Hard", 1.3,
         "The discriminant b²−4ac < 0 means the quadratic has:",
         ["Two distinct real roots", "One repeated real root",
          "Two complex non-real roots", "No solution"], 2),

        ("algebra", "Hard", 1.4,
         "If f(x) = x³ − 3x, what is f′(x)?",
         ["3x² − 3", "x² − 3", "3x²", "x³ − 3"], 0),

        ("algebra", "Hard", 1.5,
         "Which value of k makes x² + kx + 9 = 0 have exactly one real root?",
         ["3", "6", "9", "12"], 1),

        # ══════════════════════════════════════════════════════════
        # GEOMETRY
        # ══════════════════════════════════════════════════════════

        # Geometry · Easy
        ("geometry", "Easy", -0.9,
         "What is the area of a rectangle with width 4 m and length 5 m?",
         ["16 m²", "18 m²", "20 m²", "22 m²"], 2),

        ("geometry", "Easy", -0.8,
         "How many degrees are in a straight angle?",
         ["90°", "120°", "180°", "360°"], 2),

        ("geometry", "Easy", -0.7,
         "What is the perimeter of a square with side 7 cm?",
         ["14 cm", "21 cm", "28 cm", "49 cm"], 2),

        # Geometry · Medium
        ("geometry", "Medium", 0.2,
         "A triangle has angles 60° and 70°. What is the third angle?",
         ["40°", "50°", "60°", "70°"], 1),

        ("geometry", "Medium", 0.3,
         "What is the area of a circle with radius 5? (use π ≈ 3.14)",
         ["15.7", "31.4", "78.5", "157.0"], 2),

        ("geometry", "Medium", 0.4,
         "A right triangle has legs 6 and 8. What is the hypotenuse?",
         ["7", "9", "10", "12"], 2),

        # Geometry · Hard
        ("geometry", "Hard", 1.1,
         "A right triangle has legs of length 3 and 4. What is the hypotenuse?",
         ["5", "6", "7", "√7"], 0),

        ("geometry", "Hard", 1.2,
         "What is the volume of a sphere with radius 3? (use π ≈ 3.14)",
         ["28.3", "56.5", "113.1", "339.3"], 2),

        ("geometry", "Hard", 1.3,
         "Two parallel lines are cut by a transversal. "
         "Alternate interior angles are:",
         ["Supplementary", "Complementary", "Equal", "Unrelated"], 2),

        # ══════════════════════════════════════════════════════════
        # STATISTICS
        # ══════════════════════════════════════════════════════════

        # Statistics · Easy
        ("statistics", "Easy", -1.0,
         "What is the mean of [2, 4, 6, 8]?",
         ["4", "5", "6", "7"], 1),

        ("statistics", "Easy", -0.9,
         "What is the median of [3, 5, 7, 9, 11]?",
         ["5", "6", "7", "9"], 2),

        ("statistics", "Easy", -0.8,
         "In the dataset [4, 4, 5, 6, 4], what is the mode?",
         ["4", "5", "6", "4.6"], 0),

        # Statistics · Medium
        ("statistics", "Medium", 0.3,
         "Which measure is most resistant to outliers?",
         ["Mean", "Median", "Mode", "Range"], 1),

        ("statistics", "Medium", 0.4,
         "If the standard deviation of a dataset is 0, what does that mean?",
         ["All values are 0", "All values are equal",
          "The mean is 0", "The dataset is empty"], 1),

        ("statistics", "Medium", 0.5,
         "A dataset has mean 50 and standard deviation 10. "
         "What z-score corresponds to a value of 70?",
         ["1", "1.5", "2", "2.5"], 2),

        # Statistics · Hard
        ("statistics", "Hard", 1.4,
         "In a hypothesis test at α = 0.05, you get p = 0.03. You should:",
         ["Fail to reject H₀", "Reject H₀",
          "Accept H₀", "Increase the sample size"], 1),

        ("statistics", "Hard", 1.5,
         "Which condition is required for the Central Limit Theorem to apply?",
         ["Population must be normal", "Sample size must be large (n ≥ 30)",
          "Data must be symmetric", "Variance must equal mean"], 1),

        ("statistics", "Hard", 1.6,
         "A 95% confidence interval means:",
         ["95% of data falls in this range",
          "There is a 95% chance the parameter is in this interval",
          "95% of such intervals constructed would contain the true parameter",
          "The p-value is 0.05"], 2),

        # ══════════════════════════════════════════════════════════
        # CALCULUS
        # ══════════════════════════════════════════════════════════

        # Calculus · Easy
        ("calculus", "Easy", -0.8,
         "What is the derivative of f(x) = x²?",
         ["x", "2x", "x²", "2x²"], 1),

        ("calculus", "Easy", -0.7,
         "What is the derivative of a constant, e.g. f(x) = 7?",
         ["7", "1", "0", "−7"], 2),

        ("calculus", "Easy", -0.6,
         "What is the derivative of f(x) = 3x?",
         ["3x²", "3", "x", "0"], 1),

        # Calculus · Medium
        ("calculus", "Medium", 0.0,
         "Evaluate the definite integral ∫₀³ 2x dx.",
         ["6", "9", "12", "18"], 1),

        ("calculus", "Medium", 0.1,
         "What is the derivative of f(x) = sin(x)?",
         ["−sin(x)", "cos(x)", "−cos(x)", "tan(x)"], 1),

        ("calculus", "Medium", 0.2,
         "Find the critical point of f(x) = x² − 4x + 3.",
         ["x = 1", "x = 2", "x = 3", "x = 4"], 1),

        # Calculus · Hard
        ("calculus", "Hard", 1.5,
         "The Fundamental Theorem of Calculus states ∫ₐᵇ f(x) dx = ?",
         ["F(a) − F(b)", "F(b) − F(a)", "f(b) − f(a)", "f(a) − f(b)"], 1),

        ("calculus", "Hard", 1.6,
         "What is the second derivative of f(x) = x⁴?",
         ["4x³", "12x²", "24x", "x³"], 1),

        ("calculus", "Hard", 1.7,
         "Using the chain rule, what is d/dx [sin(x²)]?",
         ["cos(x²)", "2x·cos(x²)", "cos(2x)", "2sin(x)·cos(x)"], 1),

        # ══════════════════════════════════════════════════════════
        # ARITHMETIC
        # ══════════════════════════════════════════════════════════

        # Arithmetic · Easy
        ("arithmetic", "Easy", -1.5,
         "What is 144 ÷ 12?",
         ["11", "12", "13", "14"], 1),

        ("arithmetic", "Easy", -1.4,
         "What is 17 × 8?",
         ["126", "130", "136", "144"], 2),

        ("arithmetic", "Easy", -1.3,
         "What is 256 − 89?",
         ["157", "167", "177", "187"], 1),

        # Arithmetic · Medium
        ("arithmetic", "Medium", -0.2,
         "What is 15% of 200?",
         ["25", "30", "35", "40"], 1),

        ("arithmetic", "Medium", -0.1,
         "A jacket costs ₹800 after a 20% discount. What was the original price?",
         ["₹960", "₹1,000", "₹1,050", "₹1,200"], 1),

        ("arithmetic", "Medium", 0.0,
         "If 3 pens cost ₹45, how much do 7 pens cost?",
         ["₹95", "₹100", "₹105", "₹115"], 2),

        # Arithmetic · Hard
        ("arithmetic", "Hard", 0.8,
         "₹1,000 at 10% p.a. compound interest for 2 years. "
         "How much more than simple interest is earned?",
         ["₹0", "₹10", "₹20", "₹100"], 1),

        ("arithmetic", "Hard", 0.9,
         "A train travels 360 km in 4 hours. "
         "How long to travel 270 km at the same speed?",
         ["2.5 hours", "3 hours", "3.5 hours", "4 hours"], 1),

        ("arithmetic", "Hard", 1.0,
         "What is the LCM of 12 and 18?",
         ["6", "24", "36", "72"], 2),
    ]

    for qid, (skill, diff, b, text, opts, correct_idx) in enumerate(templates):
        bank.append(Question(
            question_id=f"q{qid:03d}",
            text=text,
            question_type="mcq",
            skill_id=skill,
            difficulty=diff,
            difficulty_b=b,
            options=opts,
            correct_option=correct_idx,
            correct_keywords=None,
            correct_numeric=None,
            correct_answer_text=None,
        ))

    return bank