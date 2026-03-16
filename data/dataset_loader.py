"""
Dataset Loader — ASSISTments 2009 / EdNet / Synthetic fallback
Loads real student response data and prepares it for RL training.
"""

import numpy as np
import pandas as pd
import os
import requests
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic ASSISTments-style data (used when real download is unavailable)
# ---------------------------------------------------------------------------

def generate_synthetic_assistments(n_students=500, n_questions=50, seed=42):
    """
    Generate synthetic student response data following ASSISTments schema.
    Columns: student_id, question_id, skill_id, correct, response_time_ms,
             attempt_count, hint_count, difficulty_irt (b-param)
    """
    rng = np.random.default_rng(seed)
    SKILLS = ["algebra", "geometry", "statistics", "calculus", "arithmetic"]
    records = []

    for sid in range(n_students):
        # Each student has a latent ability θ ~ N(0,1)
        theta = rng.normal(0, 1)
        n_responses = rng.integers(10, n_questions)

        for _ in range(n_responses):
            qid = rng.integers(0, n_questions)
            skill = SKILLS[qid % len(SKILLS)]
            # IRT difficulty b ~ N(0,1), discrimination a ~ Uniform(0.5,2)
            b = rng.normal(0, 1)
            a = rng.uniform(0.5, 2.0)
            # 3-parameter logistic: P(correct) = c + (1-c) / (1 + exp(-a(θ-b)))
            c = 0.25  # guessing param
            p = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
            correct = int(rng.random() < p)
            response_time = rng.integers(5000, 120000)  # ms
            attempts = rng.integers(1, 4)
            hints = rng.integers(0, attempts)

            records.append({
                "student_id": sid,
                "question_id": qid,
                "skill_id": skill,
                "correct": correct,
                "response_time_ms": response_time,
                "attempt_count": attempts,
                "hint_count": hints,
                "difficulty_irt": b,
                "discrimination_irt": a,
                "theta_true": theta,   # latent ability (ground truth for eval)
            })

    df = pd.DataFrame(records)
    path = DATA_DIR / "synthetic_assistments.csv"
    df.to_csv(path, index=False)
    print(f"[DataLoader] Synthetic dataset saved → {path}  ({len(df)} rows)")
    return df


def download_assistments(dest: Path) -> bool:
    """
    Attempt to download ASSISTments 2009-2010 skill-builder dataset.
    Primary:  direct CSV from CAIDS / Harvard Dataverse mirror
    Fallback: second known mirror
    Returns True if successful, False if network unavailable.
    """
    # Known public mirrors for ASSISTments 2009-10 skill builder data
    URLS = [
        # Harvard Dataverse (official host)
        "https://dataverse.harvard.edu/api/access/datafile/2365462",
        # Direct CSV mirror maintained by several ML-edu repos
        "https://raw.githubusercontent.com/arghosh/AKT/master/data/assist2009_pid/assist2009_pid.csv",
    ]
    for url in URLS:
        try:
            print(f"[DataLoader] Downloading ASSISTments data from:\n  {url}")
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"[DataLoader] Download complete → {dest} ({dest.stat().st_size:,} bytes)")
            return True
        except Exception as e:
            print(f"[DataLoader] Mirror failed ({e}), trying next...")
    return False


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from any known ASSISTments schema variant."""
    rename_maps = [
        # 2009-10 skill-builder schema
        {"user_id":           "student_id",
         "problem_id":        "question_id",
         "skill_name":        "skill_id",
         "ms_first_response": "response_time_ms",
         "attempt_count":     "attempt_count",
         "hint_count":        "hint_count"},
        # AKT / PID variant
        {"user_id":           "student_id",
         "item_id":           "question_id",
         "skill_id":          "skill_id",
         "elapsed_time":      "response_time_ms"},
        # Generic fallback
        {"userId":            "student_id",
         "problemId":         "question_id",
         "skill":             "skill_id"},
    ]
    for rmap in rename_maps:
        df = df.rename(columns={k: v for k, v in rmap.items() if k in df.columns})

    # Ensure required columns exist
    for col in ("student_id", "question_id", "correct"):
        if col not in df.columns:
            raise ValueError(
                f"[DataLoader] Required column '{col}' not found after renaming.\n"
                f"  Available columns: {list(df.columns)}\n"
                f"  Place the ASSISTments CSV manually at:\n"
                f"  {DATA_DIR / 'skill_builder_data.csv'}"
            )

    # Map skill_id to one of the 5 canonical skills if not already done
    SKILL_MAP = {
        "algebra":     "algebra",    "Algebra":     "algebra",
        "geometry":    "geometry",   "Geometry":    "geometry",
        "statistics":  "statistics", "Statistics":  "statistics",
        "calculus":    "calculus",   "Calculus":    "calculus",
        "arithmetic":  "arithmetic", "Arithmetic":  "arithmetic",
    }
    SKILLS = ["algebra", "geometry", "statistics", "calculus", "arithmetic"]
    if "skill_id" in df.columns:
        # Map known names; assign remaining by hash mod 5
        def _map_skill(s):
            s = str(s)
            if s in SKILL_MAP:
                return SKILL_MAP[s]
            return SKILLS[hash(s) % 5]
        df["skill_id"] = df["skill_id"].apply(_map_skill)

    df = df.dropna(subset=["student_id", "question_id", "correct"])
    df["correct"] = df["correct"].astype(int)
    return df


def load_dataset(use_real=False):
    """
    Load student response data.

    use_real=False  → always use synthetic ASSISTments-style data
    use_real=True   → load real ASSISTments 2009-2010 data:
                        1. Check data/raw/skill_builder_data.csv
                        2. If missing, attempt auto-download from public mirrors
                        3. If download fails, raise a clear FileNotFoundError
                           (never silently fall back to synthetic)

    Returns: DataFrame with columns:
        student_id, question_id, skill_id, correct,
        response_time_ms (optional), attempt_count (optional)
    """
    synth_path = DATA_DIR / "synthetic_assistments.csv"
    real_path  = DATA_DIR / "skill_builder_data.csv"

    # ── use_real=True path ────────────────────────────────────────
    if use_real:
        # 1. Already on disk?
        if real_path.exists():
            print(f"[DataLoader] Loading real ASSISTments dataset from {real_path} ...")
        else:
            # 2. Try to download
            print("[DataLoader] Real dataset not found locally. Attempting download...")
            ok = download_assistments(real_path)
            if not ok:
                raise FileNotFoundError(
                    "\n\n[DataLoader] ERROR: use_real=True but the ASSISTments dataset\n"
                    "could not be downloaded (no network access or mirrors unavailable).\n\n"
                    "To fix this, manually download the file from:\n"
                    "  https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data\n\n"
                    f"Then place it at:\n  {real_path}\n\n"
                    "Alternatively, set use_real=False to use synthetic data."
                )

        df = pd.read_csv(real_path, encoding="latin-1", low_memory=False)
        df = _standardise_columns(df)
        print(f"[DataLoader] Real dataset loaded: {len(df):,} responses, "
              f"{df['student_id'].nunique()} students, "
              f"{df['question_id'].nunique()} questions.")
        return df

    # ── use_real=False path ───────────────────────────────────────
    if synth_path.exists():
        print("[DataLoader] Loading cached synthetic dataset...")
        return pd.read_csv(synth_path)

    print("[DataLoader] Generating synthetic dataset...")
    return generate_synthetic_assistments()


def train_test_student_split(df, test_size=0.2, seed=42):
    """
    Split on student_id — no data leakage between train/test populations.
    Returns: train_df, test_df
    """
    students = df["student_id"].unique()
    train_ids, test_ids = train_test_split(students, test_size=test_size,
                                           random_state=seed)
    train_df = df[df["student_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["student_id"].isin(test_ids)].reset_index(drop=True)
    print(f"[DataLoader] Train students: {len(train_ids)} | Test students: {len(test_ids)}")
    return train_df, test_df