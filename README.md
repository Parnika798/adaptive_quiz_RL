# Adaptive Quiz Difficulty using Reinforcement Learning
### Research-Grade Implementation

> From coursework demo → published-quality research project

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit dashboard
streamlit run app.py

# 3. Or run headless experiment
python train.py
```

---

## 📁 Project Structure

```
adaptive_quiz_rl/
├── app.py                        # Streamlit dashboard (5 tabs)
├── train.py                      # Headless training + evaluation pipeline
├── requirements.txt
│
├── data/
│   ├── dataset_loader.py         # ASSISTments loader + synthetic fallback
│   └── raw/                      # Downloaded / generated CSVs
│
├── models/
│   ├── environment.py            # AdaptiveQuizEnv (Gym-compatible)
│   └── agents.py                 # Q-Learning, SARSA, DQN, Policy Iteration
│
├── baselines/
│   └── baseline_agents.py        # Random, Fixed-*, Rule-Based
│
├── utils/
│   ├── irt_calibration.py        # 1PL/2PL/3PL IRT via marginal MLE
│   ├── forgetting_model.py       # Ebbinghaus decay + multi-skill tracker
│   └── answer_evaluator.py       # MCQ + Numeric + NLP Token-F1 grader
│
└── results/                      # Experiment outputs (CSVs)
```

---

## 🔬 Research Enhancements

### 1. Real Student Data (ASSISTments Schema)
- Loads ASSISTments 2009-2010 dataset if available at `data/raw/skill_builder_data.csv`
- Falls back to synthetic data with identical schema (500 students, IRT-generated responses)
- **Train/test split on student populations** — no data leakage

### 2. Item Response Theory (IRT)
- **1PL / 2PL / 3PL** models via Expectation-Maximisation / Marginal MLE
- Estimates per-item: difficulty (b), discrimination (a), guessing (c)
- Per-student latent ability (θ) via MAP estimation
- Item Characteristic Curves (ICC) visualised in dashboard

### 3. NLP Answer Evaluation
- **MCQ**: exact option matching
- **Numeric**: tolerance-based comparison
- **Short answer**: Token F1 (SQuAD metric) + fuzzy string similarity
- Speed bonus (< 10s response) + hint penalty

### 4. Multi-Dimensional Skill Tracking
- 5 skill dimensions: algebra, geometry, statistics, calculus, arithmetic
- Independent mastery levels per skill
- Agent targets weakest skill (zone of proximal development)

### 5. Ebbinghaus Forgetting Model
- Retention: `R(t) = exp(−t/S)` where S = stability (hours)
- Stability increases after each successful recall (spacing effect)
- Configurable session gap to simulate real-world forgetting

### 6. Baseline Agents
| Agent | Strategy |
|---|---|
| Random | Uniform random difficulty |
| Fixed-Easy | Always Easy |
| Fixed-Medium | Always Medium |
| Fixed-Hard | Always Hard |
| Rule-Based | Threshold heuristic on skill level |

### 7. Statistical Testing
- **Mann-Whitney U** (non-parametric, two-sided)
- **Cohen's d** effect size
- Pairwise comparison heatmap
- Results highlighted at p < 0.05

### 8. Zone-of-Proximal-Development Reward
```
R = base_reward × difficulty_multiplier
  + speed_bonus (correct & fast)
  + ZPD_bonus   (right difficulty for skill level)
  - mismatch_penalty (too hard/easy for student)
```

---

## 🖥️ Streamlit Dashboard Tabs

| Tab | Contents |
|---|---|
| **Overview** | Architecture diagram, enhancement summary |
| **Training** | Smoothed learning curves per RL agent |
| **Comparison** | Bar + box plots on test population |
| **IRT Analysis** | ICC curves, difficulty/discrimination scatter |
| **Student Simulator** | Interactive session, spider skill chart |
| **Statistics** | Pairwise tests, Cohen's d heatmap |

---

## ☁️ Streamlit Cloud Deployment

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch, and set `app.py` as the main file
4. Add `requirements.txt` — Streamlit Cloud installs dependencies automatically

For CPU-only deployment (no PyTorch), remove `torch` from requirements.txt — the DQN agent will fall back to a linear approximation automatically.

---

## 📖 References

1. Lord, F.M. (1980). *Applications of Item Response Theory*. ETS.
2. Ebbinghaus, H. (1885). *Über das Gedächtnis*. Duncker & Humblot.
3. Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
4. Corbett & Anderson (1994). Knowledge Tracing. *User Modeling & User-Adapted Interaction*.
5. ASSISTments Dataset: [assistments.org](https://sites.google.com/site/assistmentsdata)
