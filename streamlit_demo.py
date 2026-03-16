"""
Adaptive Quiz RL — Interactive Demo
Connects to train.py pipeline exactly:

  train.py functions used:
    run_episode(env, agent, train=True)
    train_agent(agent, env, episodes=300)        → cols: episode, agent, total_reward, accuracy, skill_gain
    evaluate_agent(agent, env, episodes=100)     → cols: episode, agent, total_reward, accuracy, skill_gain
    statistical_comparison(results_df)           → cols: agent_A, agent_B, p_value, cohen_d

  train.py CSV outputs read here:
    results/training_results.csv
    results/test_results.csv
    results/stats.csv

  load_or_train_agents() mirrors run_full_experiment() exactly:
    - load_dataset(use_real=False)
    - train_test_student_split(df)
    - IRTCalibrator("2PL").fit(train_df_data)
    - train_agent(agent, env, episodes=300)   ← same signature
    - evaluate_agent(agent, env, episodes=100) ← same signature

Pages: Home | Quiz Simulation | Session Results
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Exact same imports as train.py ──────────────────────────────────────────
from models.environment import AdaptiveQuizEnv
from models.agents import QLearningAgent, SARSAAgent, PolicyIterationAgent, DQNAgent
from baselines.baseline_agents import ALL_BASELINES
from data.dataset_loader import load_dataset, train_test_student_split
from utils.irt_calibration import IRTCalibrator
from utils.answer_evaluator import build_sample_question_bank, Question

# ── train.py functions imported directly ────────────────────────────────────
from train import run_episode, train_agent, evaluate_agent, statistical_comparison

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdaptiveQuiz RL",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0e1a !important; color: #e8eaf0 !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0f1221 !important; border-right: 1px solid #1e2340;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.q-card {
    background: linear-gradient(160deg, #111830 0%, #0d1222 100%);
    border: 1px solid #1e2a4a; border-radius: 18px; padding: 28px 32px;
    margin-bottom: 18px;
}
.q-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; }
.q-number { font-size:0.75rem; font-family:'DM Mono',monospace; color:#4a5280;
    letter-spacing:0.1em; text-transform:uppercase; }
.diff-badge { font-size:0.7rem; font-weight:700; padding:4px 12px; border-radius:99px;
    font-family:'DM Mono',monospace; letter-spacing:0.06em; }
.diff-Easy   { background:#0e2e1a; color:#3dd68c; border:1px solid #1d5934; }
.diff-Medium { background:#2c2000; color:#f0b429; border:1px solid #5a4000; }
.diff-Hard   { background:#2e0e14; color:#f25f6a; border:1px solid #5a1a22; }
.q-text { font-size:1.12rem; font-weight:600; color:#d8dff5; line-height:1.65; }

.opt-wrap { display:flex; align-items:center; gap:12px; padding:13px 16px;
    background:#0d1222; border:1px solid #1a2040; border-radius:10px; margin-bottom:8px; }
.opt-wrap.sel { border-color:#4a7fff; background:linear-gradient(90deg,#0d1a3a,#0a1228); }
.opt-letter { width:28px; height:28px; border-radius:8px; background:#151c38;
    color:#4a5a8a; font-size:0.75rem; font-family:'DM Mono',monospace; font-weight:600;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.opt-wrap.sel .opt-letter { background:#1a2f6a; color:#7eb3ff; }
.opt-text { font-size:0.9rem; color:#8a92b8; font-weight:500; }
.opt-wrap.sel .opt-text { color:#7eb3ff; }

.result-ok  { background:linear-gradient(135deg,#0a2a18,#081f12);
    border:1px solid #1a5c30; border-radius:14px; padding:18px 22px; margin:14px 0; }
.result-bad { background:linear-gradient(135deg,#2a0a12,#1f0810);
    border:1px solid #5c1a22; border-radius:14px; padding:18px 22px; margin:14px 0; }
.res-top { display:flex; align-items:center; gap:14px; }
.res-icon { font-size:1.8rem; }
.res-title { font-size:1rem; font-weight:700; }
.result-ok  .res-title { color:#3dd68c; }
.result-bad .res-title { color:#f25f6a; }
.res-sub  { font-size:0.8rem; color:#5a7060; font-family:'DM Mono',monospace; margin-top:3px; }
.result-bad .res-sub { color:#705a5a; }
.res-next { margin-top:10px; padding-top:10px; border-top:1px solid #1e3a28;
    font-size:0.78rem; color:#4a7060; font-family:'DM Mono',monospace; }
.result-bad .res-next { border-color:#3a1e22; color:#705a5a; }

.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }
.metric-box { background:#0d1222; border:1px solid #1a2040; border-radius:12px; padding:14px 16px; }
.metric-val { font-size:1.5rem; font-weight:800; color:#7eb3ff;
    font-family:'DM Mono',monospace; line-height:1; }
.metric-lbl { font-size:0.68rem; color:#3a4468; margin-top:4px;
    text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }

.prog-track { height:4px; background:#141a2e; border-radius:99px; margin:8px 0 18px; overflow:hidden; }
.prog-fill  { height:100%; border-radius:99px;
    background:linear-gradient(90deg,#2a4fff,#7eb3ff);
    transition:width 0.4s cubic-bezier(0.4,0,0.2,1); }

.agent-card { background:#0d1222; border:1px solid #1a2040; border-radius:14px;
    padding:18px 20px; margin-bottom:10px; }
.agent-tag  { display:inline-block; font-size:0.68rem; font-weight:600;
    padding:2px 8px; border-radius:99px; margin-top:4px;
    font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.tag-rl   { background:#1a3a5e; color:#5ba3ff; }
.tag-on   { background:#1a3a2e; color:#4ecb8a; }
.tag-deep { background:#2e1a3a; color:#b87dff; }
.tag-base { background:#2e2a1a; color:#d4a04a; }
.agent-desc { font-size:0.78rem; color:#5a6180; margin-top:6px; line-height:1.5; }

.sum-card { background:linear-gradient(135deg,#111830,#0d1222);
    border:1px solid #1e2a4a; border-radius:16px; padding:24px; text-align:center; }
.sum-val { font-size:2.2rem; font-weight:800; color:#7eb3ff; font-family:'DM Mono',monospace; }
.sum-lbl { font-size:0.75rem; color:#3a4468; text-transform:uppercase;
    letter-spacing:0.1em; font-weight:600; margin-top:6px; }

.diff-pills { display:flex; gap:10px; margin-top:16px; }
.diff-pill  { flex:1; padding:16px 12px; border-radius:12px; text-align:center; }
.pill-E { background:#0a2018; border:1px solid #1a4030; }
.pill-M { background:#201800; border:1px solid #4a3800; }
.pill-H { background:#200810; border:1px solid #4a1820; }
.pill-count { font-size:1.6rem; font-weight:800; font-family:'DM Mono',monospace; }
.pill-E .pill-count { color:#3dd68c; }
.pill-M .pill-count { color:#f0b429; }
.pill-H .pill-count { color:#f25f6a; }
.pill-lbl { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
    font-weight:600; margin-top:4px; }
.pill-E .pill-lbl { color:#2a6040; }
.pill-M .pill-lbl { color:#6a5000; }
.pill-H .pill-lbl { color:#6a2030; }

.section-label { font-size:0.68rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; color:#2a3255; font-family:'DM Mono',monospace; margin-bottom:10px; }

.skill-row { margin-bottom:10px; }
.skill-row-header { display:flex; justify-content:space-between;
    font-size:0.78rem; color:#5a6280; margin-bottom:4px; }
.skill-bar-track { height:5px; background:#141a2e; border-radius:99px; overflow:hidden; }
.skill-bar-fill  { height:100%; border-radius:99px; transition:width 0.6s ease; }

.hist-row { display:flex; align-items:center; justify-content:space-between;
    padding:9px 14px; border-radius:9px; background:#0d1222;
    border:1px solid #141c34; margin-bottom:4px; gap:8px; }

.insight-up   { background:#0a2018; border:1px solid #1a4030; border-radius:12px; padding:16px; text-align:center; }
.insight-hold { background:#201800; border:1px solid #4a3800; border-radius:12px; padding:16px; text-align:center; }
.insight-down { background:#200810; border:1px solid #4a1820; border-radius:12px; padding:16px; text-align:center; }
.insight-val  { font-size:1.6rem; font-weight:800; font-family:'DM Mono',monospace; }
.insight-up   .insight-val { color:#3dd68c; }
.insight-hold .insight-val { color:#f0b429; }
.insight-down .insight-val { color:#f25f6a; }
.insight-lbl  { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-top:4px; }
.insight-up   .insight-lbl { color:#2a6040; }
.insight-hold .insight-lbl { color:#5a4800; }
.insight-down .insight-lbl { color:#6a2030; }

div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1a2fff,#4a7fff) !important;
    color:#fff !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; border:none !important;
    border-radius:10px !important; padding:12px 28px !important;
    font-size:0.88rem !important; letter-spacing:0.04em !important;
    width:100%;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  AGENT METADATA
# ═════════════════════════════════════════════════════════════════

AGENT_META = {
    "Q-Learning":       {"tag": "value-based RL",    "cls": "tag-rl",   "icon": "◈",
                         "desc": "Off-policy TD. Trained by train_agent(agent, env, episodes=300)."},
    "SARSA":            {"tag": "on-policy RL",       "cls": "tag-on",   "icon": "◇",
                         "desc": "On-policy TD. run_episode() detects SARSAAgent automatically."},
    "DQN":              {"tag": "deep RL",            "cls": "tag-deep", "icon": "⬡",
                         "desc": "Neural net Q-function trained via train_agent() loop."},
    "Policy Iteration": {"tag": "model-based RL",     "cls": "tag-rl",   "icon": "◆",
                         "desc": "Solves MDP exactly via Bellman. No train_agent() needed."},
    "Random":           {"tag": "baseline",           "cls": "tag-base", "icon": "○",
                         "desc": "Uniform random difficulty. evaluate_agent() baseline."},
    "Rule-Based":       {"tag": "heuristic baseline", "cls": "tag-base", "icon": "▷",
                         "desc": "Threshold heuristic from ALL_BASELINES in baselines/."},
}

DIFF_ACTIONS = {0: "Easy", 1: "Medium", 2: "Hard"}
DIFF_ORDER   = ["Easy", "Medium", "Hard"]
N_QUESTIONS  = 15


# ═════════════════════════════════════════════════════════════════
#  LOAD OR TRAIN AGENTS
#  Mirrors run_full_experiment() from train.py exactly:
#    - same function calls
#    - same parameter names (episodes=300, episodes=100)
#    - same CSV column structure (episode, agent, total_reward, accuracy, skill_gain)
# ═════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_or_train_agents():
    """
    Replicates run_full_experiment() from train.py.
    Uses train_agent(agent, env, episodes=300) and
    evaluate_agent(agent, env, episodes=100) with exact signatures.
    Results cached in results/ as training_results.csv, test_results.csv, stats.csv.
    """
    # ── Same as run_full_experiment() ──────────────────────────────
    df = load_dataset(use_real=False)
    train_df_data, test_df_data = train_test_student_split(df)

    irt = IRTCalibrator(model="2PL")
    irt.fit(train_df_data)

    qbank     = build_sample_question_bank()
    obs_dim   = AdaptiveQuizEnv(question_bank=qbank).observation_space.shape[0]

    train_env = AdaptiveQuizEnv(question_bank=qbank, irt_calibrator=irt)
    test_env  = AdaptiveQuizEnv(question_bank=qbank, irt_calibrator=irt)

    # ── Same agent list as run_full_experiment() ───────────────────
    agents = [
        QLearningAgent(),
        SARSAAgent(),
        DQNAgent(obs_dim=obs_dim),
    ]

    pi_agent = PolicyIterationAgent()
    pi_agent.fit()                        # model-based — no train_agent() needed

    # ── train_agent(agent, env, episodes=300) — exact signature ───
    train_results = []
    for agent in agents:
        df_train = train_agent(agent, train_env, episodes=300)
        train_results.append(df_train)
    train_df = pd.concat(train_results, ignore_index=True)

    # ── evaluate_agent(agent, env, episodes=100) — exact signature ─
    eval_agents  = agents + [pi_agent] + ALL_BASELINES
    test_results = []
    for agent in eval_agents:
        df_eval = evaluate_agent(agent, test_env, episodes=100)
        test_results.append(df_eval)
    test_df = pd.concat(test_results, ignore_index=True)

    # ── statistical_comparison(results_df) — exact signature ───────
    stat_df = statistical_comparison(test_df)

    # ── Save with exact filenames from train.py ────────────────────
    train_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)
    test_df.to_csv(RESULTS_DIR  / "test_results.csv",     index=False)
    stat_df.to_csv(RESULTS_DIR  / "stats.csv",            index=False)

    # ── Freeze epsilon for inference ───────────────────────────────
    for a in agents:
        if hasattr(a, "epsilon"):
            a.epsilon = a.epsilon_min

    # Build name → agent dict (including ALL_BASELINES by .name attr)
    agent_dict = {a.name: a for a in agents + [pi_agent] + ALL_BASELINES}

    return agent_dict, irt, qbank, train_df, test_df, stat_df


# ═════════════════════════════════════════════════════════════════
#  HUMAN QUIZ SESSION
#  Uses run_episode() logic but intercepts the student response:
#  the human answers instead of the IRT simulator.
#  env internals (skill_tracker, _get_obs, _compute_reward) are
#  updated manually, consistent with how run_episode() drives them.
# ═════════════════════════════════════════════════════════════════

class HumanQuizSession:
    """
    Wraps AdaptiveQuizEnv for human play.
    Agent: the real trained agent from load_or_train_agents().
    After each human answer:
      1. env skill state updated (mirrors env.step internals)
      2. agent.update(obs, action, reward, new_obs, done) called
         — or SARSA variant: agent.update(..., next_action)
      3. agent.act(new_obs) returns next difficulty
    Columns logged: episode=step, agent, total_reward, accuracy, skill_gain
    — same schema as train.py's train_agent / evaluate_agent.
    """

    def __init__(self, agent, qbank, irt, n_questions=N_QUESTIONS):
        self.agent        = agent
        self.n_questions  = n_questions
        self.env          = AdaptiveQuizEnv(
            question_bank=qbank, irt_calibrator=irt,
            session_gap_hours=0.0, seed=int(time.time()) % 10000,
        )
        self.obs, _       = self.env.reset()
        self.done         = False
        self.history      = []      # per-step dicts matching train.py column schema
        self.total_reward = 0.0

        # agent.act(obs) picks first action — same as run_episode()
        self.current_action     = self.agent.act(self.obs)
        self.current_difficulty = DIFF_ACTIONS[self.current_action]
        self._current_q         = self._pick_question(self.current_difficulty)

    def _pick_question(self, difficulty: str) -> Question:
        used = {h["question_id"] for h in self.history}
        pool = [q for q in self.env.question_bank
                if q.difficulty == difficulty and q.question_id not in used]
        if not pool:
            pool = [q for q in self.env.question_bank if q.question_id not in used]
        if not pool:
            pool = self.env.question_bank
        return pool[int(self.env._rng.integers(len(pool)))]

    @property
    def current_question(self) -> Question:
        return self._current_q

    @property
    def step_number(self) -> int:
        return len(self.history)

    def submit_answer(self, chosen_option_idx: int) -> dict:
        """
        Human submits answer.
        Updates env state the same way env.step() does internally,
        then calls agent.update() — mirrors run_episode() update block.
        """
        q       = self._current_q
        correct = (chosen_option_idx == q.correct_option)
        diff    = self.current_difficulty
        action  = self.current_action
        skill   = q.skill_id
        rt_ms   = 8000 if correct else 15000

        # ── Mirror env.step() internals ────────────────────────────
        self.env._sim_time += 300
        self.env.skill_tracker.update(skill, correct, diff, self.env._sim_time)
        self.env._recent_correct.append(int(correct))
        if len(self.env._recent_correct) > 5:
            self.env._recent_correct.pop(0)
        self.env._step += 1

        reward = self.env._compute_reward(correct, action, rt_ms, skill)
        self.total_reward += reward
        new_obs = self.env._get_obs()

        n_done  = self.step_number + 1
        is_done = (n_done >= self.n_questions)

        # ── Mirror run_episode() update block ──────────────────────
        # run_episode() uses isinstance(agent, SARSAAgent) inline
        sarsa = isinstance(self.agent, SARSAAgent)
        if sarsa:
            next_action = self.agent.act(new_obs)
            self.agent.update(self.obs, action, reward, new_obs, is_done, next_action)
        else:
            self.agent.update(self.obs, action, reward, new_obs, is_done)
            next_action = self.agent.act(new_obs)

        # ── Log — same columns as train_agent / evaluate_agent ─────
        # train.py cols: episode, agent, total_reward, accuracy, skill_gain
        n_corr_so_far = sum(1 for h in self.history if h["correct"]) + int(correct)
        acc_so_far    = n_corr_so_far / n_done
        skill_gain    = self.env.skill_tracker.overall_level()  # proxy for cumulative gain

        entry = {
            # train.py compatible columns
            "episode":      n_done,
            "agent":        self.agent.name,
            "total_reward": self.total_reward,
            "accuracy":     acc_so_far,
            "skill_gain":   skill_gain,
            # extra UI columns (not in train.py, used by display only)
            "q_num":        n_done,
            "question_id":  q.question_id,
            "skill":        skill,
            "difficulty":   diff,
            "action":       action,
            "correct":      correct,
            "reward":       reward,
            "next_diff":    DIFF_ACTIONS[next_action],
            "obs_acc":      float(self.obs[-1]),
        }
        self.history.append(entry)

        self.obs                = new_obs
        self.done               = is_done
        self.current_action     = next_action
        self.current_difficulty = DIFF_ACTIONS[next_action]
        if not is_done:
            self._current_q = self._pick_question(self.current_difficulty)

        return entry

    def skill_summary(self) -> list:
        return [
            {"skill": sk, "level": round(st.level, 3),
             "mastery": st.mastery_category,
             "correct": st.correct_count, "n": st.practice_count}
            for sk, st in self.env.skill_tracker.skills.items()
        ]

    def session_df(self) -> pd.DataFrame:
        """Returns history as DataFrame — same columns as evaluate_agent()."""
        return pd.DataFrame(self.history)


# ═════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════════

def init_state():
    for k, v in {
        "page": "Home", "agent_name": None, "session": None,
        "submitted": False, "last_result": None, "sel_opt": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:6px 0 20px;">
      <div style="font-size:1.3rem;font-weight:800;color:#c8d4f5;letter-spacing:-0.02em;">
        AdaptiveQuiz <span style="color:#4a7fff;">RL</span></div>
      <div style="font-size:0.7rem;color:#2a3255;font-family:'DM Mono',monospace;
                  margin-top:3px;letter-spacing:0.06em;">POWERED BY train.py</div>
    </div>""", unsafe_allow_html=True)

    for icon, pg in [("🏠","Home"), ("🎮","Quiz Simulation"), ("📋","Session Results")]:
        if st.button(f"{icon}  {pg}", key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    sess: Optional[HumanQuizSession] = st.session_state.session
    if sess and sess.history:
        st.divider()
        n_done = len(sess.history)
        n_corr = sum(1 for h in sess.history if h["correct"])
        acc    = n_corr / n_done * 100
        diff   = sess.current_difficulty
        diff_c = "#3dd68c" if diff=="Easy" else "#f0b429" if diff=="Medium" else "#f25f6a"
        st.markdown(f"""
        <div style="font-size:0.68rem;color:#2a3255;font-family:'DM Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
          LIVE · {st.session_state.agent_name}</div>
        <div style="display:flex;flex-direction:column;gap:7px;">
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#5a6280;">
            <span>Questions</span>
            <span style="color:#7eb3ff;font-family:'DM Mono',monospace;">
              {n_done}/{N_QUESTIONS}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#5a6280;">
            <span>Accuracy</span>
            <span style="color:#3dd68c;font-family:'DM Mono',monospace;">{acc:.0f}%</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#5a6280;">
            <span>Reward</span>
            <span style="color:#f0b429;font-family:'DM Mono',monospace;">
              {sess.total_reward:+.1f}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#5a6280;">
            <span>Next diff.</span>
            <span style="font-family:'DM Mono',monospace;color:{diff_c};">{diff}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#5a6280;">
            <span>Skill</span>
            <span style="color:#c8d4f5;font-family:'DM Mono',monospace;">
              {sess.env.skill_tracker.overall_level():.3f}</span></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset", use_container_width=True):
        for k in ["agent_name","session","submitted","last_result","sel_opt"]:
            st.session_state[k] = None if k != "submitted" else False
        st.session_state.page = "Home"
        init_state()
        st.rerun()


# ═════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═════════════════════════════════════════════════════════════════

if st.session_state.page == "Home":

    st.markdown("""
    <div style="max-width:720px;padding-bottom:8px;">
      <div style="font-size:0.72rem;font-family:'DM Mono',monospace;color:#2a3a6a;
                  letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px;">
        Reinforcement Learning · Adaptive Education</div>
      <div style="font-size:2.5rem;font-weight:800;color:#c8d4f5;
                  letter-spacing:-0.03em;line-height:1.2;margin-bottom:14px;">
        Quiz difficulty that<br><span style="color:#4a7fff;">learns you.</span></div>
      <div style="font-size:0.9rem;color:#4a5480;line-height:1.75;max-width:580px;
                  margin-bottom:28px;">
        Agents trained by <code>train.py</code> — using
        <code>train_agent(agent, env, episodes=300)</code> and
        <code>evaluate_agent(agent, env, episodes=100)</code> —
        observe your answers in real time and call
        <code>agent.act(obs)</code> to select the next difficulty.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3], [
        ("📄", "train.py pipeline",
         "train_agent() · evaluate_agent() · statistical_comparison() all wired in."),
        ("🎯", "IRT-Calibrated",
         "IRTCalibrator('2PL').fit(train_df_data) — same call as run_full_experiment()."),
        ("📊", "CSV-compatible",
         "Session logs use same columns: episode, agent, total_reward, accuracy, skill_gain."),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#0d1222;border:1px solid #1a2040;border-radius:14px;
                        padding:20px;height:165px;">
              <div style="font-size:1.4rem;margin-bottom:10px;">{icon}</div>
              <div style="font-size:0.88rem;font-weight:700;color:#c0c8ec;margin-bottom:6px;">
                {title}</div>
              <div style="font-size:0.77rem;color:#3a4468;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">train.py → streamlit_demo.py connection</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1222;border:1px solid #1a2040;border-radius:14px;
                padding:20px 24px;font-family:'DM Mono',monospace;font-size:0.78rem;
                color:#4a5a8a;line-height:2.2;">
      <span style="color:#3dd68c;">from train import</span>
        run_episode, train_agent, evaluate_agent, statistical_comparison<br>
      <br>
      <span style="color:#f0b429;">train_agent</span>(agent, env,
        <span style="color:#7eb3ff;">episodes=300</span>)
        → cols: episode, agent, total_reward, accuracy, skill_gain<br>
      <span style="color:#f0b429;">evaluate_agent</span>(agent, env,
        <span style="color:#7eb3ff;">episodes=100</span>)
        → same columns<br>
      <span style="color:#f0b429;">statistical_comparison</span>(results_df)
        → cols: agent_A, agent_B, p_value, cohen_d<br>
      <br>
      CSVs saved: <span style="color:#c8d4f5;">training_results.csv · test_results.csv · stats.csv</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("Running train.py pipeline (first run ~20 s)…"):
        load_or_train_agents()
    st.success("✓ Agents trained — ready to play")

    if st.button("Start Quiz →"):
        st.session_state.page = "Quiz Simulation"
        st.rerun()


# ═════════════════════════════════════════════════════════════════
#  PAGE: QUIZ SIMULATION
# ═════════════════════════════════════════════════════════════════

elif st.session_state.page == "Quiz Simulation":

    agent_dict, irt, qbank, train_df, test_df, stat_df = load_or_train_agents()

    # ── Agent selection ──────────────────────────────────────────
    if st.session_state.agent_name is None:
        st.markdown('<div style="font-size:2rem;font-weight:800;color:#c8d4f5;'
                    'letter-spacing:-0.02em;margin-bottom:6px;">Choose your agent</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.85rem;color:#3a4468;margin-bottom:24px;">'
                    'All RL agents trained via <code>train_agent(agent, env, episodes=300)</code>. '
                    'Baselines evaluated via <code>evaluate_agent(agent, env, episodes=100)</code>. '
                    'Select one — its frozen policy will call <code>agent.act(obs)</code> '
                    'after every answer.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        for i, (name, meta) in enumerate(AGENT_META.items()):
            with (col1 if i % 2 == 0 else col2):
                if name not in agent_dict:
                    continue
                if st.button(f"{meta['icon']}  {name}",
                             key=f"pick_{name}", use_container_width=True):
                    st.session_state.agent_name = name
                    st.session_state.session = HumanQuizSession(
                        agent=agent_dict[name], qbank=qbank, irt=irt)
                    st.session_state.submitted   = False
                    st.session_state.last_result = None
                    st.session_state.sel_opt     = None
                    st.rerun()
                st.markdown(f"""
                <div class="agent-card">
                  <span class="agent-tag {meta['cls']}">{meta['tag']}</span>
                  <div class="agent-desc">{meta['desc']}</div>
                </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Active quiz ──────────────────────────────────────────────
    sess: HumanQuizSession = st.session_state.session

    if sess.done:
        st.markdown("""
        <div style="text-align:center;padding:40px;">
          <div style="font-size:2.5rem;margin-bottom:14px;">🎉</div>
          <div style="font-size:1.4rem;font-weight:800;color:#c8d4f5;">Session complete!</div>
          <div style="font-size:0.85rem;color:#3a4468;margin-top:8px;">
            Results logged in same schema as evaluate_agent().</div>
        </div>""", unsafe_allow_html=True)
        if st.button("View Results →"):
            st.session_state.page = "Session Results"
            st.rerun()
        st.stop()

    n_done = sess.step_number
    pct    = int(n_done / N_QUESTIONS * 100)
    q      = sess.current_question
    diff   = sess.current_difficulty

    left, right = st.columns([3, 1], gap="large")

    with left:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;
                    color:#2a3255;font-family:'DM Mono',monospace;margin-bottom:4px;">
          <span>QUESTION {n_done+1} OF {N_QUESTIONS}
            · agent.act(obs) → {diff}</span>
          <span>{pct}% COMPLETE</span></div>
        <div class="prog-track">
          <div class="prog-fill" style="width:{pct}%;"></div></div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="q-card">
          <div class="q-header">
            <span class="q-number">Q{n_done+1} · {q.skill_id}</span>
            <span class="diff-badge diff-{diff}">{diff}
              &nbsp;·&nbsp; IRT b = {q.difficulty_b:.2f}</span>
          </div>
          <div class="q-text">{q.text}</div>
        </div>""", unsafe_allow_html=True)

        # ── Answer phase ─────────────────────────────────────────
        if not st.session_state.submitted:
            letters = ["A","B","C","D"]
            for i, opt in enumerate(q.options or []):
                sel = "sel" if st.session_state.sel_opt == i else ""
                col_btn, col_label = st.columns([1, 11])
                with col_btn:
                    if st.button(letters[i], key=f"opt_{n_done}_{i}"):
                        st.session_state.sel_opt = i
                        st.rerun()
                with col_label:
                    st.markdown(f"""
                    <div class="opt-wrap {sel}" style="margin-top:2px;">
                      <span class="opt-letter">{letters[i]}</span>
                      <span class="opt-text">{opt}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Submit Answer",
                         disabled=(st.session_state.sel_opt is None)):
                result = sess.submit_answer(st.session_state.sel_opt)
                st.session_state.last_result = result
                st.session_state.submitted   = True
                st.session_state.sel_opt     = None
                st.rerun()

        # ── Result phase ─────────────────────────────────────────
        else:
            res     = st.session_state.last_result
            correct = res["correct"]
            reward  = res["reward"]
            next_d  = res["next_diff"]
            obs_acc = res["obs_acc"]

            if correct:
                st.markdown(f"""
                <div class="result-ok">
                  <div class="res-top">
                    <div class="res-icon">✅</div>
                    <div>
                      <div class="res-title">Correct!</div>
                      <div class="res-sub">
                        Reward: <strong>+{reward:.2f}</strong>
                        &nbsp;·&nbsp; Skill tracked: {res['skill']}
                        &nbsp;·&nbsp; Difficulty: {res['difficulty']}
                      </div>
                      <div class="res-next">
                        {st.session_state.agent_name}.act(obs)
                        &nbsp; obs[-1] (recent_acc) = {obs_acc:.2f}
                        &nbsp;→&nbsp; next difficulty:
                        <strong>{next_d}</strong>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                correct_text = (q.options[q.correct_option] if q.options else "—")
                st.markdown(f"""
                <div class="result-bad">
                  <div class="res-top">
                    <div class="res-icon">❌</div>
                    <div>
                      <div class="res-title">Incorrect</div>
                      <div class="res-sub">
                        Correct: <strong>{correct_text}</strong>
                        &nbsp;·&nbsp; Reward: <strong>{reward:.2f}</strong>
                      </div>
                      <div class="res-next">
                        {st.session_state.agent_name}.act(obs)
                        &nbsp; obs[-1] (recent_acc) = {obs_acc:.2f}
                        &nbsp;→&nbsp; next difficulty:
                        <strong>{next_d}</strong>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            if st.button("Next Question →"):
                st.session_state.submitted   = False
                st.session_state.last_result = None
                st.session_state.sel_opt     = None
                st.rerun()

    # ── Right panel: live metrics ────────────────────────────────
    with right:
        n_hist = len(sess.history)
        n_corr = sum(1 for h in sess.history if h["correct"])
        acc    = (n_corr / n_hist * 100) if n_hist else 50.0
        diff_c = "#3dd68c" if diff=="Easy" else "#f0b429" if diff=="Medium" else "#f25f6a"

        st.markdown('<div class="section-label" style="margin-top:36px;">LIVE METRICS</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-box">
            <div class="metric-val">{acc:.0f}%</div>
            <div class="metric-lbl">Accuracy</div></div>
          <div class="metric-box">
            <div class="metric-val">{sess.total_reward:+.1f}</div>
            <div class="metric-lbl">Reward</div></div>
          <div class="metric-box">
            <div class="metric-val">{n_hist}</div>
            <div class="metric-lbl">Done</div></div>
          <div class="metric-box">
            <div class="metric-val" style="font-size:0.9rem;color:{diff_c};">
              {diff[:3].upper()}</div>
            <div class="metric-lbl">Difficulty</div></div>
        </div>
        <div class="metric-box" style="margin-bottom:10px;">
          <div class="metric-val" style="font-size:1.1rem;">
            {sess.env.skill_tracker.overall_level():.3f}</div>
          <div class="metric-lbl">Overall Skill (env)</div></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:14px;">SKILL STATE</div>',
                    unsafe_allow_html=True)
        for row in sess.skill_summary():
            pct_sk = row["level"] * 100
            bar_c  = "#3dd68c" if pct_sk>=60 else "#f0b429" if pct_sk>=30 else "#f25f6a"
            st.markdown(f"""
            <div class="skill-row">
              <div class="skill-row-header">
                <span>{row['skill']}</span>
                <span style="font-family:'DM Mono',monospace;">{pct_sk:.0f}%</span>
              </div>
              <div class="skill-bar-track">
                <div class="skill-bar-fill"
                     style="width:{pct_sk}%;background:{bar_c};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        if sess.history:
            st.markdown('<div class="section-label" style="margin-top:14px;">HISTORY</div>',
                        unsafe_allow_html=True)
            for h in reversed(sess.history[-6:]):
                tc = "#3dd68c" if h["correct"] else "#f25f6a"
                dc = "#3dd68c" if h["difficulty"]=="Easy" else \
                     "#f0b429" if h["difficulty"]=="Medium" else "#f25f6a"
                tick = "✓" if h["correct"] else "✗"
                st.markdown(f"""
                <div class="hist-row">
                  <span style="font-size:0.7rem;color:#3a4468;
                               font-family:'DM Mono',monospace;">Q{h['q_num']}</span>
                  <span style="font-size:0.7rem;color:#4a5480;">{h['skill'][:4]}</span>
                  <span style="font-size:0.68rem;color:{dc};
                               font-family:'DM Mono',monospace;">{h['difficulty'][:3]}</span>
                  <span style="font-size:0.85rem;color:{tc};">{tick}</span>
                  <span style="font-size:0.7rem;color:#f0b429;
                               font-family:'DM Mono',monospace;">{h['reward']:+.1f}</span>
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  PAGE: SESSION RESULTS
# ═════════════════════════════════════════════════════════════════

elif st.session_state.page == "Session Results":

    sess: Optional[HumanQuizSession] = st.session_state.session
    if not sess or not sess.history:
        st.markdown("""
        <div style="text-align:center;padding:60px;">
          <div style="font-size:2rem;margin-bottom:12px;">📋</div>
          <div style="font-size:1.1rem;font-weight:700;color:#c8d4f5;">No session yet</div>
          <div style="font-size:0.82rem;color:#3a4468;margin-top:8px;">
            Complete a quiz to see your results.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    hist    = sess.history
    n_q     = len(hist)
    n_corr  = sum(1 for h in hist if h["correct"])
    acc     = n_corr / n_q * 100
    reward  = sess.total_reward
    agent   = st.session_state.agent_name or "—"
    easy_n  = sum(1 for h in hist if h["difficulty"]=="Easy")
    med_n   = sum(1 for h in hist if h["difficulty"]=="Medium")
    hard_n  = sum(1 for h in hist if h["difficulty"]=="Hard")
    streak  = 0
    for h in reversed(hist):
        if h["correct"]: streak += 1
        else: break

    st.markdown(f"""
    <div style="font-size:0.72rem;font-family:'DM Mono',monospace;color:#2a3a6a;
                letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px;">
      Session Complete · {agent} · {n_q} Questions</div>
    <div style="font-size:2rem;font-weight:800;color:#c8d4f5;
                letter-spacing:-0.02em;margin-bottom:4px;">Quiz Summary</div>
    <div style="font-size:0.85rem;color:#3a4468;margin-bottom:24px;">
      Logged in same schema as <code>evaluate_agent()</code>:
      episode, agent, total_reward, accuracy, skill_gain.</div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="sum-card"><div class="sum-val">{n_q}</div>'
                    '<div class="sum-lbl">Questions Attempted</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sum-card"><div class="sum-val">{acc:.0f}%</div>'
                    '<div class="sum-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        rc = "#3dd68c" if reward>=0 else "#f25f6a"
        st.markdown(f'<div class="sum-card"><div class="sum-val" style="color:{rc};">'
                    f'{reward:+.1f}</div><div class="sum-lbl">Total Reward</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="sum-card"><div class="sum-val">{streak}</div>'
                    '<div class="sum-lbl">Final Streak</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1,1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">Difficulty Distribution</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="diff-pills">
          <div class="diff-pill pill-E">
            <div class="pill-count">{easy_n}</div>
            <div class="pill-lbl">Easy</div></div>
          <div class="diff-pill pill-M">
            <div class="pill-count">{med_n}</div>
            <div class="pill-lbl">Medium</div></div>
          <div class="diff-pill pill-H">
            <div class="pill-count">{hard_n}</div>
            <div class="pill-lbl">Hard</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:24px;">Final Skill Profile</div>',
                    unsafe_allow_html=True)
        for row in sess.skill_summary():
            pct_sk = row["level"] * 100
            bar_c  = "#3dd68c" if pct_sk>=60 else "#f0b429" if pct_sk>=30 else "#f25f6a"
            st.markdown(f"""
            <div class="skill-row">
              <div class="skill-row-header">
                <span>{row['skill']}
                  <span style="font-size:0.65rem;color:#2a3255;margin-left:6px;">
                    ({row['correct']}/{row['n']})</span></span>
                <span style="font-family:'DM Mono',monospace;">{pct_sk:.1f}%</span>
              </div>
              <div class="skill-bar-track">
                <div class="skill-bar-fill"
                     style="width:{pct_sk}%;background:{bar_c};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:24px;">Agent Behaviour</div>',
                    unsafe_allow_html=True)
        upgrades   = sum(1 for i in range(1,len(hist))
                         if DIFF_ORDER.index(hist[i]["difficulty"]) >
                            DIFF_ORDER.index(hist[i-1]["difficulty"]))
        downgrades = sum(1 for i in range(1,len(hist))
                         if DIFF_ORDER.index(hist[i]["difficulty"]) <
                            DIFF_ORDER.index(hist[i-1]["difficulty"]))
        holds      = max(0, len(hist)-1-upgrades-downgrades)
        ca,cb,cc   = st.columns(3)
        with ca:
            st.markdown(f'<div class="insight-up"><div class="insight-val">{upgrades}</div>'
                        '<div class="insight-lbl">Difficulty ↑</div></div>',
                        unsafe_allow_html=True)
        with cb:
            st.markdown(f'<div class="insight-hold"><div class="insight-val">{holds}</div>'
                        '<div class="insight-lbl">Held Same</div></div>',
                        unsafe_allow_html=True)
        with cc:
            st.markdown(f'<div class="insight-down"><div class="insight-val">{downgrades}</div>'
                        '<div class="insight-lbl">Difficulty ↓</div></div>',
                        unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-label">Response Log</div>', unsafe_allow_html=True)
        for h in hist:
            tick = "✓" if h["correct"] else "✗"
            tc   = "#3dd68c" if h["correct"] else "#f25f6a"
            dc   = "#3dd68c" if h["difficulty"]=="Easy" else \
                   "#f0b429" if h["difficulty"]=="Medium" else "#f25f6a"
            rc   = "#f0b429" if h["reward"]>=0 else "#f25f6a"
            st.markdown(f"""
            <div class="hist-row">
              <span style="font-size:0.7rem;color:#3a4468;
                           font-family:'DM Mono',monospace;width:26px;">Q{h['q_num']}</span>
              <span style="font-size:0.72rem;color:#4a5480;flex:1;">{h['skill']}</span>
              <span style="font-size:0.68rem;color:{dc};
                           font-family:'DM Mono',monospace;width:52px;">{h['difficulty']}</span>
              <span style="font-size:0.85rem;color:{tc};width:18px;">{tick}</span>
              <span style="font-size:0.7rem;color:{rc};
                           font-family:'DM Mono',monospace;width:34px;text-align:right;">
                {h['reward']:+.1f}</span>
              <span style="font-size:0.65rem;color:#2a3255;font-family:'DM Mono',monospace;">
                →{h['next_diff'][:1]}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Start New Session"):
        for k in ["agent_name","session","submitted","last_result","sel_opt"]:
            st.session_state[k] = None if k != "submitted" else False
        st.session_state.page = "Home"
        init_state()
        st.rerun()
