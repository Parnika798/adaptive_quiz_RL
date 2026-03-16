"""
Adaptive Quiz RL — Interactive Demo
Fully integrated with the real training pipeline:
  - Uses AdaptiveQuizEnv (IRT, multi-skill tracker, forgetting model)
  - Loads real trained agents (Q-Learning / SARSA / DQN / Policy Iteration)
    OR trains them on-the-fly if saved_models/ is empty
  - Human answers questions; env updates obs + skill state each step
  - Agent's .act(obs) picks the NEXT difficulty every round

Pages: Home | Quiz Simulation | Session Results
"""

import streamlit as st
import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.environment import AdaptiveQuizEnv, SKILLS
from models.agents import QLearningAgent, SARSAAgent, PolicyIterationAgent, DQNAgent
from baselines.baseline_agents import RandomAgent, RuleBasedAgent, ALL_BASELINES 
from utils.answer_evaluator import build_sample_question_bank, Question
from data.dataset_loader import load_dataset, train_test_student_split
from utils.irt_calibration import IRTCalibrator

MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

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
    background: #242f69 !important; color: #e8eaf0 !important;
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
    background:#0d1222; border:1px solid #1a2040; border-radius:10px;
    margin-bottom:8px; }
.opt-wrap.sel { border-color:#4a7fff; background:linear-gradient(90deg,#0d1a3a,#0a1228); }
.opt-letter { width:28px; height:28px; border-radius:8px; background:#151c38;
    color:#4a5a8a; font-size:0.75rem; font-family:'DM Mono',monospace; font-weight:600;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.opt-wrap.sel .opt-letter { background:#1a2f6a; color:#7eb3ff; }
.opt-text { font-size:0.9rem; color:#FFFFFF; font-weight:500; }
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
.res-sub { font-size:0.9rem; color:#FFFFFF; font-family:'DM Mono',monospace; margin-top:3px; }
.result-bad .res-sub { color:#705a5a; }
.res-next { margin-top:10px; padding-top:10px; border-top:1px solid #1e3a28;
    font-size:0.78rem; color:#4a7060; font-family:'DM Mono',monospace; }
.result-bad .res-next { border-color:#3a1e22; color:#FFFFFF; }

.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }
.metric-box { background:#0d1222; border:1px solid #1a2040; border-radius:12px;
    padding:14px 16px; }
.metric-val { font-size:1.5rem; font-weight:800; color:#7eb3ff;
    font-family:'DM Mono',monospace; line-height:1; }
.metric-lbl { font-size:0.88rem; color:#FFFFFF; margin-top:4px;
    text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }

.prog-track { height:4px; background:#141a2e; border-radius:99px; margin:8px 0 18px; overflow:hidden; }
.prog-fill  { height:100%; border-radius:99px;
    background:linear-gradient(90deg,#2a4fff,#7eb3ff);
    transition:width 0.4s cubic-bezier(0.4,0,0.2,1); }

.agent-card { background:#0d1222; border:1px solid #1a2040; border-radius:14px;
    padding:18px 20px; margin-bottom:10px; }
.agent-tag  { display:inline-block; font-size:0.88rem; font-weight:600;
    padding:2px 8px; border-radius:99px; margin-top:4px;
    font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.tag-rl   { background:#1a3a5e; color:#5ba3ff; }
.tag-on   { background:#1a3a2e; color:#4ecb8a; }
.tag-deep { background:#2e1a3a; color:#b87dff; }
.tag-base { background:#2e2a1a; color:#d4a04a; }
.agent-desc { font-size:0.98rem; color:#b8c1e0; margin-top:6px; line-height:1.5; }

.sum-card { background:linear-gradient(135deg,#111830,#0d1222);
    border:1px solid #1e2a4a; border-radius:16px; padding:24px; text-align:center; }
.sum-val { font-size:2.2rem; font-weight:800; color:#7eb3ff; font-family:'DM Mono',monospace; }
.sum-lbl { font-size:0.75rem; color:#FFFFFF; text-transform:uppercase;
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

.section-label { font-size:0.88rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; color:FFFFFF; font-family:'DM Mono',monospace;
    margin-bottom:10px; }

.skill-row { margin-bottom:10px; }
.skill-row-header { display:flex; justify-content:space-between;
    font-size:0.78rem; color:#FFFFFF; margin-bottom:4px; }
.skill-bar-track { height:5px; background:#141a2e; border-radius:99px; overflow:hidden; }
.skill-bar-fill  { height:100%; border-radius:99px; transition:width 0.6s ease; }

.hist-row { display:flex; align-items:center; justify-content:space-between;
    padding:9px 14px; border-radius:9px; background:#0d1222;
    border:1px solid #141c34; margin-bottom:4px; gap:8px; }

.insight-up   { background:#0a2018; border:1px solid #1a4030; border-radius:12px;
    padding:16px; text-align:center; }
.insight-hold { background:#201800; border:1px solid #4a3800; border-radius:12px;
    padding:16px; text-align:center; }
.insight-down { background:#200810; border:1px solid #4a1820; border-radius:12px;
    padding:16px; text-align:center; }
.insight-val  { font-size:1.6rem; font-weight:800; font-family:'DM Mono',monospace; }
.insight-up   .insight-val { color:#3dd68c; }
.insight-hold .insight-val { color:#f0b429; }
.insight-down .insight-val { color:#f25f6a; }
.insight-lbl  { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
    font-weight:600; margin-top:4px; }
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
    "Q-Learning":       {"tag": "value-based RL",     "cls": "tag-rl",   "icon": "◈",
                         "desc": "Off-policy TD. Bootstraps from greedy next action. Trained via train.py."},
    "SARSA":            {"tag": "on-policy RL",        "cls": "tag-on",   "icon": "◇",
                         "desc": "On-policy TD. Updates using the action actually taken — more conservative."},
    "DQN":              {"tag": "deep RL",             "cls": "tag-deep", "icon": "⬡",
                         "desc": "Neural net Q-function with experience replay and a target network."},
    "Policy Iteration": {"tag": "model-based RL",      "cls": "tag-rl",   "icon": "◆",
                         "desc": "Solves the discretised MDP exactly via Bellman equations."},
    "Random":           {"tag": "baseline",            "cls": "tag-base", "icon": "○",
                         "desc": "Picks difficulty uniformly at random. Lower-bound baseline."},
    "Rule-Based":       {"tag": "heuristic baseline",  "cls": "tag-base", "icon": "▷",
                         "desc": "Threshold heuristic: low accuracy → easier, high accuracy → harder."},
}

DIFF_ACTIONS = {0: "Easy", 1: "Medium", 2: "Hard"}
DIFF_ORDER   = ["Easy", "Medium", "Hard"]
N_QUESTIONS  = 15


# ═════════════════════════════════════════════════════════════════
#  TRAIN / LOAD AGENTS — same pipeline as train.py
# ═════════════════════════════════════════════════════════════════

import pickle

AGENTS_CACHE_FILE = MODELS_DIR / "trained_agents.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train_agents():
    import pandas as pd
    qbank   = build_sample_question_bank()
    obs_dim = AdaptiveQuizEnv(question_bank=qbank).observation_space.shape[0]

    # ── Load from disk if already trained ─────────────────────────
    if AGENTS_CACHE_FILE.exists():
        with open(AGENTS_CACHE_FILE, "rb") as f:
            saved = pickle.load(f)
        return saved["agents"], saved["irt"], qbank

    # ── First run: train everything ────────────────────────────────
    df = load_dataset(use_real=True)
    train_df_data, _ = train_test_student_split(df)

    irt = IRTCalibrator(model="2PL")
    irt.fit(train_df_data)

    env = AdaptiveQuizEnv(question_bank=qbank, irt_calibrator=irt, seed=0)

    agents_list = [QLearningAgent(seed=1), SARSAAgent(seed=2), DQNAgent(obs_dim=obs_dim, seed=3)]
    pi_agent    = PolicyIterationAgent()
    pi_agent.fit()

    for agent in agents_list:
        sarsa = isinstance(agent, SARSAAgent)
        for _ in range(200):
            obs, _ = env.reset()
            done   = False
            if sarsa:
                action = agent.act(obs)
            while not done:
                if sarsa:
                    nobs, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    na   = agent.act(nobs)
                    agent.update(obs, action, r, nobs, done, na)
                    action, obs = na, nobs
                else:
                    action = agent.act(obs)
                    nobs, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    agent.update(obs, action, r, nobs, done)
                    obs = nobs
        if hasattr(agent, "epsilon"):
            agent.epsilon = agent.epsilon_min

    agent_dict = {a.name: a for a in agents_list + [pi_agent] + ALL_BASELINES}

    # ── Save to disk ───────────────────────────────────────────────
    with open(AGENTS_CACHE_FILE, "wb") as f:
        pickle.dump({"agents": agent_dict, "irt": irt}, f)

    return agent_dict, irt, qbank
# ═════════════════════════════════════════════════════════════════
#  HUMAN QUIZ SESSION WRAPPER
#  The real AdaptiveQuizEnv.step() uses an IRT student simulator.
#  Here we intercept: show the question to the human, receive their
#  answer, then manually advance the env state (skill tracker, obs,
#  reward) so the agent's .act(obs) sees the correct state.
# ═════════════════════════════════════════════════════════════════

class HumanQuizSession:
    """
    Wraps AdaptiveQuizEnv for human play.
    Keeps env's skill tracker, IRT model, obs vector, and forgetting
    model all in sync with the human's actual answers.
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
        self.history      = []
        self.total_reward = 0.0

        # Agent picks first action from initial obs
        self.current_action     = self.agent.act(self.obs)
        self.current_difficulty = DIFF_ACTIONS[self.current_action]
        self._current_q         = self._pick_question(self.current_difficulty)

    # ── Private helpers ───────────────────────────────────────────

    def _pick_question(self, difficulty: str) -> Question:
        used = {h["question_id"] for h in self.history}
        mcq_bank = [q for q in self.env.question_bank
                if q.question_type == "mcq" and q.options is not None]
        pool = [q for q in mcq_bank
                if q.difficulty == difficulty and q.question_id not in used]
        if not pool:
            pool = [q for q in mcq_bank if q.question_id not in used]
        if not pool:
            pool = mcq_bank
        idx = int(self.env._rng.integers(len(pool)))
        return pool[idx]

    # ── Public API ────────────────────────────────────────────────

    @property
    def current_question(self) -> Question:
        return self._current_q

    @property
    def step_number(self) -> int:
        return len(self.history)

    def submit_answer(self, chosen_option_idx: int) -> dict:
        """
        Receive human answer, update env state, advance agent.
        Mirrors AdaptiveQuizEnv.step() but uses the human's response
        instead of the IRT simulator.
        """
        q       = self._current_q
        correct = (chosen_option_idx == q.correct_option)
        diff    = self.current_difficulty
        action  = self.current_action
        skill   = q.skill_id

        # ── Update env internals (mirrors env.step logic) ──────────
        rt_ms = 8000 if correct else 15000
        self.env._sim_time += 300
        self.env.skill_tracker.update(skill, correct, diff, self.env._sim_time)
        self.env._recent_correct.append(int(correct))
        if len(self.env._recent_correct) > 5:
            self.env._recent_correct.pop(0)
        self.env._step += 1

        # Reward (same formula as env._compute_reward)
        reward = self.env._compute_reward(correct, action, rt_ms, skill)
        self.total_reward += reward

        # New observation after this human response
        new_obs = self.env._get_obs()

        # ── Update agent with (s, a, r, s', done) ─────────────────
        n_done  = self.step_number + 1
        is_done = (n_done >= self.n_questions)

        if isinstance(self.agent, SARSAAgent):
            next_action = self.agent.act(new_obs)
            self.agent.update(self.obs, action, reward, new_obs, is_done, next_action)
        else:
            self.agent.update(self.obs, action, reward, new_obs, is_done)
            next_action = self.agent.act(new_obs)

        # ── Record ─────────────────────────────────────────────────
        entry = {
            "q_num":         n_done,
            "question_id":   q.question_id,
            "skill":         skill,
            "difficulty":    diff,
            "action":        action,
            "correct":       correct,
            "reward":        reward,
            "next_diff":     DIFF_ACTIONS[next_action],
            "overall_skill": self.env.skill_tracker.overall_level(),
            "obs_acc":       float(self.obs[-1]),   # recent accuracy in obs
        }
        self.history.append(entry)

        # ── Advance state ──────────────────────────────────────────
        self.obs                = new_obs
        self.done               = is_done
        self.current_action     = next_action
        self.current_difficulty = DIFF_ACTIONS[next_action]

        return entry
    def advance_question(self):
        """Call this ONLY when the user clicks Next Question →"""
        if not self.done:
            self._current_q = self._pick_question(self.current_difficulty)

    def skill_summary(self) -> list:
        rows = []
        for sk, state in self.env.skill_tracker.skills.items():
            rows.append({
                "skill":   sk,
                "level":   round(state.level, 3),
                "mastery": state.mastery_category,
                "correct": state.correct_count,
                "n":       state.practice_count,
            })
        return rows


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
                  margin-top:3px;letter-spacing:0.06em;">POWERED BY REAL RL AGENTS</div>
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
        <div style="font-size:0.88rem;color:#FFFFFF;font-family:'DM Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
          LIVE · {st.session_state.agent_name}</div>
        <div style="display:flex;flex-direction:column;gap:7px;">
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#FFFFFF;">
            <span>Questions</span>
            <span style="color:#7eb3ff;font-family:'DM Mono',monospace;">
              {n_done}/{N_QUESTIONS}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#FFFFFF;">
            <span>Accuracy</span>
            <span style="color:#3dd68c;font-family:'DM Mono',monospace;">{acc:.0f}%</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#FFFFFF;">
            <span>Reward</span>
            <span style="color:#f0b429;font-family:'DM Mono',monospace;">
              {sess.total_reward:+.1f}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#FFFFFF;">
            <span>Next diff.</span>
            <span style="font-family:'DM Mono',monospace;color:{diff_c};">{diff}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#FFFFFF;">
            <span>Skill level</span>
            <span style="color:#c8d4f5;font-family:'DM Mono',monospace;">
              {sess.env.skill_tracker.overall_level():.3f}</span></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset", use_container_width=True):
        st.cache_resource.clear()
        if AGENTS_CACHE_FILE.exists():
            AGENTS_CACHE_FILE.unlink()   # forces retrain on next load
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
      <div style="font-size:0.9rem;color:#a8b4d0;line-height:1.75;max-width:580px;margin-bottom:28px;">
        
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3], [
        ("🎯","IRT-Calibrated","2PL Item Response Theory calibrates difficulty per question."),
        ("🧠","Multi-Skill Observation","Agent sees 5 skill levels + progress + recent accuracy."),
        ("📐","ZPD Reward","Bonus reward when difficulty matches your Zone of Proximal Development."),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#0d1222;border:1px solid #1a2040;border-radius:14px;
                        padding:20px;height:170px;">
              <div style="font-size:1.4rem;margin-bottom:10px;">{icon}</div>
              <div style="font-size:0.98rem;font-weight:700;color:#c0c8ec;margin-bottom:6px;">
                {title}</div>
              <div style="font-size:0.90rem;color:#FFFFFF;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("Loading trained agents (first run ~20 s)…"):
        load_or_train_agents()
    st.success("✓ Agents ready")

    if st.button("Start Quiz →"):
        st.session_state.page = "Quiz Simulation"
        st.rerun()


# ═════════════════════════════════════════════════════════════════
#  PAGE: QUIZ SIMULATION
# ═════════════════════════════════════════════════════════════════

elif st.session_state.page == "Quiz Simulation":

    agents, irt, qbank = load_or_train_agents()

    # ── Agent selection ──────────────────────────────────────────
    if st.session_state.agent_name is None:
        st.markdown('<div style="font-size:2rem;font-weight:800;color:#c8d4f5;'
                    'letter-spacing:-0.02em;margin-bottom:6px;">Choose your agent-</div>',
                    unsafe_allow_html=True)


        col1, col2 = st.columns(2)
        for i, (name, meta) in enumerate(AGENT_META.items()):
            with (col1 if i % 2 == 0 else col2):
                if st.button(f"{meta['icon']}  {name}",
                             key=f"pick_{name}", use_container_width=True):
                    st.session_state.agent_name = name
                    st.session_state.session = HumanQuizSession(
                        agent=agents[name], qbank=qbank, irt=irt)
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
            Your answers updated the agent's skill model in real time.</div>
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
            · Agent: {st.session_state.agent_name}</span>
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
                        &nbsp;·&nbsp; Skill: {res['skill']}
                        &nbsp;·&nbsp; Difficulty: {res['difficulty']}
                      </div>
                      <div class="res-next">
                        {st.session_state.agent_name}.act(obs)
                        &nbsp; obs[recent_acc]={obs_acc:.2f}
                        &nbsp;→&nbsp; next difficulty:
                        <strong>{next_d}</strong>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                correct_text = (q.options[q.correct_option]
                                if q.options else "—")
                st.markdown(f"""
                <div class="result-bad">
                  <div class="res-top">
                    <div class="res-icon">❌</div>
                    <div>
                      <div class="res-title">Incorrect</div>
                      <div class="res-sub">
                        Correct answer: <strong>{correct_text}</strong>
                        &nbsp;·&nbsp; Reward: <strong>{reward:.2f}</strong>
                      </div>
                      <div class="res-next">
                        {st.session_state.agent_name}.act(obs)
                        &nbsp; obs[recent_acc]={obs_acc:.2f}
                        &nbsp;→&nbsp; next difficulty:
                        <strong>{next_d}</strong>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            if st.button("Next Question →"):
                sess.advance_question()
                st.session_state.submitted   = False
                st.session_state.last_result = None
                st.session_state.sel_opt     = None
                st.rerun()

    # ── Right: live metrics ──────────────────────────────────────
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
          <div class="metric-lbl">Overall Skill Level</div></div>
        """, unsafe_allow_html=True)

        # Per-skill bars (live from env.skill_tracker)
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

        # History log
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
      How your real answers shaped the agent's difficulty policy.</div>
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
              <span style="font-size:0.72rem;color:#FFFFFF;flex:1;">{h['skill']}</span>
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
