"""
Adaptive Quiz RL Training & Evaluation Pipeline
-----------------------------------------------
• Trains RL agents
• Evaluates on test students
• Performs statistical testing
• Generates Plotly visualizations
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.environment import AdaptiveQuizEnv
from models.agents import QLearningAgent, SARSAAgent, PolicyIterationAgent, DQNAgent
from baselines.baseline_agents import ALL_BASELINES
from data.dataset_loader import load_dataset, train_test_student_split
from utils.irt_calibration import IRTCalibrator
from utils.answer_evaluator import build_sample_question_bank


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Episode Runner
# --------------------------------------------------

def run_episode(env, agent, train=True):

    obs, _ = env.reset()
    done = False
    total_reward = 0

    sarsa = isinstance(agent, SARSAAgent)

    if sarsa:
        action = agent.act(obs)

    while not done:

        if sarsa:

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = agent.act(next_obs)

            if train:
                agent.update(obs, action, reward, next_obs, done, next_action)

            action = next_action
            obs = next_obs

        else:

            action = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if train:
                agent.update(obs, action, reward, next_obs, done)

            obs = next_obs

        total_reward += reward

    summary = env.get_episode_summary()
    summary["total_reward"] = total_reward

    return summary


# --------------------------------------------------
# Train Agent
# --------------------------------------------------

def train_agent(agent, env, episodes=300):

    records = []

    for ep in range(episodes):

        summary = run_episode(env, agent, train=True)

        records.append({
            "episode": ep,
            "agent": agent.name,
            "total_reward": summary["total_reward"],
            "accuracy": summary["accuracy"],
            "skill_gain": summary["skill_gain"]
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# Evaluate Agent
# --------------------------------------------------

def evaluate_agent(agent, env, episodes=100):

    records = []

    for ep in range(episodes):

        summary = run_episode(env, agent, train=False)

        records.append({
            "episode": ep,
            "agent": agent.name,
            "total_reward": summary["total_reward"],
            "accuracy": summary["accuracy"],
            "skill_gain": summary["skill_gain"]
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# Statistical Testing
# --------------------------------------------------

def statistical_comparison(results_df):

    agents = results_df["agent"].unique()
    rows = []

    for i, a1 in enumerate(agents):

        for a2 in agents[i+1:]:

            x = results_df[results_df["agent"] == a1]["total_reward"]
            y = results_df[results_df["agent"] == a2]["total_reward"]

            u, p = stats.mannwhitneyu(x, y)

            d = (x.mean() - y.mean()) / np.sqrt((x.std()**2 + y.std()**2)/2)

            rows.append({
                "agent_A": a1,
                "agent_B": a2,
                "p_value": p,
                "cohen_d": d
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Plotly Visualizations
# --------------------------------------------------

def generate_plots(train_df, test_df, stat_df):

    print("\nGenerating Plotly visualizations...")

    # ------------------------------------------------
    # Learning Curves
    # ------------------------------------------------

    train_df["reward_smooth"] = (
        train_df.groupby("agent")["total_reward"]
        .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )

    fig = px.line(
        train_df,
        x="episode",
        y="reward_smooth",
        color="agent",
        title="RL Learning Curves"
    )

    fig.write_html(RESULTS_DIR / "learning_curves.html")
    fig.show()


    # ------------------------------------------------
    # Performance Comparison
    # ------------------------------------------------

    perf = test_df.groupby("agent")["total_reward"].mean().reset_index()

    fig = px.bar(
        perf,
        x="agent",
        y="total_reward",
        title="Average Reward per Agent",
        color="agent"
    )

    fig.write_html(RESULTS_DIR / "performance_comparison.html")
    fig.show()


    # ------------------------------------------------
    # Accuracy Comparison
    # ------------------------------------------------

    acc = test_df.groupby("agent")["accuracy"].mean().reset_index()

    fig = px.bar(
        acc,
        x="agent",
        y="accuracy",
        title="Accuracy Comparison",
        color="agent"
    )

    fig.write_html(RESULTS_DIR / "accuracy_comparison.html")
    fig.show()


    # ------------------------------------------------
    # Effect Size Heatmap
    # ------------------------------------------------

    agents = pd.unique(stat_df[["agent_A","agent_B"]].values.ravel())

    matrix = pd.DataFrame(np.nan, index=agents, columns=agents)

    for _, row in stat_df.iterrows():

        matrix.loc[row["agent_A"], row["agent_B"]] = row["cohen_d"]
        matrix.loc[row["agent_B"], row["agent_A"]] = -row["cohen_d"]

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Effect Size Heatmap (Cohen's d)"
    )

    fig.write_html(RESULTS_DIR / "effect_size_heatmap.html")
    fig.show()


# --------------------------------------------------
# Full Experiment
# --------------------------------------------------

def run_full_experiment():

    print("="*60)
    print("ADAPTIVE QUIZ RL — FULL EXPERIMENT")
    print("="*60)

    df = load_dataset(use_real=True)

    train_df_data, test_df_data = train_test_student_split(df)

    print("\nCalibrating IRT...")

    irt = IRTCalibrator(model="2PL")
    irt.fit(train_df_data)

    qbank = build_sample_question_bank()

    train_env = AdaptiveQuizEnv(question_bank=qbank, irt_calibrator=irt)
    test_env = AdaptiveQuizEnv(question_bank=qbank, irt_calibrator=irt)

    obs_dim = train_env.observation_space.shape[0]

    agents = [
        QLearningAgent(),
        SARSAAgent(),
        DQNAgent(obs_dim=obs_dim)
    ]

    pi_agent = PolicyIterationAgent()
    pi_agent.fit()

    train_results = []

    print("\nTraining agents...")

    for agent in agents:

        df_train = train_agent(agent, train_env)

        train_results.append(df_train)

    train_df = pd.concat(train_results)

    print("\nEvaluating agents...")

    eval_agents = agents + [pi_agent] + ALL_BASELINES

    test_results = []

    for agent in eval_agents:

        df_eval = evaluate_agent(agent, test_env)

        test_results.append(df_eval)

    test_df = pd.concat(test_results)

    stat_df = statistical_comparison(test_df)

    train_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    stat_df.to_csv(RESULTS_DIR / "stats.csv", index=False)

    generate_plots(train_df, test_df, stat_df)

    print("\nExperiment complete.")
    print("Results saved to:", RESULTS_DIR)

    return train_df, test_df, stat_df


# --------------------------------------------------

if __name__ == "__main__":

    run_full_experiment()