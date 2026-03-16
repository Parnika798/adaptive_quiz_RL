# Adaptive Quiz Difficulty using Reinforcement Learning

## Overview
This project develops an **adaptive quiz system** that dynamically adjusts question difficulty using **Reinforcement Learning (RL)**.  
The goal is to maintain questions within the learner’s **optimal challenge zone**, improving engagement and learning efficiency.

Traditional quiz systems use fixed difficulty levels, which often result in:
- Questions that are **too easy → minimal learning**
- Questions that are **too difficult → student frustration**

This system learns a **difficulty selection policy** that adapts to a student’s performance to maximize learning rewards.

---

# Methodology

The quiz environment is modeled as a **Markov Decision Process (MDP)**.

## State Representation
The RL agent observes a continuous state vector containing:

- Skill levels in **5 domains**
  - Algebra  
  - Geometry  
  - Statistics  
  - Calculus  
  - Arithmetic
- Question progress within the quiz
- Recent student accuracy (last 5 responses)

## Actions
The agent selects the difficulty of the next question:

- Easy  
- Medium  
- Hard

## Transition Dynamics
Student responses are simulated using **Item Response Theory (IRT)** models, which estimate the probability of a correct response based on:

- Student ability
- Question difficulty
- Question discrimination

Skill levels are updated after each response using a **multi-skill learning model**.

## Reward Function
The reward function encourages the agent to select appropriate difficulty levels.

Reward components include:

- Correct answer reward (scaled by difficulty)
- Speed bonus for quick correct responses
- **Zone of Proximal Development bonus** when difficulty matches student ability
- Penalty when difficulty is too easy or too difficult

---

# Algorithms

## Reinforcement Learning Agents
The following RL algorithms are implemented and compared:

- **Q-Learning** — off-policy tabular value learning  
- **SARSA** — on-policy temporal difference learning  
- **Deep Q-Network (DQN)** — neural network approximation of Q-values  
- **Policy Iteration** — model-based dynamic programming approach  

## Baseline Strategies
To evaluate the effectiveness of RL agents, the following baselines are used:

- Random difficulty selection
- Fixed difficulty (Easy / Medium / Hard)
- Rule-based heuristic using student accuracy

---

# Dataset

Experiments use the **ASSISTments 2009–2010 educational dataset**, which contains real student response data.

If the dataset is unavailable, the system generates **synthetic student responses with the same schema**.

Dataset features include:

- Student ability estimates
- Question skill tags
- Historical correctness data

Data is split into **training and test student populations** to prevent data leakage.

---

# Experimental Pipeline

The training and evaluation pipeline consists of:

1. Loading student response data  
2. Estimating question parameters using **Item Response Theory**  
3. Training reinforcement learning agents through simulated quiz sessions  
4. Evaluating agents on a **held-out test student population**  
5. Comparing performance against baseline strategies  

---

# Results

Performance is evaluated using:

- **Average cumulative reward**
- **Student answer accuracy**
- **Difficulty adaptation patterns**

Results show that **reinforcement learning agents outperform non-adaptive baselines**.

<img width="1264" height="615" alt="newplot" src="https://github.com/user-attachments/assets/b0097704-b41c-43ec-8eb8-d83646acaeb4" />
<img width="1264" height="615" alt="newplot (1)" src="https://github.com/user-attachments/assets/158a3899-c9f1-4a45-ab46-fcddc3b41892" />
<img width="1264" height="615" alt="newplot (2)" src="https://github.com/user-attachments/assets/72642192-a146-4c9d-9c1d-9be6f6f0d77b" />
<img width="1264" height="615" alt="newplot (3)" src="https://github.com/user-attachments/assets/4d18aded-707f-4d6f-82c0-8aba7fa209d0" />





### Key Observations
- RL agents learned to **increase difficulty as student skill improved**
- Difficulty adjustments followed **student learning progression**
- Fixed difficulty strategies produced **lower reward and accuracy**

### Algorithm Comparison
- **Policy Iteration and Q-Learning** achieved the highest cumulative rewards
- **DQN** performed well with continuous state representations
- **SARSA** produced slightly more conservative difficulty adjustments

---

# Statistical Validation

Performance differences were validated using statistical testing:

- **Mann–Whitney U test** for non-parametric comparison
- **Cohen’s d effect size** for measuring practical significance

Statistical analysis confirmed that **RL-based adaptive strategies outperform baseline methods**.

---

# Running the Project


```bash
pip install -r requirements.txt

streamlit run app.py

python train.py
