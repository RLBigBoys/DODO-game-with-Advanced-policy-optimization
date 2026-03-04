# Reinforcement Learning Project: DODO Simulation

## 1. Problem Definition

### Task Description
The project focuses on training an autonomous Reinforcement Learning agent to play the **DODO Simulation**, a browser-based 3D block-stacking game. The agent interacts directly with a web environment running via Playwright, receiving visual observations (pixels) and sending discrete inputs (mouse clicks) to drop oscillating blocks onto a supporting tower.

### Objective of the Agent
The primary objective of the agent is to maximize the expected cumulative return by continuously stacking blocks directly on top of each other. This entails:
1. Identifying the optimal timing to drop a continuously sliding block onto the static platform below.
2. Building the highest possible tower (maximizing survival time).
3. Maintaining placement precision (maximizing perfectly overlapping area).
4. Collecting in-game DODO coins for sparse bonus rewards.

### Environment Dynamics
The game physics involve blocks that oscillate horizontally across the screen along a single axis at a time, alternating between the X-axis and Z-axis for each new block. When the agent acts (clicks), the current block falls downwards. If it lands on the previous block, the overlapping area between the falling block and the base block becomes the new platform surface for the next round. Any overhanging portion is sliced off and falls away using a Boolean geometric difference. The tower itself also exhibits simulated physical sway based on a multi-body chain simulation.

### Transitions
The core state transitions $p(s_{t+1} \mid s_t, a_t)$ are inherently **deterministic** within the JavaScript engine for a given tick rate. However, from the agent's perspective, the environment is framed as an **MDP with stochastic transitions** due to several factors:
- Asynchronous framerate latency between the Playwright browser and the Python training loop.
- The initial velocity of a newly spawned block is sampled uniformly $v \sim \mathcal{U}[v_{min}, v_{max}]$, with speed scaling up slightly every 10 successful placements.

---

## 2. Environment Specification

### State Representation ($S_t \in \mathcal{S}$)
The state $\mathcal{S}$ is represented by a dual-input observation tensor:
1. **Visual State ($S_t^{vis}$)**: A stack of the previous $N=5$ grayscale rendering frames resized to $84 \times 84$ pixels. This captures the spatial configuration and rate of movement of the block.
2. **Action History ($S_t^{act}$)**: A flat vector representing the last $N=5$ discrete actions $A_{t-N}, \dots, A_{t-1}$ taken by the agent, providing immediate short-term memory of its own control frequency.

### Action Space ($\mathcal{A}$)
The agent operates in a discrete action space with cardinality $|\mathcal{A}| = 2$:
- **0** (Wait): Do nothing. The block continues its horizontal oscillation $x_{t+dt} = x_t + v \cdot dt$.
- **1** (Click): Drop the block. Triggers the boolean slice operation and physically drops the block onto the tower.

### Transition Logic
- If $A_t=0$, the simulation advances by $dt$, and the block translates along its current axis $c \in \{X, Z\}$. The agent remains in the episode.
- If $A_t=1$, the block drops. 
    - If the overlap area $> 0$, the remaining area becomes the new block size, a new block spawns moving along the perpendicular axis, and the camera shifts upwards.
    - If the overlap area $= 0$, the episode flags termination.

### Episode Termination Conditions
An episode formally terminates at step $T$ if any of the following occur:
- **Game Over (Terminal State)**: A dropped block completely misses the platform below ($\frac{\text{New Area}}{\text{Old Area}} = 0$).
- **Time Horizon (Truncation)**: The agent reaches the maximum allowed temporal bounds (`TIME_HORIZON`), artificially ending the episode to prevent infinite loops.

### Full Reward Function ($R_t \sim p^R(\bullet \mid S_t, A_t)$)
The reward function is an event-driven mapping that provides gradients for the policy.

| Event                   | Condition                   | Reward mapping $R_t$                                                      |
| :---------------------- | :-------------------------- | :------------------------------------------------------------------------ |
| **Survival**            | Every physical step $A_t=0$ | $+0.001$                                                                  |
| **Coin Collection**     | Bounding box intersection   | $+R_{coins}$                                                              |
| **Placement Precision** | Successful drop $A_t=1$     | $+R_{area} \times \left( \frac{\text{New Area}}{\text{Old Area}} \right)$ |
| **Game Over**           | Missed drop $A_t=1$         | $-5.0$                                                                    |

---

## 3. History of Agent Evolution

Our development architecture went through the following iterations to stabilize learning:
1. **Iteration 0**: CNN Policy evaluated via REINFORCE, REINFORCE with Baseline, and finally TRPO. Initial state representation relied solely on the previous 5 visual frames.
2. **Iteration 1**: Added the simplest dense reward of `+0.001` per step for time spent surviving in the game.
3. **Iteration 2**: Introduced sparse conditional rewards correlating with the collection of ingame DODO Coins.
4. **Iteration 3**: Implemented block-precision placement rewards heavily punishing overhangs based on the ratio between the new resulting surface area and the previous old area.
5. **Iteration 4**: Supplemented the neural network's state input to include a vector of the **last 5 discrete actions** taking place alongside the 5 visual frames, dramatically boosting the agent's contextual awareness of its own immediate past decisions.
6. **Iteration 5**: Integrated **Advantage Standardization** across the entire batch (mean-centering and variance normalization) to stabilize gradient updates and added a **Categorical Entropy Bonus** to the loss function to drastically encourage exploration during early training.

---

## 4. Reproducibility

### Exact Commands to Run Training
To run the training loop from scratch using the baseline configuration:
```bash
# Ensure the local game server is running first
python -m http.server 8080

# To train using REINFORCE Baseline Policy
uv run python train.py --policy cnn
```

To resume training from the best checkpoint:
*Set `LOAD_FROM_CHECKPOINT = True` and `LOAD_BEST_WEIGHTS = True` in `rl_config.py`.*
```bash
uv run python train.py
```

### Exact Commands to Evaluate
To evaluate a fully trained agent and render the output without exploring (exploitation only):
```bash
uv run python evaluate.py
```

### Expected Output Description
During training, the script outputs the exact progression of steps and rewards to the active terminal. Upon termination (`Ctrl+C` once), it will output:
- `weights.npy` and `weights_best.npy`: The trained neural network weights.
- `learning_data.json`: The raw history of cumulative rewards and coins collected per episode.
- `learning_curve_reward.png`: Visual plot denoting Moving Average (10) cumulative rewards.
- `learning_curve_dodo_coins.png`: Visual plot denoting Moving Average (10) collected DODO coins.
- `config.json`: Environment hyperparameter snapshot locking the evaluation logic used.

---

## 5. Methodology and Results

### 5.1. REINFORCE

In the REINFORCE algorithm, we directly optimize the policy $\pi$ via a parametrized model $\pi^\theta$ with weights $\theta$. We want to solve an optimal control problem of a Markov Decision Process (MDP) by explicitly maximizing the expected trajectory return:
$$ v^\pi(s) := \mathbb{E} \left[ \sum_{t=0}^{\tau-1} \gamma^t R_t \mid S_0 = s \right] \to \max_{\pi}, s \in \mathcal{S} $$

With the state-action trajectory defined as $z_t := (s_t, a_t)$, the simplest form of the iterative gradient update rule from iteration $i \in \mathbb{Z}_{\ge 0}$ evaluated for $\tau$ steps follows:
$$ \theta_{i+1} \leftarrow \theta_i + \alpha \sum_{t=0}^{\tau-1} \gamma^t r_t^i \cdot \sum_{t=0}^{\tau-1} \nabla_\theta \ln \pi^\theta (a_t^i \mid s_t^i) $$

In practice, we estimate this via a batch of sampled episodes. The empirical policy-gradient estimator $\hat{g}$ over a batch of $N$ trajectories becomes:
$$ \hat{g} = \frac{1}{N} \sum_{k=1}^N \left( G(\tau^{(k)}) \right) \sum_{t=0}^{\tau_k-1} \nabla_\theta \ln \pi^\theta(a_t^{(k)} \mid s_t^{(k)}) $$

And the update rule is structured via gradient ascent:
$$ \theta \leftarrow \theta + \alpha \hat{g} $$

However, in our evolved execution (Iteration 5), we heavily stabilize the variance of this scalar magnitude via **Advantage Standardization**. Before the gradient evaluates, the individual empirical step returns $G_t^{(k)} = \sum_{t'=t}^{\tau_k-1} \gamma^{t'-t} r_{t'}^{(k)}$ are uniformly standardized against the whole batch to form an advantage-like proxy term:
$$ \hat{A}_t^{(k)} = \frac{G_t^{(k)} - \mu_G}{\sigma_G + \epsilon} $$

To dramatically encourage exploration during the early random-weight stages of training, a Categorical Entropy bonus $\mathcal{H}$ is appended to the policy loss computation using a scaling coefficient $c$. The true final empirical formula processed in our code follows:

$$ \text{Loss}_{policy} = - \left( \frac{1}{N} \sum_{k=1}^N \sum_{t=0}^{\tau_k-1} \hat{A}_t^{(k)} \cdot \ln \pi^\theta(a_t^{(k)} \mid s_t^{(k)}) \right) - c \cdot \frac{1}{N} \sum_{k=1}^N \sum_{t=0}^{\tau_k-1} \mathcal{H}(\pi^\theta(\bullet \mid s_t^{(k)})) $$

**Learning Curve (Cumulative Reward & Coins):**

**[PLACEHOLDER: Insert `learning_curve_reinforce.png` here]**

**Qualitative Demonstration:**

**[PLACEHOLDER: Insert REINFORCE Evaluation Gameplay.gif here]**

### 5.2. REINFORCE with Baseline

In REINFORCE, we introduce specific control variates to drastically reduce scalar variance in the sample mean. This is known as the baseline mechanism in policy gradient. A baseline $B_t$ is introduced into the expectation like:

$$ \mathbb{E}_{\pi^\theta} \left[ \sum_{t=0}^{\tau-1} \nabla_\theta \ln \pi^\theta(A_t \mid S_t) \cdot \left( \sum_{k=t}^{\tau-1} \gamma^k R_k - B_t \right) \right] $$

The requirement on the baseline is that it must be independent of $A_t$ conditioned on $S_t$. Conditional independence allows to factor out $\mathbb{E}[B_t \mid S_t]$ from the inner expectation, whereas the other, remaining, expectation is zero by the token $\int \nabla_\theta \pi^\theta = \nabla_\theta (\int \pi^\theta) = \nabla_\theta (1) = 0$ as in the "don't let the past distract you" trick.

In direct implementation within `agent.py`, rather than utilizing a value estimate Actor-Critic representation $\hat{v}(S_t)$, the baseline $b$ is computed uniformly across the batch using a dynamic Exponential Moving Average (EMA) of explicit step returns to iteratively track the expected aggregate rewards. For an incoming batch of $N$ trajectories, the baseline updates as follows:
$$ b \leftarrow 0.9 b + 0.1 \left( \frac{1}{N} \sum_{k=1}^N \frac{1}{\tau_k} \sum_{t=0}^{\tau_k-1} G_t^{(k)} \right) $$

Our code uniformly subtracts this moving scalar from all batch returns explicitly via a centered difference $A_t^{(k)} = G_t^{(k)} - b$. The resulting policy-gradient estimator becomes:
$$ \hat{g} = \frac{1}{N} \sum_{k=1}^N \sum_{t=0}^{\tau_k-1} \left( G_t^{(k)} - b \right) \nabla_\theta \ln \pi^\theta(a_t^{(k)} \mid s_t^{(k)}) $$

Before backpropagation, these advantages are structurally standardized (identically to Iteration 5's naive REINFORCE formulation) and uniformly subtracted by the exact Categorical Entropy modifier to optimize policy loss $\mathcal{L}_{policy}$.

**Learning Curve (Cumulative Reward & Coins):**

**[PLACEHOLDER: Insert `learning_curve_reinforce_baseline.png` here]**

**Qualitative Demonstration:**

**[PLACEHOLDER: Insert REINFORCE Baseline Evaluation Gameplay.gif here]**

### 5.3. Analysis of Results

**[PLACEHOLDER: Insert detailed comparative analysis of REINFORCE and REINFORCE Baseline performance relative to the naive baseline here]**
