## 5.3. TRPO (Trust Region Policy Optimization)

Let $\pi_{\theta_i}$ denote the behavior (old) policy that generated the batch, and let $\pi_{\theta}$ be the updated policy.
TRPO performs a careful policy update by maximizing a surrogate objective under a trust region constraint.

### Nominal TRPO

A nominal TRPO iteration reads:
$
\theta_{i+1} = \arg\max_{\theta}\ \hat{L}_{\theta_i}(\theta)
\quad \text{subject to} \quad
\bar{D}_{\mathrm{KL}}\!\left(\pi_{\theta_i}\ \|\ \pi_{\theta}\right) \le \delta,
$

where the (empirical) average KL is
$
\bar{D}_{\mathrm{KL}}\!\left(\pi_{\theta_i}\ \|\ \pi_{\theta}\right)
:= \mathbb{E}_{t}\!\left[
D_{\mathrm{KL}}\!\left(\pi_{\theta_i}(\cdot\mid S_t)\ \|\ \pi_{\theta}(\cdot\mid S_t)\right)
\right].
$

### Practical surrogate (importance sampling)

The practical TRPO surrogate is:
$
\hat{L}_{\theta_i}(\theta)
:= \mathbb{E}_{t}\!\left[
\frac{\pi_{\theta}(A_t\mid S_t)}{\pi_{\theta_i}(A_t\mid S_t)}\ \hat{A}_t
\right],
$
where $\hat{A}_t$ is an advantage estimate computed from the batch (standardized in our implementation).

### How it is solved in our code

In `agent.py`, TRPO is implemented for a categorical policy (discrete actions) using:
- the surrogate loss with ratio $r_t=\exp(\log\pi_{\theta}(A_t\!\mid\!S_t)-\log\pi_{\theta_i}(A_t\!\mid\!S_t))$;
- Fisher-vector product based on the mean KL divergence;
- **Conjugate Gradient** to compute the step direction;
- step rescaling to satisfy the KL budget (`max_kl`);
- **backtracking line search** until KL is within the trust region and the surrogate improves.

Default TRPO constants in our implementation: `max_kl=1e-2`, `cg_damping=1e-2`, `cg_iters=10`, `backtrack_coeff=0.8`, `backtrack_iters=10`.


## 5.4. PPO (Proximal Policy Optimization)

PPO is a practical variant of TRPO that replaces explicit KL constraints with a clipped surrogate objective.

Let $\pi_{\text{old}}$ be the behavior policy that generated the batch and $\pi_{\text{new}}$ be the updated policy.
Define the policy ratio:
$
r_t := \frac{\pi_{\text{new}}(A_t\mid S_t)}{\pi_{\text{old}}(A_t\mid S_t)}.
$

### PPO clipped surrogate

The PPO clipped objective is:
$
\hat{L}_{\mathrm{CLIP}}
:= \mathbb{E}_{t}\!\left[
\min\left(
r_t\,\hat{A}_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\,\hat{A}_t
\right)
\right].
$

### How it is implemented in our code

In `agent.py`, PPO:
- stores $\log\pi_{\text{old}}(A_t\mid S_t)$ from the rollout and recomputes $\log\pi_{\text{new}}(A_t\mid S_t)$ during updates;
- forms the ratio via `ratio = exp(logp_new - logp_old)` and applies clipping with `PPO_CLIP_EPS`;
- performs multiple epochs over the same batch (`PPO_EPOCHS`) with minibatches (`PPO_MINIBATCH_SIZE`);
- optionally adds entropy bonus (`PPO_ENTROPY_COEF`) and uses gradient clipping (`PPO_MAX_GRAD_NORM`).

All PPO hyperparameters are set in `rl_config.py`.