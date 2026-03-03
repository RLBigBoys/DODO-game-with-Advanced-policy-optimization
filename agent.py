import numpy as np
import random
from rl_config import RLConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
#                 NN MODELS
# ==========================================

# class CNNModel(nn.Module):
#     def __init__(self, config: RLConfig):


#         self.conv1 = nn.Conv2d()

# ==========================================
#                  POLICIES
# ==========================================

class BasePolicy:
    """Base class for all policies."""
    def __init__(self, config: RLConfig):
        self.config = config
        
    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError
        
    def save(self, filepath: str):
        np.save(filepath, np.array([self.config.CLICK_PROBABILITY]))
        
    def load(self, filepath: str):
        try:
            weights = np.load(filepath)
            self.config.CLICK_PROBABILITY = float(weights[0])
        except FileNotFoundError:
            pass

class DummyPolicy(BasePolicy):
    """Random policy (uniform random clicks)."""
    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.config.CLICK_PROBABILITY:
            return 1  # Click
        return 0
        
    def save(self, filepath: str):
        np.save(filepath, np.array([self.config.CLICK_PROBABILITY]))
        
    def load(self, filepath: str):
        try:
            weights = np.load(filepath)
            self.config.CLICK_PROBABILITY = float(weights[0])
        except FileNotFoundError:
            pass

class CNNPolicy(BasePolicy, nn.Module):
    """Trainable CNN-based policy (PyTorch)."""
    def __init__(self, config: RLConfig):
        BasePolicy.__init__(self, config)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_channels = config.FRAMES_STACK * config.CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out_shape(in_channels, config.FRAME_HEIGHT, config.FRAME_WIDTH)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, config.ACTION_SPACE_SIZE),
        )

        self.to(self.device)

    def _get_conv_out_shape(self, in_channels: int, h: int, w: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            x = self.conv(x)
            return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape [B, in_channels, H, W].
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert state (FRAMES, H, W, C) into tensor [1, in_channels, H, W].
        """
        s = torch.from_numpy(state).float() / 255.0
        # [F, H, W, C] -> [F, C, H, W] -> [F*C, H, W]
        s = s.permute(0, 3, 1, 2).contiguous()
        f, c, h, w = s.shape
        s = s.view(1, f * c, h, w)
        return s.to(self.device)

    def get_action(self, state: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            x = self._preprocess_state(state)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
        return int(action)

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        try:
            state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
            self.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load weights from {filepath} ({e}). Starting fresh.")


# ==========================================
#                  TRAINERS
# ==========================================

class BaseTrainer:
    """Base class for all training algorithms."""
    def __init__(self, policy: BasePolicy, config: RLConfig):
        self.policy = policy
        self.config = config
        
    def train_step(self, batch_of_trajectories) -> None:
        raise NotImplementedError

class DummyTrainer(BaseTrainer):
    """No-op trainer used for tests (no learning)."""
    def train_step(self, batch_of_trajectories) -> None:
        pass

class ReinforceTrainer(BaseTrainer):
    def __init__(self, policy: BasePolicy, config: RLConfig):
        super().__init__(policy, config)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.LEARNING_RATE)

    def train_step(self, batch_of_trajectories) -> None:
        # trajectory[t] = (state, action, reward)
        self.optimizer.zero_grad()
        batch_loss = []

        for traj in batch_of_trajectories:
            states, actions, rewards = zip(*traj)
            
            states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32)
            states_tensor = states_tensor.permute(0, 1, 4, 2, 3).reshape(len(states), -1, 84, 84)
            
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            
            logits = self.policy(states_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            
            log_probs = distribution.log_prob(actions_tensor) 

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.config.GAMMA * G
                returns.insert(0, G)
            
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            
            traj_loss = -(returns_tensor * log_probs).sum()
            batch_loss.append(traj_loss)

        total_loss = torch.stack(batch_loss).mean()
        total_loss.backward()
        self.optimizer.step()
 

class ReinforceBaselineTrainer(BaseTrainer):
    def __init__(self, policy: BasePolicy, config: RLConfig):
        super().__init__(policy, config)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.LEARNING_RATE)
        self.baseline = None

    def train_step(self, batch_of_trajectories) -> None:
        # trajectory[t] = (state, action, reward)
        self.optimizer.zero_grad()
        batch_loss = []

        for traj in batch_of_trajectories:
            states, actions, rewards = zip(*traj)
            
            states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32)
            states_tensor = states_tensor.permute(0, 1, 4, 2, 3).reshape(len(states), -1, 84, 84)
            
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            
            logits = self.policy(states_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            
            log_probs = distribution.log_prob(actions_tensor) 

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.config.GAMMA * G
                returns.insert(0, G)

            if self.baseline is None:
                self.baseline = np.mean(returns)
            else:
                self.baseline = 0.9 * self.baseline + 0.1 * np.mean(returns)
            
            returns_tensor = torch.tensor(returns - self.baseline, dtype=torch.float32)
            
            traj_loss = -(returns_tensor * log_probs).sum()
            batch_loss.append(traj_loss)

        total_loss = torch.stack(batch_loss).mean()
        total_loss.backward()
        self.optimizer.step()
 

class TrpoTrainer(BaseTrainer):
    """
    Minimal implementation of Trust Region Policy Optimization (TRPO)
    for a discrete action space.
    """
    def __init__(self, policy: BasePolicy, config: RLConfig):
        super().__init__(policy, config)
        if not isinstance(self.policy, CNNPolicy):
            raise ValueError("TrpoTrainer expects policy of type CNNPolicy.")

        self.max_kl = 1e-2
        self.cg_damping = 1e-2
        self.cg_iters = 10
        self.backtrack_coeff = 0.8
        self.backtrack_iters = 10

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for p in self.policy.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx : idx + numel].view_as(p))
            idx += numel

    def _flat_grad(self, loss: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        grads = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=retain_graph)
        return torch.cat([g.view(-1) for g in grads])

    def _get_dist(self, states: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.policy(states)
        return torch.distributions.Categorical(logits=logits)

    def _fisher_vector_product(
        self,
        states: torch.Tensor,
        old_dist: torch.distributions.Categorical,
        v: torch.Tensor,
    ) -> torch.Tensor:
        kl = torch.distributions.kl_divergence(old_dist, self._get_dist(states)).mean()
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])
        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad2 = torch.cat([g.contiguous().view(-1) for g in grads2])
        return flat_grad2 + self.cg_damping * v

    def _conjugate_gradient(
        self,
        states: torch.Tensor,
        old_dist: torch.distributions.Categorical,
        b: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.cg_iters):
            avp = self._fisher_vector_product(states, old_dist, p)
            alpha = rdotr / (torch.dot(p, avp) + 1e-8)
            x += alpha * p
            r -= alpha * avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _prepare_batch(self, batch_of_trajectories):
        states = []
        actions = []
        returns = []

        gamma = getattr(self.config, "GAMMA", 0.99)

        for trajectory in batch_of_trajectories:
            rewards = [step[2] for step in trajectory]
            G = 0.0
            discounted = []
            for r in reversed(rewards):
                G = r + gamma * G
                discounted.insert(0, G)

            for (state, action, _), ret in zip(trajectory, discounted):
                states.append(state)
                actions.append(action)
                returns.append(ret)

        states_np = np.stack(states, axis=0)  # [N, F, H, W, C]
        s = torch.from_numpy(states_np).float() / 255.0
        # [N, F, H, W, C] -> [N, F, C, H, W] -> [N, F*C, H, W]
        s = s.permute(0, 1, 4, 2, 3).contiguous()
        n, f, c, h, w = s.shape
        s = s.view(n, f * c, h, w).to(self.policy.device)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.policy.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)

        advantages = returns_t - returns_t.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        return s, actions_t, advantages

    def train_step(self, batch_of_trajectories) -> None:
        if not batch_of_trajectories:
            return

        self.policy.train()

        states, actions, advantages = self._prepare_batch(batch_of_trajectories)

        with torch.no_grad():
            old_dist = self._get_dist(states)
            old_log_probs = old_dist.log_prob(actions)

        def surrogate_loss():
            dist = self._get_dist(states)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        loss = surrogate_loss()
        loss_grad = self._flat_grad(loss)

        step_direction = self._conjugate_gradient(states, old_dist, -loss_grad)

        shs = 0.5 * (step_direction * self._fisher_vector_product(states, old_dist, step_direction)).sum()
        shs = shs.abs() + 1e-8
        step_size = torch.sqrt(self.max_kl / shs)
        full_step = step_direction * step_size

        old_params = self._flatten_params()

        def set_and_eval(step: torch.Tensor):
            new_params = old_params + step
            self._set_flat_params(new_params)
            with torch.no_grad():
                dist = self._get_dist(states)
                kl = torch.distributions.kl_divergence(old_dist, dist).mean()
                new_loss = surrogate_loss()
            return kl, new_loss

        step = full_step
        for _ in range(self.backtrack_iters):
            kl, new_loss = set_and_eval(step)
            if kl <= self.max_kl and new_loss < loss:
                break
            step = step * self.backtrack_coeff
        else:
            self._set_flat_params(old_params)


# ==========================================
#                   AGENT
# ==========================================

class Agent:
    POLICIES = {
        "dummy": DummyPolicy,
        "cnn": CNNPolicy
    }
    
    TRAINERS = {
        "dummy": DummyTrainer,
        "reinforce": ReinforceTrainer,
        "reinforce_baseline": ReinforceBaselineTrainer,
        "trpo": TrpoTrainer
    }

    def __init__(self, config: RLConfig):
        self.config = config
        
        # 1. Load policy (Policy Structure)
        policy_class = self.POLICIES.get(self.config.POLICY_TYPE)
        if not policy_class:
            raise ValueError(f"Unknown POLICY_TYPE: {self.config.POLICY_TYPE}")
        self.policy = policy_class(config)
        
        # 2. Load training algorithm (Train Algorithm Structure)
        trainer_class = self.TRAINERS.get(self.config.TRAIN_METHOD)
        if not trainer_class:
            raise ValueError(f"Unknown TRAIN_METHOD: {self.config.TRAIN_METHOD}")
        self.trainer = trainer_class(self.policy, config)

    def get_action(self, state: np.ndarray) -> int:
        """Pass state into the selected policy and return chosen action."""
        return self.policy.get_action(state)
        
    def train_step(self, batch_of_trajectories) -> None:
        """Pass collected trajectories to the training algorithm to update policy weights."""
        self.trainer.train_step(batch_of_trajectories)
            
    def save(self, filepath: str):
        self.policy.save(filepath)
        print(f"Agent weights saved to {filepath}")
        
    def load(self, filepath: str):
        if hasattr(self.config, "LOAD_BEST_WEIGHTS") and self.config.LOAD_BEST_WEIGHTS:
            filepath = filepath.replace(".npy", "_best.npy")
            
        self.policy.load(filepath)
        print(f"Agent weights loaded from {filepath}")
