import numpy as np
import random
from rl_config import RLConfig
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


# ==========================================
#                 NN MODELS
# ==========================================

class CNNModel(nn.Module):
    def __init__(self, config: RLConfig):
        super().__init__()

        self.config = config

        output_shape = self.config.ACTION_SPACE_SIZE
        input_channels = self.config.FRAMES_STACK * self.config.CHANNELS

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride= 4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 
                                     self.config.FRAME_HEIGHT, 
                                     self.config.FRAME_WIDTH)
            dummy_out = self.features(dummy_input)
            self.linear_size = dummy_out.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.linear_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        features = self.features(x)
        return self.head(features)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)


# ==========================================
#                  POLICIES
# ==========================================

class BasePolicy:
    """Базовый класс для всех политик."""
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
    """Случайная политика (рандомные клики)."""
    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.config.CLICK_PROBABILITY:
            return 1 # Кликнуть
        return 0
        
    def save(self, filepath: str):
        np.save(filepath, np.array([self.config.CLICK_PROBABILITY]))
        
    def load(self, filepath: str):
        try:
            weights = np.load(filepath)
            self.config.CLICK_PROBABILITY = float(weights[0])
        except FileNotFoundError:
            pass

class CNNPolicy(BasePolicy):
    """Обучаемая политика на основе CNN (Например, PyTorch)."""
    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.model = CNNModel(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.OPTIMAIZER_LEARNING_RATE)
        
    def get_action(self, state: np.ndarray) -> int:
        state_permuted = torch.as_tensor(state, dtype=torch.float32).permute(0, 3, 1, 2)
        state_input = state_permuted.reshape(-1, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)
        state_input_simple_batch = state_input.unsqueeze(0)
        action_logits = self.model(state_input_simple_batch)
        action_probs = F.softmax(action_logits, dim = 1)
        action = action_probs.multinomial(num_samples=1).item() 
        return action
        

# ==========================================
#                  TRAINERS
# ==========================================

class BaseTrainer:
    """Базовый класс для всех методов обучения."""
    def __init__(self, policy: BasePolicy, config: RLConfig):
        self.policy = policy
        self.config = config
        
    def train_step(self, batch_of_trajectories) -> None:
        raise NotImplementedError

class DummyTrainer(BaseTrainer):
    """Пустой тренер для тестов (ничему не обучается)."""
    def train_step(self, batch_of_trajectories) -> None:
        pass

class ReinforceTrainer(BaseTrainer):
    def train_step(self, batch_of_trajectories) -> None:
        # TODO: Реализация REINFORCE (Vanilla Policy Gradient)
        # trajectory[t] = (state, action, reward)
        self.policy.optimizer.zero_grad()
        batch_loss = []

        for traj in batch_of_trajectories:
            states, actions, rewards = zip(*traj)
            
            states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32)
            states_tensor = states_tensor.permute(0, 1, 4, 2, 3).reshape(len(states), -1, 84, 84)
            
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            
            logits = self.policy.model(states_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            
            log_probs = distribution.log_prob(actions_tensor) 

            G = 0
            for t, r in enumerate(rewards):
                G += (self.config.GAMMA ** t) * r
            
            traj_loss = -G * log_probs.sum()
            batch_loss.append(traj_loss)

        total_loss = torch.stack(batch_loss).mean()
        
        total_loss.backward()
        self.policy.optimizer.step()
        

class ReinforceBaselineTrainer(BaseTrainer):
    def train_step(self, batch_of_trajectories) -> None:
        # TODO: Реализация REINFORCE с бейзлайном (Value функция)
        pass

class TrpoTrainer(BaseTrainer):
    def train_step(self, batch_of_trajectories) -> None:
        # TODO: Реализация Trust Region Policy Optimization (TRPO)
        pass


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
        
        # 1. Загрузка политики (Policy Structure)
        policy_class = self.POLICIES.get(self.config.POLICY_TYPE)
        if not policy_class:
            raise ValueError(f"Unknown POLICY_TYPE: {self.config.POLICY_TYPE}")
        self.policy = policy_class(config)
        
        # 2. Загрузка логики обучения (Train Algorithm Structure)
        trainer_class = self.TRAINERS.get(self.config.TRAIN_METHOD)
        if not trainer_class:
            raise ValueError(f"Unknown TRAIN_METHOD: {self.config.TRAIN_METHOD}")
        self.trainer = trainer_class(self.policy, config)

    def get_action(self, state: np.ndarray) -> int:
        """Передает стейт в выбранную политику и получает решение."""
        return self.policy.get_action(state)
        
    def train_step(self, batch_of_trajectories) -> None:
        """Передает собранные траектории в алгоритм обучения обновлять веса политики."""
        self.trainer.train_step(batch_of_trajectories)
            
    def save(self, filepath: str):
        self.policy.save(filepath)
        print(f"Agent weights saved to {filepath}")
        
    def load(self, filepath: str):
        self.policy.load(filepath)
        print(f"Agent weights loaded from {filepath}")
