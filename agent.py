import numpy as np
import random
from rl_config import RLConfig

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
        # TODO: Инициализация нейронной сети (nn.Module)
        
    def get_action(self, state: np.ndarray) -> int:
        # TODO: Прогон состояния через CNN
        pass


# ==========================================
#                  TRAINERS
# ==========================================

class BaseTrainer:
    """Базовый класс для всех методов обучения."""
    def __init__(self, policy: BasePolicy, config: RLConfig):
        self.policy = policy
        self.config = config
        
    def train_step(self, trajectory) -> None:
        raise NotImplementedError

class DummyTrainer(BaseTrainer):
    """Пустой тренер для тестов (ничему не обучается)."""
    def train_step(self, trajectory) -> None:
        pass

class ReinforceTrainer(BaseTrainer):
    def train_step(self, trajectory) -> None:
        # TODO: Реализация REINFORCE (Vanilla Policy Gradient)
        pass

class ReinforceBaselineTrainer(BaseTrainer):
    def train_step(self, trajectory) -> None:
        # TODO: Реализация REINFORCE с бейзлайном (Value функция)
        pass

class TrpoTrainer(BaseTrainer):
    def train_step(self, trajectory) -> None:
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
        
    def train_step(self, trajectory) -> None:
        """Передает собранную траекторию в алгоритм обучения обновлять веса политики."""
        self.trainer.train_step(trajectory)
            
    def save(self, filepath: str):
        self.policy.save(filepath)
        print(f"Agent weights saved to {filepath}")
        
    def load(self, filepath: str):
        self.policy.load(filepath)
        print(f"Agent weights loaded from {filepath}")
