class RLConfig:
    # ── Environment & Browser ──
    RANDOM_SEED = 42            # Задает глобальный сид для генерации блоков, скоростей и тд. Если None - случайны
    GAME_URL = "http://localhost:8080"
    HEADLESS = False            # True = браузер работает в фоне без окна, False = окно видно на экране
    BROWSER_WIDTH = 800         # Ширина окна браузера
    BROWSER_HEIGHT = 800        # Высота окна браузера
    
    # ── State Representation ──
    # Агент видит последние 5 последовательных кадров
    FRAMES_STACK = 5
    # Разрешение, до которого даунсэмплим скриншоты для нейросети
    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    CHANNELS = 3  # RGB (или 1 для Grayscale)
    
    # ── Actions ──
    # Дискретное пространство действий: 
    # 0 = ничего не делать (ждать)
    # 1 = клик (поставить коробку)
    ACTION_SPACE_SIZE = 2
    MAX_EPISODES = 1000000      # Максимальное количество эпизодов для обучения
    TIME_HORIZON = 1000000000      # Максимальное количество шагов в эпизоде
    SIMULATION_STEP_DT = 0.005   # Сколько "физического" времени движка проходит за 1 шаг агента
    
    # ── Agent Architecture ──
    # Выбор политики: "dummy" или "cnn"
    POLICY_TYPE = "cnn" #"dummy"
    
    # Выбор метода обучения: "dummy", "reinforce", "reinforce_baseline", "trpo"
    TRAIN_METHOD = "dummy"
    
    # ── Dummy Policy ──
    # Вероятность нажать на экран (клик) — 1%
    CLICK_PROBABILITY = 0.01 
    
    # ── RL Hyperparameters (Placeholder) ──
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    TRAJECTORIES_PER_BATCH = 10
    
    # ── Saving & Loading ──
    LOAD_FROM_CHECKPOINT = True
    WEIGHTS_DIR = "weights_dir"
    
    # ── Rewards ──
    REWARD_PER_FRAME = 0.001
    
    # ── Debugging ──
    SAVE_DEBUG_FRAMES = True
    DEBUG_DIR = "debug_frames"
