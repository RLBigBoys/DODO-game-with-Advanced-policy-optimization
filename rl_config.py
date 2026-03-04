class RLConfig:
    # ── Environment & Browser ──
    RANDOM_SEED = 42            # Global seed for blocks, speeds, etc. If None - random each run
    GAME_URL = "http://localhost:8080"
    HEADLESS = False            # True = run browser headless, False = show window
    BROWSER_WIDTH = 800         # Browser window width
    BROWSER_HEIGHT = 800        # Browser window height
    
    # ── State Representation ──
    FRAMES_STACK = 5
    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    CHANNELS = 1  # RGB (or 1 for Grayscale)
    
    # ── Actions ──
    # Discrete action space:
    # 0 = do nothing (wait)
    # 1 = click (drop a box)
    ACTION_SPACE_SIZE = 2
    MAX_EPISODES = 10000000      # Maximum number of training episodes
    TIME_HORIZON = 1000000000   # Maximum number of steps per episode
    SIMULATION_STEP_DT = 0.005  # Physical time (seconds) progressed per agent step in manual mode
    
    # ── Agent Architecture ──
    # Policy type: "dummy", "cnn"
    POLICY_TYPE = "cnn"
    
    # Выбор метода обучения: "dummy", "reinforce", "reinforce_baseline", "trpo", "ppo"
    TRAIN_METHOD = "ppo"
    
    # ── Dummy Policy ──
    # Probability to click on the screen — 1%
    CLICK_PROBABILITY = 0.01 
    
    # ── RL Hyperparameters ──
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    TRAJECTORIES_PER_BATCH = 10

    # ── PPO Hyperparameters ──
    PPO_LR = 1e-4
    PPO_CLIP_EPS = 0.2
    PPO_EPOCHS = 4
    PPO_MINIBATCH_SIZE = 256
    PPO_ENTROPY_COEF = 0.01
    PPO_MAX_GRAD_NORM = 0.5
    ENTROPY_COEF = 0.01
    
    # ── Saving & Loading ──
    LOAD_FROM_CHECKPOINT = False
    LOAD_BEST_WEIGHTS = True    # True = try to load weights_best.npy instead of weights.npy
    WEIGHTS_DIR = f"weights_dir_3_{POLICY_TYPE}_{TRAIN_METHOD}"
    
    # ── Rewards ──
    REWARD_PER_FRAME = 0.0001
    REWARD_COINS_MULTIPLIER = 0.0
    REWARD_PLACEMENT_MULTIPLIER = 10.0
    PENALTY_GAME_OVER = -5.0
    
    # ── Debugging ──
    SAVE_DEBUG_FRAMES = False
    DEBUG_DIR = "debug_frames"
