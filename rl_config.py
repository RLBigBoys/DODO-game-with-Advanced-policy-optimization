class RLConfig:
    # ── Environment & Browser ──
    RANDOM_SEED = 42            # Global seed for blocks, speeds, etc. If None - random each run
    GAME_URL = "http://localhost:8080"
    HEADLESS = False            # True = run browser headless, False = show window
    BROWSER_WIDTH = 800         # Browser window width
    BROWSER_HEIGHT = 800        # Browser window height
    
    # ── State Representation ──
    # Agent sees the last 5 consecutive frames
    FRAMES_STACK = 5
    # Resolution to which we downsample screenshots for the neural network
    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    CHANNELS = 3  # RGB (or 1 for Grayscale)
    
    # ── Actions ──
    # Discrete action space:
    # 0 = do nothing (wait)
    # 1 = click (drop a box)
    ACTION_SPACE_SIZE = 2
    MAX_EPISODES = 1000000      # Maximum number of training episodes
    TIME_HORIZON = 1000000000   # Maximum number of steps per episode
    SIMULATION_STEP_DT = 0.005  # Physical time (seconds) progressed per agent step in manual mode
    
    # ── Agent Architecture ──
    # Policy type: "dummy" or "cnn"
    POLICY_TYPE = "cnn"
    
    # Выбор метода обучения: "dummy", "reinforce", "reinforce_baseline", "trpo"
    TRAIN_METHOD = "reinforce_baseline"
    
    # ── Dummy Policy ──
    # Probability to click on the screen — 1%
    CLICK_PROBABILITY = 0.01 
    
    # ── RL Hyperparameters (Placeholder) ──
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    TRAJECTORIES_PER_BATCH = 10
    
    # ── Saving & Loading ──
    LOAD_FROM_CHECKPOINT = True
    LOAD_BEST_WEIGHTS = False    # True = try to load weights_best.npy instead of weights.npy
    WEIGHTS_DIR = "weights_dir"
    
    # ── Rewards ──
    REWARD_PER_FRAME = 0.0001
    REWARD_COINS_MULTIPLIER = 0.0
    REWARD_PLACEMENT_MULTIPLIER = 10.0
    
    # ── Debugging ──
    SAVE_DEBUG_FRAMES = True
    DEBUG_DIR = "debug_frames"
