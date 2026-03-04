import time
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
from rl_config import RLConfig
from playwright.sync_api import sync_playwright

class GameSimEnvironment(gym.Env):
    """
    Environment: connects to the browser game via Playwright.
    Conforms to the Gymnasium API.
    """
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        self.frame_buffer = deque(maxlen=config.FRAMES_STACK)
        self.action_buffer = deque(maxlen=config.FRAMES_STACK)
        self.t = 0
        
        # Define action and observation spaces (Gymnasium)
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        # State: contains frames and previous actions
        obs_shape = (config.FRAMES_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH, config.CHANNELS)
        self.observation_space = gym.spaces.Dict({
            "frames": gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
            "previous_actions": gym.spaces.Box(low=0, high=config.ACTION_SPACE_SIZE-1, shape=(config.FRAMES_STACK,), dtype=np.float32)
        })
        
        # Start Playwright
        print("Starting Playwright to launch the game...")
        self.playwright = sync_playwright().start()
        # Launch Chromium with real window size and without mobile viewport emulation
        self.browser = self.playwright.chromium.launch(
            headless=config.HEADLESS,
            args=[f'--window-size={config.BROWSER_WIDTH},{config.BROWSER_HEIGHT}']
        )
        self.context = self.browser.new_context(no_viewport=True)
        self.page = self.context.new_page()
        
        # Navigate to the game
        self.page.goto(config.GAME_URL)
        print("Waiting for game to load...")
        time.sleep(2)  # Wait for page load
        
        self.current_episode_seed = getattr(self.config, 'RANDOM_SEED', None)
        if self.current_episode_seed is not None:
            self.page.evaluate(f"window.setRLSeed && window.setRLSeed({self.current_episode_seed});")
        
        # Enable exclusive RL mode: browser no longer auto-updates itself
        try:
            self.page.evaluate("window.rlMode = true;")
        except Exception:
            pass
        
    def seed(self, seed=None):
        """Seed generator for reproducability."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
        pass

    def _capture_screenshot(self, pre_fetched_b64=None) -> np.ndarray:
        """Capture current game canvas screenshot, returns 84x84 frame."""
        import base64
        try:
            # Hardware-accelerated 84x84 screenshot via hidden JS canvas (< 1ms)
            b64_str = pre_fetched_b64 if pre_fetched_b64 else self.page.evaluate("window.getRlFrame()")
            if not b64_str or "," not in b64_str:
                raise ValueError("Empty or invalid image data")
                
            b64_data = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            
            # Decode bytes into numpy array (image)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Resize image to configured size
            if img.shape[0] != self.config.FRAME_HEIGHT or img.shape[1] != self.config.FRAME_WIDTH:
                img = cv2.resize(img, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB if 3 channels, or to Grayscale (1 channel)
            if self.config.CHANNELS == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # If grayscale 84x84, add explicit channel dimension (84, 84, 1)
            if self.config.CHANNELS == 1:
                img = np.expand_dims(img, axis=-1)
                
            return img
            
        except Exception as e:
            # Fallback if connection closed mid-shot or script evaluates before JS load
            print(f"Screenshot Error: {e}")
            shape = (84, 84, self.config.CHANNELS)
            return np.zeros(shape, dtype=np.uint8)
        
    def _save_debug_frames(self, obs: np.ndarray):
        """Save 5 frames from the current state into the debug directory."""
        if getattr(self.config, 'SAVE_DEBUG_FRAMES', False):
            import os
            os.makedirs(self.config.DEBUG_DIR, exist_ok=True)
            for i in range(self.config.FRAMES_STACK):
                frame = obs[i]
                # Convert color back to BGR for correct saving in OpenCV
                if self.config.CHANNELS == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                filename = os.path.join(self.config.DEBUG_DIR, f"frame_{i}.png")
                cv2.imwrite(filename, frame)
                                 
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset: refresh page if needed, click to start, collect first frames."""
        super().reset(seed=seed)  # Initializes self.np_random
        self.t = 0
        
        # Update seed for the new episode
        if self.current_episode_seed is not None:
            self.current_episode_seed += 1
            try:
                self.page.evaluate(f"window.setRLSeed && window.setRLSeed({self.current_episode_seed});")
            except Exception:
                pass
        
        try:
            is_game_over = self.page.locator("#gameover-overlay").is_visible()
            
            if is_game_over:
                # Soft restart without full page reload (prevents flicker)
                self.page.locator("#btn-restart").click(force=True)
                
            # Wait for Tap to Start to become visible (module might still be loading/transitioning)
            self.page.wait_for_selector("#tap-to-start", state="visible", timeout=5000)
            
            # Click it until it disappears (which means the JS listener successfully caught it)
            start_wait = time.time()
            while self.page.locator("#tap-to-start").is_visible() and (time.time() - start_wait < 5.0):
                self.page.locator("#tap-to-start").click(force=True)
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Reset warning: {e}")
            self.page.reload()
            try:
                self.page.wait_for_selector("#tap-to-start", state="visible", timeout=10000)
                start_wait = time.time()
                while self.page.locator("#tap-to-start").is_visible() and (time.time() - start_wait < 5.0):
                    self.page.locator("#tap-to-start").click(force=True)
                    time.sleep(0.1)
            except Exception:
                pass
            
        # Unfreeze physics if it was frozen by a previous episode in Step Mode
        try:
            self.page.evaluate("window.rlResumeAuto && window.rlResumeAuto();")
        except Exception:
            pass
        
        self.frame_buffer.clear()
        self.action_buffer.clear()
        for _ in range(self.config.FRAMES_STACK):
            self.action_buffer.append(0)  # default no-click for pre-history
            
        for _ in range(self.config.FRAMES_STACK):
            self.frame_buffer.append(self._capture_screenshot())
        
        state = self._get_observation()
        info = {}
        
        return state, info

    def _get_observation(self) -> dict:
        """Return dict with stacked frames and previous actions."""
        frames = np.stack(self.frame_buffer)
        prev_actions = np.array(self.action_buffer, dtype=np.float32)
        state = {
            "frames": frames,
            "previous_actions": prev_actions
        }
        self._save_debug_frames(frames)
        return state
        
    def _get_episode_status(self, drop_info: dict = None):
        """Parse game state (Game Over flag) and coin count from the DOM."""
        if drop_info is None:
            drop_info = {"dropped": False, "success": False, "areaRatio": 0.0, "perfect": False}
        try:
            terminal_state = self.page.locator("#gameover-overlay").is_visible()
            coins_text = self.page.locator('#coin-text').inner_text()
            coins = int(coins_text.split(' / ')[0].strip())
        except Exception:
            terminal_state = False
            coins = 0

        # Base survival reward (per frame)
        reward = self.config.REWARD_PER_FRAME
        
        # Extra reward ONLY if the block successfully landed on the tower
        if drop_info and drop_info.get("success", False):
            ratio = drop_info.get("areaRatio", 0.0)
            reward += self.config.REWARD_PLACEMENT_MULTIPLIER * ratio
            reward += self.config.REWARD_COINS_MULTIPLIER * coins
            
        if terminal_state:
            reward += getattr(self.config, 'PENALTY_GAME_OVER', 0.0)
            
        truncated = self.t >= self.config.TIME_HORIZON
        info = {"coins": coins}
        return reward, terminal_state, truncated, info

    def step_auto(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Auto Mode: wait for the next real frame from the 60 FPS game loop."""
        self.t += 1
        
        drop_info = {"dropped": False, "success": False, "areaRatio": 0.0, "perfect": False}
        if action == 1:
            print(f"  [{self.t}] ➡️ Action chosen: CLICK (Drop block)")
            try:
                res = self.page.evaluate("window.executeDropBlock();")
                if isinstance(res, dict):
                    drop_info = res
            except Exception:
                pass
                
        # Wait until browser renders exactly 1 frame and then extract it
        try:
            self.page.evaluate("await window.waitForNextFrame();")
            b64_str = self.page.evaluate("window.getRlFrame();")
        except Exception:
            b64_str = None
            
        next_frame = self._capture_screenshot(pre_fetched_b64=b64_str)
        self.frame_buffer.append(next_frame)
        self.action_buffer.append(action)
        
        reward, terminal_state, truncated, info = self._get_episode_status(drop_info)
        state = self._get_observation()
        
        return state, reward, terminal_state, truncated, info

    def step_manual(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step Mode: physics is frozen and is advanced manually by dt."""
        self.t += 1
        
        drop_info = {"dropped": False, "success": False, "areaRatio": 0.0, "perfect": False}
        if action == 1:
            print(f"  [{self.t}] ➡️ Action chosen: CLICK (Drop block)")
            try:
                res = self.page.evaluate("window.executeDropBlock();")
                if isinstance(res, dict):
                    drop_info = res
            except Exception:
                pass
                
        # Advance physics synchronously by fixed dt (frozen mode)
        try:
            b64_str = self.page.evaluate(f"window.rlManualStep({self.config.SIMULATION_STEP_DT});")
        except Exception:
            b64_str = None
            
        next_frame = self._capture_screenshot(pre_fetched_b64=b64_str)
        self.frame_buffer.append(next_frame)
        self.action_buffer.append(action)
        
        reward, terminal_state, truncated, info = self._get_episode_status(action)
        state = self._get_observation()
        
        return state, reward, terminal_state, truncated, info
        
    def auto_mode(self, agent):
        """
        Simulation runs on its own, querying actions from the agent.
        Ideal for simple demos without an external training loop.
        """
        print("\n--- Starting AUTO MODE ---")
        state, info = self.reset()
        done = False
        
        try:
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminal_state, truncated, info = self.step(action)
                
                done = terminal_state or truncated
                state = next_state
                
                # Delay to roughly match real-time FPS
                time.sleep(0.01)
                
                if getattr(self, "t", 0) % 10 == 0:
                    print(f"Auto Mode Step {self.t}: running...")
                    
        except KeyboardInterrupt:
            print("\nAuto Mode Interrupted.")
        print("--- Finished AUTO MODE ---\n")
        
    def close(self):
        """Close browser and stop Playwright."""
        import contextlib
        import asyncio
        
        with contextlib.suppress(Exception):
            if hasattr(self, 'context') and self.context:
                self.context.close()
                
        with contextlib.suppress(Exception):
            if hasattr(self, 'browser') and self.browser:
                self.browser.close()
                
        with contextlib.suppress(Exception):
            if hasattr(self, 'playwright') and self.playwright:
                self.playwright.stop()
                
        # Hard terminate any dangling playwright async tasks
        with contextlib.suppress(Exception):
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                for pending_task in asyncio.all_tasks(loop):
                    pending_task.cancel()

def main():
    print("This module defines the environment. Use train.py or evaluate.py instead.")

if __name__ == "__main__":
    main()
