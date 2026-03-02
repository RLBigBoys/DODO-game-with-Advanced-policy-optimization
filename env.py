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
        self.t = 0
        
        # Define action and observation spaces (Gymnasium)
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        # State: (FRAMES, HEIGHT, WIDTH, CHANNELS)
        obs_shape = (config.FRAMES_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH, config.CHANNELS)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
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
            is_start_screen = self.page.locator("#tap-to-start").is_visible()
            
            if is_game_over:
                # Soft restart without full page reload (prevents flicker)
                self.page.locator("#btn-restart").click(force=True)
                time.sleep(0.3)
                # Always click on "Tap to Start" to actually start the game
                self.page.locator("#tap-to-start").click(force=True)
                time.sleep(0.3)
            elif is_start_screen:
                # First launch
                self.page.locator("#tap-to-start").click(force=True)
                time.sleep(0.3)
            else:
                # If stuck in the middle of a game (e.g., step limit) - hard reset
                self.page.reload()
                time.sleep(1)
                if self.page.locator("#tap-to-start").is_visible():
                    self.page.locator("#tap-to-start").click(force=True)
                    time.sleep(0.3)
        except Exception as e:
            print(f"Reset warning: {e}")
            self.page.reload()
            time.sleep(1)
            try:
                if self.page.locator("#tap-to-start").is_visible():
                    self.page.locator("#tap-to-start").click(force=True)
                    time.sleep(0.3)
            except Exception:
                pass
            
        # Unfreeze physics if it was frozen by a previous episode in Step Mode
        try:
            self.page.evaluate("window.rlResumeAuto && window.rlResumeAuto();")
        except Exception:
            pass
        
        self.frame_buffer.clear()
        for _ in range(self.config.FRAMES_STACK):
            self.frame_buffer.append(self._capture_screenshot())
        
        state = self._get_observation()
        info = {}
        
        return state, info

    def _get_observation(self) -> np.ndarray:
        """Stack last frames from buffer into a single 4D agent state tensor."""
        state = np.stack(self.frame_buffer)
        self._save_debug_frames(state)
        return state
        
    def _get_episode_status(self, action: int = 0):
        """Parse game state (Game Over flag) and coin count from the DOM."""
        try:
            terminal_state = self.page.locator("#gameover-overlay").is_visible()
            coins_text = self.page.locator('#coin-text').inner_text()
            coins = int(coins_text.split(' / ')[0].strip())
        except Exception:
            terminal_state = False
            coins = 0

        # Base survival reward (per frame)
        reward = self.config.REWARD_PER_FRAME
        
        # Extra reward for taking action "click"
        if action == 1:
            reward += self.config.REWARD_PER_CLICK
            
        truncated = self.t >= self.config.TIME_HORIZON
        info = {"coins": coins}
        return reward, terminal_state, truncated, info

    def step_auto(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Auto Mode: wait for the next real frame from the 60 FPS game loop."""
        self.t += 1
        
        if action == 1:
            print(f"  [{self.t}] ➡️ Action chosen: CLICK (Drop block)")
            try:
                self.page.evaluate("window.executeDropBlock();")
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
        
        reward, terminal_state, truncated, info = self._get_episode_status(action)
        state = self._get_observation()
        
        return state, reward, terminal_state, truncated, info

    def step_manual(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step Mode: physics is frozen and is advanced manually by dt."""
        self.t += 1
        
        if action == 1:
            print(f"  [{self.t}] ➡️ Action chosen: CLICK (Drop block)")
            try:
                self.page.evaluate("window.executeDropBlock();")
            except Exception:
                pass
                
        # Advance physics synchronously by fixed dt (frozen mode)
        try:
            b64_str = self.page.evaluate(f"window.rlManualStep({self.config.SIMULATION_STEP_DT});")
        except Exception:
            b64_str = None
            
        next_frame = self._capture_screenshot(pre_fetched_b64=b64_str)
        self.frame_buffer.append(next_frame)
        
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
        try:
            if hasattr(self, 'browser') and self.browser:
                self.browser.close()
        except Exception:
            pass
            
        try:
            if hasattr(self, 'playwright') and self.playwright:
                self.playwright.stop()
        except Exception:
            pass

def main():
    print("This module defines the environment. Use train.py or evaluate.py instead.")

if __name__ == "__main__":
    main()
