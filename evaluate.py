import os
import time
import numpy as np
from rl_config import RLConfig
from env import GameSimEnvironment
from agent import Agent

def main():
    config = RLConfig()
    
    # Принудительно ставим загрузку весов для evaluation скрипта
    config.LOAD_FROM_CHECKPOINT = True
    
    env = GameSimEnvironment(config)
    agent = Agent(config)
    
    weights_path = os.path.join(config.WEIGHTS_DIR, "weights.npy")
    agent.load(weights_path)
    
    num_episodes = 5
    print(f"\nStarting Evaluation for {num_episodes} episodes...")
    
    try:
        total_rewards = []
        total_coins = []
        auto_mode = True
        
        for ep in range(1, num_episodes + 1):
            state, info = env.reset()
            terminal_state = False
            ep_reward = 0.0
            
            while not terminal_state:
                # Проверяем, перехватываем ли мы ручное закрытие окна
                try:
                    if getattr(env, 'page', None) and env.page.is_closed():
                        raise KeyboardInterrupt("Browser closed by user")
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    pass
                    
                # ── UI Отрисовка ──
                prob = agent.config.CLICK_PROBABILITY * 100
                ui_script = f"""try {{
                    const el = document.getElementById('policy-info');
                    if (el) {{
                        el.classList.remove('hidden');
                        const newText = 'P(Click): {prob:.1f}%';
                        if (el.innerText !== newText) el.innerText = newText;
                    }}
                }} catch(e) {{}}"""
                try:
                    env.page.evaluate(ui_script)
                except Exception:
                    pass
                    
                # ── Чтение кнопок ──
                try:
                    key = env.page.evaluate("window.pythonRlAction;")
                    if key:
                        env.page.evaluate("window.pythonRlAction = null;")
                except Exception:
                    key = None
                    
                if key == 'r':
                    state, info = env.reset()
                    continue
                elif key == 'a':
                    auto_mode = not auto_mode
                    if auto_mode:
                        try:
                            env.page.evaluate("window.rlResumeAuto && window.rlResumeAuto();")
                        except Exception:
                            pass
                    else:
                        try:
                            env.page.evaluate("window.rlPhysicsFrozen = true;")
                        except Exception:
                            pass
                elif key == 's':
                    auto_mode = False
                    
                # ── Шаги ──
                if auto_mode or key in ['s', 'c']:
                    if key == 'c':
                        action = 1
                    else:
                        action = agent.get_action(state)
                        
                    if auto_mode:
                        state, reward, terminal_state, truncated, info = env.step_auto(action)
                    else:
                        state, reward, terminal_state, truncated, info = env.step_manual(action)
                        
                    ep_reward += reward
                    terminal_state = terminal_state or truncated
                else:
                    time.sleep(0.05)
                
            coins = info.get("coins", 0)
            print(f"Eval Episode {ep} - Reward: {ep_reward:.3f}, Coins: {coins}")
            total_rewards.append(ep_reward)
            total_coins.append(coins)
            
        print("\n--- Evaluation Results ---")
        print(f"Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"Average Coins:  {np.mean(total_coins):.1f} ± {np.std(total_coins):.1f}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
