import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from rl_config import RLConfig
from env import GameSimEnvironment
from agent import Agent

def moving_average(a, n=10):
    if len(a) == 0:
        return []
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    config = RLConfig()
    env = GameSimEnvironment(config)
    agent = Agent(config)
    
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(config.WEIGHTS_DIR, "weights.npy")
    json_path = os.path.join(config.WEIGHTS_DIR, "learning_data.json")
    
    episode_rewards = []
    episode_coins = []
    
    if config.LOAD_FROM_CHECKPOINT:
        agent.load(weights_path)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                episode_rewards = data.get("rewards", [])
                episode_coins = data.get("coins", [])
            print(f"Loaded existing learning curve with {len(episode_rewards)} episodes.")
    else:
        print("Starting training from scratch.")
    
    print("\nStarting Training Loop... Press Ctrl+C to abort training and save plots.")
    print("Controls (Browser must have focus):")
    print("  'a' : Toggle Auto Mode")
    print("  'r' : Force Reset Episode")
    print("  's' : Single Step (turns off Auto Mode)")
    
    episode_count = 0
    auto_mode = True # По умолчанию крутим обучение быстро
    max_episodes = config.MAX_EPISODES # Assuming MAX_EPISODES is defined in RLConfig
    time_horizon = config.TIME_HORIZON
    
    batch_of_trajectories = []

    try:
        while episode_count < max_episodes:
            state, info = env.reset()
            terminal_state = False
            total_reward = 0.0
            trajectory = []  # Список для сбора траектории всего эпизода
            
            while not terminal_state:
                # Проверяем, жив ли браузер (если пользователь закрыл окно)
                try:
                    if getattr(env, 'page', None) and env.page.is_closed():
                        print("\nBrowser closed by user. Stopping training...")
                        raise KeyboardInterrupt
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    # Если проверка упала - скорее всего вебсокет уже мертв
                    pass
                    
                # ── 1. Обновляем UI текст на экране ──
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
                
                # ── 2. Читаем клавиши, нажатые в браузере (через JS window) ──
                try:
                    key = env.page.evaluate("window.pythonRlAction;")
                    if key:
                        env.page.evaluate("window.pythonRlAction = null;") # сбросили флаг
                except Exception:
                    key = None
                    
                # ── 3. Обработка кнопок ──
                if key == 'r':
                    print(">> Manual Reset triggered")
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
                    print(f">> Auto Mode: {'ON' if auto_mode else 'OFF'}")
                elif key == 's':
                    auto_mode = False
                    
                # ── 4. Логика шага симуляции ──
                if auto_mode or key in ['s', 'c']:
                    if key == 'c':
                        action = 1
                    else:
                        action = agent.get_action(state)
                        
                    if auto_mode:
                        next_state, reward, terminal_state, truncated, info = env.step_auto(action)
                    else:
                        next_state, reward, terminal_state, truncated, info = env.step_manual(action)
                    
                    # Формируем опыт только из нужных элементов (без next_state и terminal_state)
                    experience = (state, action, reward)
                    trajectory.append(experience)
                    
                    state = next_state
                    total_reward += reward
                    terminal_state = terminal_state or truncated
                    
                    # Когда эпизод завершается, добавляем траекторию в батч
                    if terminal_state:
                        batch_of_trajectories.append(trajectory)
                        
                        # Если накопили достаточно траекторий, делаем шаг обучения
                        if len(batch_of_trajectories) >= config.TRAJECTORIES_PER_BATCH:
                            agent.train_step(batch_of_trajectories=batch_of_trajectories)
                            batch_of_trajectories = []
                else:
                    time.sleep(0.05)
            
            coins = info.get("coins", 0)
            episode_rewards.append(total_reward)
            episode_coins.append(coins)
            episode_count += 1
            
            print(f"Episode {len(episode_rewards)} finished. Total Reward: {total_reward:.3f}, Coins: {coins}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving data ...")
    except Exception as e:
        print(f"\nTraining stopped due to error: {e}. Saving data ...")
    finally:
        # Guarantee save
        agent.save(weights_path)
        
        with open(json_path, 'w') as f:
            json.dump({"rewards": episode_rewards, "coins": episode_coins}, f)
        print(f"Learning data saved to {json_path}")
        
        # Plotting
        if len(episode_rewards) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards, label="Reward per Episode", alpha=0.3)
            if len(episode_rewards) >= 10:
                ma_rewards = moving_average(episode_rewards, n=10)
                plt.plot(range(9, len(episode_rewards)), ma_rewards, label="Moving Average (10)", color='red')
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title("Learning Curve - Cumulative Reward")
            plt.legend()
            plt.savefig(os.path.join(config.WEIGHTS_DIR, "learning_curve_reward.png"))
            plt.close()
            
            plt.figure(figsize=(10, 5))
            plt.plot(episode_coins, label="DODO Coins per Episode", alpha=0.3, color='orange')
            if len(episode_coins) >= 10:
                ma_coins = moving_average(episode_coins, n=10)
                plt.plot(range(9, len(episode_coins)), ma_coins, label="Moving Average (10)", color='blue')
            plt.xlabel("Episode")
            plt.ylabel("DODO Coins")
            plt.title("Learning Curve - DODO Coins")
            plt.legend()
            plt.savefig(os.path.join(config.WEIGHTS_DIR, "learning_curve_dodo_coins.png"))
            plt.close()
            print("Plots saved to " + config.WEIGHTS_DIR)
            
        env.close()

if __name__ == "__main__":
    main()
