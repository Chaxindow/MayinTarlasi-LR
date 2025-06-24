from env.minesweeper_env import MinesweeperEnv
from agent.q_learning_agent import QLearningAgent
import numpy as np

def create_action_space(width, height):
    return [(x, y) for y in range(height) for x in range(width)]

def test_agent_on_minesweeper():
    # Ortam parametreleri
    width, height, n_mines = 2, 2, 1
    env = MinesweeperEnv(width=width, height=height, n_mines=n_mines)
    action_space = create_action_space(width, height)

    # Agent oluştur
    agent = QLearningAgent(
        state_shape=(height, width),
        action_space=action_space,
        learning_rate=0.1,
        discount=0.99,
        epsilon=1.0
    )

    # Bir oyun turu oynat
    state = env.reset()
    env.render()

    done = False
    total_reward = 0
    step = 0

    while not done and step < 20:
        action = agent.choose_action(state)
        print(f"\nAdım {step+1} - Ajanın seçtiği hamle: {action}")

        next_state, reward, done, info = env.step(action)
        print(f"Ödül: {reward}, Oyun bitti mi? {done}, Bilgi: {info.get('msg', '')}")

        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step += 1
        env.render()

    print(f"\nToplam ödül: {total_reward}")
    print(f"Final epsilon: {agent.epsilon}")
    print(f"Q tablo boyutu: {len(agent.q_table)}")

if __name__ == "__main__":
    test_agent_on_minesweeper()
