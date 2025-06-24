from env.minesweeper_env import MinesweeperEnv
from agent.q_learning_agent import QLearningAgent

def create_action_space(width, height):
    return [(x, y) for y in range(height) for x in range(width)]

def play_trained_agent(width=4, height=4, n_mines=3):
    env = MinesweeperEnv(width, height, n_mines)
    action_space = create_action_space(width, height)

    agent = QLearningAgent(state_shape=(height, width), action_space=action_space)
    agent.load("models/q_learning/")  # En son kaydı yükle
    agent.epsilon = 0.05  # Neredeyse tamamen öğrendiklerini uygula (çok az rastgelelik)

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("🎮 Eğitilmiş ajan oyun oynuyor...")

    while not done and steps < 100:
        env.render()

        action = agent.choose_action(state)
        print(f"  ➤ Ajan hamlesi: {action}")

        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

    env.render()
    print(f"🏁 Oyun bitti. Toplam ödül: {total_reward}, Toplam adım: {steps}")
    print(f"Son mesaj: {info.get('msg', '')}")

if __name__ == "__main__":
    play_trained_agent()
