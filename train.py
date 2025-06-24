from env.minesweeper_env import MinesweeperEnv
from agent.q_learning_agent import QLearningAgent
import numpy as np
import os

def create_action_space(width, height):
    """
    Tüm (x, y) pozisyonlarını içeren aksiyon listesi üretir.
    """
    return [(x, y) for y in range(height) for x in range(width)]

def train_q_learning_agent(
        episodes=50000,
        width=4,
        height=4,
        n_mines=3,
        log_interval=100
    ):
    """
    Q-learning ajanını belirli sayıda bölüm (episode) boyunca Minesweeper ortamında eğitir.
    Eğitim boyunca ödül, adım ve hamleler terminale loglanır.

    Parameters:
    - episodes: Kaç oyun oynanacağı
    - width, height: Oyun tahtası boyutu
    - n_mines: Mayın sayısı
    - log_interval: Kaç bölümde bir detaylı log ve render yapılacağı
    """
    # Ortam ve aksiyon listesi oluştur
    env = MinesweeperEnv(width, height, n_mines)
    action_space = create_action_space(width, height)

    # Agent oluştur
    agent = QLearningAgent(
        state_shape=(height, width),
        action_space=action_space,
        learning_rate=0.1,
        discount=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    win_count = 0

    # Eğitim döngüsü
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n🎯 Episode {episode} başlıyor... [Epsilon: {agent.epsilon:.3f}]")

        while not done and steps < 100:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Log: ilk birkaç bölümde ve log_interval ile belirlenen aralıklarla
            if episode <= 5 or episode % log_interval == 0:
                print(f"  ➤ Adım {steps:02}: Aksiyon={action}, Ödül={reward}, Mesaj='{info.get('msg', '')}'")
                env.render()
                
        if done and reward > 0:  # Ödül pozitifse genelde kazanmış demektir
            win_count += 1
                
        print(f"✅ Episode {episode} bitti. Toplam Ödül: {total_reward}, Adım Sayısı: {steps},  Toplam Kazanma Sayısı: {win_count}")

    # Eğitilen model kaydedilir (zaman damgalı olarak)
    agent.save("models/q_learning")

if __name__ == "__main__":
    train_q_learning_agent()
