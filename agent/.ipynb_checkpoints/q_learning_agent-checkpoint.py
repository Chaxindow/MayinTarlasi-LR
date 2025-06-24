import numpy as np
import random
import pickle
import os
from datetime import datetime

class QLearningAgent:
    def __init__(self, state_shape, action_space, learning_rate=0.1, discount=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Q-Learning ajanı başlatılır.

        Parameters:
        - state_shape: Ortamın gözlem (tahta) boyutu (örneğin (5, 5))
        - action_space: Ajanın seçebileceği tüm (x, y) aksiyonları
        - learning_rate: Q-value güncellenme oranı (α)
        - discount: Gelecekteki ödüllerin indirim katsayısı (γ)
        - epsilon: Başlangıç keşif oranı (ε)
        - epsilon_decay: Her bölüm sonunda epsilon'un azaltılma oranı
        - epsilon_min: Epsilon'un düşebileceği minimum seviye
        """
        self.state_shape = state_shape
        self.action_space = action_space 
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-tablosu: anahtar (state_tuplu, action_tuplu), değer q_value
        self.q_table = {}

    def get_state_hash(self,state):
        """
        Verilen durumu (numpy array) hashlenebilir bir formata dönüştürür.
        Böylece q_table için anahtar olarak kullanılabilir.
        """
        return tuple(state.flatten())

    def choose_action(self,state):
        """
        Epsilon-greedy stratejisiyle eylem seçer.
        Keşif (explore) ya da en iyi bilinen eylemi (exploit) uygular.
        """
        state_hash = self.get_state_hash(state)

        # Rastgele seçim (keşif)
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)

        # Bilinen en iyi eylemi seç (sömürü)
        q_values = [self.q_table.get((state_hash, a), 0) for a in self.action_space]
        max_q = max(q_values)

        # En yüksek Q'ya sahip tüm eylemleri bul (eşit olabilirler)
        best_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
        return random.choice(best_actions)


    def learn(self, state, action, reward, next_state, done):
        """
        Ajanı öğrenmeye zorlar. Q-tablosunu günceller.

        state: Eski durum
        action: Yapılan eylem
        reward: Alınan ödül
        next_state: Yeni durum
        done: Oyun bitti mi?
        """
        state_hash = self.get_state_hash(state)
        next_state_hash = self.get_state_hash(next_state)

        # Mevcut Q değeri
        old_q = self.q_table.get((state_hash, action), 0)

        # Gelecek durumun maksimum Q değeri
        next_qs = [self.q_table.get((next_state_hash, a), 0) for a in self.action_space]
        next_max = max(next_qs) if not done else 0

        # Q-learning güncellemesi
        new_q = old_q + self.lr * (reward + self.gamma * next_max - old_q)
        self.q_table[(state_hash, action)] = new_q

        # Her bölüm sonunda epsilon'u azalt
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        


    def save(self, path_dir='models/q_learning'):
        """
        Q tablosunu zaman damgalı bir dosya adıyla belirtilen klasöre kaydeder.

        Parameters:
        - path_dir (str): Q-tablosunun kaydedileceği klasör yolu.
        """
        # Klasör yoksa oluştur
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # Zaman damgası oluştur
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Dosya adını belirle
        filename = f"q_table_{timestamp}.pkl"
        full_path = os.path.join(path_dir, filename)

        # Q-tablosunu dosyaya yaz
        with open(full_path, 'wb') as f:
            pickle.dump(self.q_table, f)

        print(f"✅ Q tablosu kaydedildi: {full_path}")

        

    def load(self, path='models/q_learning'):
        """
        Daha önce kaydedilmiş bir Q-tablosunu dosyadan yükler.

        Parameters:
        - path (str): Eğer dosya yolu verilirse o dosya yüklenir.
                      Eğer klasör verilirse, içindeki en son dosya yüklenir.
        """
        if os.path.isdir(path):
            # Eğer klasör verildiyse, içindeki en son (tarihe göre) dosyayı bul
            files = [f for f in os.listdir(path) if f.startswith('q_table') and f.endswith('.pkl')]
            if not files:
                raise FileNotFoundError(f"❌ Klasörde hiç Q-tablosu bulunamadı: {path}")

            # Tarih sırasına göre tersten sırala (en son en üstte olur)
            files.sort(reverse=True)
            latest_file = os.path.join(path, files[0])
        else:
            # Eğer doğrudan dosya verildiyse onu kullan
            latest_file = path

        # Q-tablosunu dosyadan yükle
        with open(latest_file, 'rb') as f:
            self.q_table = pickle.load(f)

        print(f"✅ Q tablosu yüklendi: {latest_file}")

        
        
        
        
    