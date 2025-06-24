import numpy as np
import random

# Oyun Sınıfım
class MinesweeperEnv :

    #Constructor
    def __init__(self, width=2, height=2, n_mines=1):
        self.width = width
        self.height = height
        self.n_mines = n_mines
        self.reset()

    #Oyunu başlatma
    def reset(self):
        self.done = False
        self.board = np.zeros((self.height,self.width), dtype=int) # Tahtayı hazırla ve hepsine 0'ı ver
        self.revealed = np.zeros((self.height,self.width), dtype=bool)
        self.flagged = np.zeros((self.height, self.width), dtype=bool)

        # Mayınları yerleştir
        mines_placed = 0
        while mines_placed < self.n_mines:
            y = random.randint(0, self.height-1)
            x = random.randint(0, self.width-1)
            if self.board[y,x] != -1:
                self.board[y,x] = -1
                mines_placed += 1

        # Mayın olmayan hücrelerin çevresindeki mayın sayılarını hesaplama
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y,x] == -1:
                    continue
                count = 0
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < self.height and 0<= nx < self.width:
                            if self.board[ny,nx] == -1: # Mayınlara bakıyor
                                count +=1
                self.board[y,x] = count

        return self.get_state()

    def get_state(self):
        # Burada anlık görüntü alınır açılmış veya kapalı hücreler işaretlenmesi açısından
        state = np.full((self.height, self.width), -1, dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if self.revealed[y,x]:
                    state[y,x] = self.board[y,x]
        return state

    def render(self):
        """
        Ortamı terminalde gösterir.
        Kapalı hücreler '#', işaretli hücreler 'F', açılmış hücreler sayı olarak gösterilir.
        """
        print("   " + " ".join([str(x) for x in range(self.width)]))
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.flagged[y, x]:
                    row.append("F")
                elif not self.revealed[y, x]:
                    row.append("#")
                else:
                    val = self.board[y, x]
                    if val == -1:
                        row.append("*")
                    elif val == 0:
                        row.append(" ")
                    else:
                        row.append(str(val))
            print(f"{y:2} " + " ".join(row))
        print()
        
    def step(self,action):
        """
        action: (x, y) tuple veya int index (örnek: (x,y))
        Hücre açma işlemini yapar.
        Returns:
          next_state, reward, done, info
        """
        if self.done:
            return self.get_state(), 0, True, {"msg": "Oyun zaten bitti."}

        x, y = action
        if self.revealed[y,x] or self.flagged[y,x]:
            return self.get_state(), -1, False, {"msg": "Geçersiz hamle: zaten açılmış veya işaretlenmiş."}

        if self.board[y,x] == -1:
            self.revealed[y,x] = True
            self.done = True
            reward = -10
            return self.get_state(), reward, True, {"msg": "Mayına bastın! Kaybettin."}

        self._flood_fill(y,x) # Hücreleri açıyor
        reward = 1 # Her hücre açışında 1 ödül

        self.total_safe_cells = self.width * self.height - self.n_mines
        self.opened_cells = np.sum(np.logical_and(self.revealed, ~self.flagged))
        if self.opened_cells == self.total_safe_cells:
            self.done = True
            reward +=50
            return self.get_state(), reward, True, {"msg": "Tebrikler! Oyunu kazandın."}

        return self.get_state(), reward, False, {}

    def _flood_fill(self,y,x):
        """
        Açılan hücre 0 ise diğer hücrelet otomatik açılacak
        """
        if not (0 <= y < self.height and 0 <= x < self.width):
            return
        if self.revealed[y,x] or self.flagged[y,x]:
            return
        self.revealed[y,x] = True
        if self.board[y,x] !=0:
            return
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                ny, nx =y+dy, x+dx
                if dy == 0 and dx == 0: # kendisi olmuş oluyor
                    continue
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self._flood_fill(ny,nx)
                
                    
    def flag(self,action):
        """
        Hücreye bayrak koyup kaldırır
        """
        x, y = action
        if self.done:
            return {"msg" : "Game Over"}
        if self.revealed[y,x]:
            return {"msg" : "Acılmış hücreye bayrak konulmaz"}
        self.flagged[y,x] = not self.flagged[y,x]
        return {"msg" : "Bayrak durumu değiştirildi"}
        
        
                            
                    
    

    
        
    