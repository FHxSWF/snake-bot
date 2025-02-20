import torch
import torch.nn as nn
import torch.optim as optim
import random

class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialisiert den Agenten.
        :param input_size: Größe des Zustandsvektors.
        :param hidden_size: Anzahl der Neuronen im Hidden-Layer.
        :param output_size: Anzahl der möglichen Aktionen (hier 4 Mölg. LEFT, RIGHT, UP, DOWN).
        :param learning_rate:
        """
        self.model = nn.Sequential( # Platzhalter später in model.py
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # Platzhalter später in model.py
        self.criterion = nn.MSELoss() # Platzhalter später in model.py
        self.gamma = 0.9 # Discount-faktor für Rewards zukünftlich.

        # Mögliche Aktionen, die der Agent ausführen kann
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']

    def get_action(self, state, epsilon=0.1):
        """
        Wählt eine Aktion basierend auf dem aktuellen Zustand mittels einer epsilon-greedy Strategie.
        :param state:
        :param epsilon:
        :return:
        """
        # Mit Wahrscheinlichkeit epsilon eine zufällige Aktion (Exploration er soll Testen)
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            # Mit Wahrscheinlichkeit 1 - epsilon die Aktion mit dem höchsten Q-Wert wählen (Exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Batch-Dimension hinzufügen
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
            return self.actions[action_index]

    def get_direction_change(self, action, snake_block_size):
        """
        Übersetzt die gewählte Aktion in eine Richtungsänderung, die dann in der Spiel-Logik verwendet werden kann.
        :param action: was er tun soll.
        :param snake_block_size: Größe der Schlange, falls nötig.
        :return: Tuple mit den Änderungen in x- und y-Richtung.
        """
        if action == 'LEFT':
            return -snake_block_size, 0
        elif action == 'RIGHT':
            return snake_block_size, 0
        elif action == 'UP':
            return 0, -snake_block_size
        elif action == 'DOWN':
            return 0, snake_block_size
        else:
            return 0, 0  # Hier falls Fehlerfall

    def find_food_direction(self, snake_head, food_position, current_direction, snake_block_size):
        """
        Bestimmt eine Richtungsänderung, die die Schlange näher an das Futter bringt.
        Dabei wird versucht, die Differenz zwischen der Position des Schlangenkopfs
        und der des Futters zu minimieren, ohne in die entgegengesetzte Richtung zu laufen.

        :param snake_head: Tuple (x, y) der aktuellen Position des Schlangenkopfs.
        :param food_position: Tuple (x, y) der Position des Futters.
        :param current_direction: Aktuelle Richtung als String ('LEFT', 'RIGHT', 'UP', 'DOWN').
        :param snake_block_size: Größe eines Schlangensegments.
        :return: Tuple (x_change, y_change) der Bewegungsänderung, um in Richtung Futter zu gelangen.
        """
        # Differenz in x- und y-Richtung
        dx = food_position[0] - snake_head[0]
        dy = food_position[1] - snake_head[1]

        # Priorität: Bewege dich in die Richtung, in der die Distanz größer ist
        if abs(dx) > abs(dy):
            if dx < 0 and current_direction != 'RIGHT':
                return self.get_direction_change('LEFT', snake_block_size)
            elif dx > 0 and current_direction != 'LEFT':
                return self.get_direction_change('RIGHT', snake_block_size)
        else:
            if dy < 0 and current_direction != 'DOWN':
                return self.get_direction_change('UP', snake_block_size)
            elif dy > 0 and current_direction != 'UP':
                return self.get_direction_change('DOWN', snake_block_size)

        # Falls die bevorzugte Richtung nicht möglich ist (z.B. wegen Richtungsumkehr), beibehalten der aktuellen Richtung
        return self.get_direction_change(current_direction, snake_block_size)

    def train_step(self, state, action, reward, next_state, done):
        """
        Führt einen Trainingsschritt durch, sodass der Agent anhand des Rewards
        seinen Q-Wert aktualisiert.

        :param state: aktueller Zustand (Liste oder Array).
        :param action: getätigte Aktion (als String, z. B. 'LEFT').
        :param reward: erhaltene Belohnung (numerisch).
        :param next_state: nächster Zustand (Liste oder Array).
        :param done: Boolean, ob die Episode beendet ist.
        """
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float)

        # Aktueller Q-Wert-Vektor
        pred = self.model(state_tensor)
        target = pred.clone().detach()
        with torch.no_grad():
            next_pred = self.model(next_state_tensor)
        Q_new = reward_tensor
        if not done:
            Q_new = reward_tensor + self.gamma * torch.max(next_pred)

        action_index = self.actions.index(action)
        target[0][action_index] = Q_new

        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()