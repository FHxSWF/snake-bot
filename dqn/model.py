import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Ein einfaches neuronales Netzwerk für Deep Q-Learning mit zwei linearen Schichten.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialisiert das neuronale Netzwerk mit zwei linearen Schichten.
        :param input_size: Anzahl der Eingabe-Features.
        :param hidden_size: Anzahl der Neuronen im versteckten Layer.
        :param output_size: Anzahl der Ausgabewerte (Anzahl der möglichen Aktionen).
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Führt eine Vorwärtspropagation durch.
        :param x: Eingabetensor
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='modelDQNthird.pth'):
        """
        Speichert das trainierte Modell als Datei.
        :param file_name: Name der zu speichernden Datei.
        :return:
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DQN_Trainer:
    """
    Trainer-Klasse für das Deep Q-Learning mit einem neuronalen Netzwerk.
    """
    def __init__(self, model, lr, gamma):
        """
        Initialisiert den Trainer mit einem Q-Netzwerk und einem Zielnetzwerk.

        Args:
            model (Linear_QNet): Das neuronale Netzwerk für das Training.
            lr (float): Lernrate für den Optimizer.
            gamma (float): Discount-Faktor für zukünftige Belohnungen.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = Linear_QNet(model.linear1.in_features, model.linear1.out_features, model.linear2.out_features)
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.update_target_every = 100  # Target Network Update-Intervall
        self.update_counter = 0

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Führt einen einzelnen Trainingsschritt für das DQN-Modell aus.
        :param states: Liste der aktuellen Zustände.
        :param actions: Liste der ausgeführten Aktionen.
        :param rewards: Liste der erhaltenen Belohnungen.
        :param next_states: Liste der nächsten Zustände.
        :param dones: Liste der Terminal-Flags (1, falls Episode beendet, sonst 0).
        """
        # Konvertiere Listen zu Numpy-Arrays und dann zu Tensoren
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)  # Konvertiere actions zu Tensor
        dones = torch.tensor(np.array(dones), dtype=torch.float)

        actions = actions.view(-1, 4)

        # Überprüfe die Form der Tensoren
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # Füge eine Batch-Dimension hinzu
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)  # Füge eine Batch-Dimension hinzu

        # Berechne Q-Zielwerte mit Target Network
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Berechne Q-Vorhersagen
        q_pred = self.model(states).gather(1, torch.argmax(actions, dim=1).unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # Aktualisiere Target Network in regelmäßigen Abständen
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
