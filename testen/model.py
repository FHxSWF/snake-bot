import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DQN_Trainer:
    def __init__(self, model, lr, gamma):
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
        # Konvertiere Listen zu Numpy-Arrays und dann zu Tensoren
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)  # Konvertiere actions zu Tensor
        dones = torch.tensor(np.array(dones), dtype=torch.float)

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