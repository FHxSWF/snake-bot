import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearNetwork(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_layer, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, output_layer)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.5)(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PPO:
    def __init__(self, model, lr, gamma, epsilon=0.2, beta=0.01):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta  # Entropy-Boost, um die Exploration zu fördern
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
    def train_step(self, state, action, reward, next_state, done, old_log_probs):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        # Berechne den Discounted Return (TD Target)
        td_target = reward + self.gamma * (1 - done) * self.model(next_state).max(1)[0]  # Max Q-Wert für nächsten Zustand
        
        # Berechne die Advantage (differenz zwischen erwarteten und realisierten Belohnungen)
        advantage = td_target - self.model(state).max(1)[0]
        
        # Alte Log-Wahrscheinlichkeiten speichern (während des Trainings)
        current_log_probs = self.log_prob(state, action)
        
        # PPO Clipping: Berechnung des Surrogat-Verlustes
        ratio = torch.exp(current_log_probs - old_log_probs)  # Verhältnis der neuen zu den alten Wahrscheinlichkeiten
        
        # Clipping
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        
        # Surrogate Loss (mit Clipping)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        # Berechnung der Entropie für Exploration
        entropy = -torch.mean(torch.exp(current_log_probs) * current_log_probs)
        
        # Gesamtverlust: Surrogat-Verlust + Entropie-Bestrafung
        total_loss = loss - self.beta * entropy
        
        # Optimierung
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def log_prob(self, state, action):
        # Berechne die Log-Wahrscheinlichkeit der Handlung basierend auf der Policy
        action_probs = F.softmax(self.model(state), dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        return dist.log_prob(action)

