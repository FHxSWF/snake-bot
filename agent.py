import torch
import numpy as np
from environment_AI import Direction, Point

class SnakeAgent:
    def __init__(self, model, environment, ppo, max_memory=10000, batch_size=64):
        self.model = model
        self.environment = environment
        self.ppo = ppo
        self.memory = []
        self.max_memory = max_memory
        self.batch_size = batch_size

    def get_state(self):
        # Zustand als Feature-Vektor definieren
        head = self.environment.head_pos
        food = self.environment.food

        # Richtung der Schlange (one-hot encoding)
        direction = [0, 0, 0, 0]  # [left, right, up, down]
        if self.environment.direction == Direction.LEFT:
            direction[0] = 1
        elif self.environment.direction == Direction.RIGHT:
            direction[1] = 1
        elif self.environment.direction == Direction.UP:
            direction[2] = 1
        elif self.environment.direction == Direction.DOWN:
            direction[3] = 1

        # Gefahr in nächster Bewegung
        danger = [
            self.environment.is_collision(Point(head.x - 25, head.y)),  # Links
            self.environment.is_collision(Point(head.x + 25, head.y)),  # Rechts
            self.environment.is_collision(Point(head.x, head.y - 25)),  # Oben
            self.environment.is_collision(Point(head.x, head.y + 25)),  # Unten
        ]

        # Entfernung zum Apfel (normalisiert)
        apple_direction = [
            food.x < head.x,  # Apfel links
            food.x > head.x,  # Apfel rechts
            food.y < head.y,  # Apfel oben
            food.y > head.y   # Apfel unten
        ]

        state = np.array(direction + danger + apple_direction, dtype=int)
        return state

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        output = self.model(state)
        action_probs = torch.softmax(output, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def remember(self, state, action, reward, next_state, done, log_prob):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(self.memory, self.batch_size, replace=False)
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)

        self.ppo.train_step(states, actions, rewards, next_states, dones, log_probs)

    def train(self, episodes=1000):
        for episode in range(episodes):
            self.environment.reset()
            state = self.get_state()
            done = False
            total_reward = 0

            while not done:
                action, log_prob = self.select_action(state)
                game_over, score = self.environment.play_step(action)

                reward = 1 if self.environment.head_pos == self.environment.food else -1 if game_over else 0
                next_state = self.get_state()
                self.remember(state, action, reward, next_state, game_over, log_prob)

                state = next_state
                total_reward += reward

                if game_over:
                    break

            self.train_long_memory()
            print(f"Episode {episode+1}/{episodes}, Score: {score}, Reward: {total_reward}")


if __name__ == "__main__":
    from environment_AI import SnakeEnvironment
    from model import PPO, LinearNetwork

    model = LinearNetwork(12, 4)
    environment = SnakeEnvironment()
    ppo = PPO(model, lr=0.001, gamma=0.99)
    agent = SnakeAgent(model, environment, ppo)

    agent.train(episodes=1000)