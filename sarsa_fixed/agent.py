from collections import deque

import numpy as np

import environment_AI
from environment_AI import *
from model import *
from helper import plot
from environment_AI import Direction, Point
import random
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MOVES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialisiert den Agenten.
        """
        self.criterion = nn.MSELoss()
        self.gamma = 0.9
        self.n_games = 0
        self.epsilon = 80  # Erhöht für mehr Exploration
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = SARSA_Trainer(self.model, learning_rate, gamma=self.gamma)
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, game):
        head = game.head_pos

        point_l = Point(head.x - environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_r = Point(head.x + environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - environment_AI.SNAKE_BLOCK_SIZE)
        point_d = Point(head.x, head.y + environment_AI.SNAKE_BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            game.is_collision(point_l),
            game.is_collision(point_r),
            game.is_collision(point_u),
            game.is_collision(point_d),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head_pos.x,
            game.food.x > game.head_pos.x,
            game.food.y < game.head_pos.y,
            game.food.y > game.head_pos.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, action_new, done):
        self.memory.append((state, action, reward, next_state, action_new, done))
        print(f"Remember: State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

    def get_action(self, state):
        """ Entscheidet die nächste Bewegung """
        self.epsilon = max(1, 80 - self.n_games * 0.5)  # Langsamer Abfall für bessere Exploration
        final_move = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            final_move = random.choice(MOVES)
            print(f"Random Move: {final_move}")
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            print(f"Model Prediction: {prediction.tolist()}, Chosen Move: {move}")

        return final_move

    def train_short_memory(self, state, action, reward, next_state, action_new, done):
        self.trainer.train_step(state, action, reward, next_state, action_new, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, action_new, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, action_new, dones)


def train():
    agent = Agent(12, 32, 4)
    game = SnakeEnvironment()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        action_new = agent.get_action(state_new) if not done else [0, 0, 0, 0]

        agent.train_short_memory(state_old, final_move, reward, state_new, action_new, done)
        agent.remember(state_old, final_move, reward, state_new, action_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
