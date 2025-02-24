from collections import deque

import numpy as np

import environment_AI
from environment_AI import *
from model import *
from helper import plot
from environment_AI import Direction, Point
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MOVES = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialisiert den Agenten.
        :param input_size: Größe des Zustandsvektors.
        :param hidden_size: Anzahl der Neuronen im Hidden-Layer.
        :param output_size: Anzahl der möglichen Aktionen (hier 4 Mölg. LEFT, RIGHT, UP, DOWN).
        :param learning_rate:
        """

        self.criterion = nn.MSELoss() # Platzhalter später in model.py
        self.gamma = 0.9 # Discount-faktor für Rewards zukünftlich.
        self.n_games = 0
        self.epsilon = 0
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = SARSA_Trainer(self.model, learning_rate, gamma=self.gamma)
        self.memory = deque(maxlen=MAX_MEMORY)


    def get_state(self, game):
        head = game.head_pos
        tail = game.snake_list[0]

        point_l = Point(head.x - environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_r = Point(head.x + environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - environment_AI.SNAKE_BLOCK_SIZE)
        point_d = Point(head.x, head.y + environment_AI.SNAKE_BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #Eventuell andere Werte wie Kopf o.ä. einsetzen
        state = [
            # Danger up
            (dir_u and game.is_collision(point_u)) ,

            # Danger right
            (dir_r and game.is_collision(point_r)) ,

            # Danger left
            (dir_l and game.is_collision(point_l)),

            # Danger down
            (dir_d and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head_pos.x,  # food left
            game.food.x > game.head_pos.x,  # food right
            game.food.y < game.head_pos.y,  # food up
            game.food.y > game.head_pos.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, action_new, done):
        self.memory.append((state, action, reward, next_state, action_new, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, action_new, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, action_new, dones)

    def train_short_memory(self, state, action, reward, next_state, action_new, done):
        self.trainer.train_step(state, action, reward, next_state, action_new, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            final_move = random.choice(MOVES)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            print(move)
            final_move[move] = 1

        return final_move



def train():
    agent = Agent(12,16,4)
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
            # train long memory, plot result
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
