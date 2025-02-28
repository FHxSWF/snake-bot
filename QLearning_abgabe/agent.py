import numpy as np
import os
from collections import defaultdict

import environment_AI
from environment_AI import *
from helper import plot
from environment_AI import Direction, Point

#Quelle
#Hilfreich für den update der Q-Table: https://github.com/Ceruleanacg/Reinforcement-Learning/blob/master/algorithms/Sarsa/sarsa.py
# https://github.com/wikibook/keras/blob/master/chapter9-drl/q-frozenlake-9.5.1.py
#

MOVES = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]  # Indexbasierte Aktionen (anstatt One-Hot-Vektoren)
MAX_MEMORY = 100_000
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class Agent:
    def __init__(self, state_size, action_size):
        """
        Initialisiert den Agenten mit einer Q-Tabelle.

        :param state_size: Anzahl der Zustandsvariablen
        :param action_size: Anzahl der möglichen Aktionen
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon = 1.0  # Start mit hoher Exploration
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.n_games = 0

    def get_state(self, game):
        """
        Wandelt das Spielfeld in einen diskreten Zustand um.

        :param game: Das aktuelle Spielobjekt
        :return: Zustand als Tupel mit relevanten Informationen
        """
        head = game.head_pos
        point_l = Point(head.x - environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_r = Point(head.x + environment_AI.SNAKE_BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - environment_AI.SNAKE_BLOCK_SIZE)
        point_d = Point(head.x, head.y + environment_AI.SNAKE_BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = (
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
        )
        return state

    def get_action(self, state):
        """
        Wählt eine Aktion nach der Epsilon-Greedy-Strategie.

        :param state: Der aktuelle Zustand des Spiels
        :return: Eine Aktion als One-Hot-Array
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            action_index = np.argmax(self.q_table[state])
            action = [0, 0, 0, 0]
            action[action_index] = 1

        print(f"Gewählte Aktion: {action}")  # Debugging-Print
        return action

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Aktualisiert die Q-Tabelle mit der Q-Learning- Aktualisierungsgleichumg.

        :param state: Der vorherige Zustand
        :param action: Die ausgeführte Aktion (One-Hot-Array)
        :param reward: Erhaltener Belohnungswert
        :param next_state: Neuer Zustand nach der Aktion
        :param done: Ob das Spiel beendet ist
        """
        # max_a' Q(s', a') -> bester Q-Wert für nächsten Zustand
        best_next_action = np.argmax(self.q_table[next_state])  # Beste Aktion im nächsten Zustand
        # r + gamma * max_a' Q(s', a') -> Belohnung plus zukünftige Erwartung
        target = reward + (self.gamma * self.q_table[next_state][best_next_action] * (1 - done))

        # Q(s, a) -> Aktueller Q-Wert
        action_index = np.argmax(action)  # Wandelt One-Hot in Index um

        # Q(s, a) = Q(s, a) + alpha * (target - Q(s, a)) -> Q-Wert-Update
        self.q_table[state][action_index] += LEARNING_RATE * (target - self.q_table[state][action_index])

        # Exploration-Reduktion: Epsilon-Decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_tableNewTrain.npy"):
        """
        Speichert die Q-Tabelle als Datei.

        :param filename: Name der Datei zum Speichern
        """
        np.save(filename, dict(self.q_table))

    def load_q_table(self, filename="q_tableNewTrain.npy"):
        """
        Lädt eine gespeicherte Q-Tabelle.

        :param filename: Name der Datei zum Laden
        """
        if os.path.exists(filename):
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), np.load(filename, allow_pickle=True).item())

def train():
    """
    Trainiert den Q-Learning-Agenten im Snake-Spiel.
    """
    agent = Agent(state_size=12, action_size=4)
    game = SnakeEnvironment()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.update_q_table(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.save_q_table()  # Q-Tabelle speichern

            print(f'Game {agent.n_games}, Score {score}, Record: {record}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
