import numpy as np
import pygame
import random
import os
from collections import defaultdict

import environment_AI
from environment_AI import SnakeEnvironment, Direction, Point

class Agent:
    def __init__(self, state_size, action_size, q_table_files=["q_table_1.npy", "q_table_2.npy"]):
        """ Initialisiert den Agenten mit zwei geladenen Q-Tabellen """
        self.state_size = state_size
        self.action_size = action_size
        self.q_tables = []
        self.q_table_files = q_table_files
        self.load_q_tables()



    def get_state(self, game):
        """ Wandelt das Spielfeld in einen diskreten Zustand um """
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

    def load_q_tables(self):
        """ Lädt die gespeicherten Q-Tabellen und kombiniert sie """
        for file in self.q_table_files:
            if os.path.exists(file):
                q_table = np.load(file, allow_pickle=True).item()
                self.q_tables.append(defaultdict(lambda: np.zeros(self.action_size), q_table))

    # Wurde hinzugefügt damit man beide Tabellen
    def get_action(self, state):
        """
        Wählt die beste Aktion basierend auf den Q-Werten aus zwei geladenen Q-Tabellen.

        - Beide Q-Tabellen werden geprüft.
        - falls der Zustand in einer oder beiden Tabellen existiert, werden die Q-Werte summiert
        - falls der Zustand in beiden Tabellen existiert, wird der Mittelwert der Q-Werte berechnet
        - falls der Zustand in keiner Tabelle existiert, bleibt die Entscheidung zufälig
        - die Aktion mit dem höchsten Q-Wert wird gewählt
        """
        q_values = np.zeros(self.action_size)
        count = 0

        for q_table in self.q_tables:
            if state in q_table:
                q_values += q_table[state]
                count += 1

        if count > 0:
            q_values /= count  # Mittelwert berechnen

        action_index = np.argmax(q_values)
        action = [0, 0, 0, 0]
        action[action_index] = 1
        return action

if __name__ == "__main__":
    game = SnakeEnvironment()
    agent = Agent(state_size=12, action_size=4)

    running = True
    while running:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)

        if done:
            game.reset()

    pygame.quit()
