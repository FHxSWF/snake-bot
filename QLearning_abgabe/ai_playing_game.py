import numpy as np
import pygame
import random
import os
from collections import defaultdict

import environment_AI
from environment_AI import SnakeEnvironment, Direction, Point

class Agent:
    def __init__(self, state_size, action_size, q_table_file="q_tableFirst.npy"):
        """ Initialisiert den Agenten mit einer geladenen Q-Tabelle """
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.q_table_file = q_table_file
        self.load_q_table()

    def load_q_table(self):
        """ Lädt eine gespeicherte Q-Tabelle """
        if os.path.exists(self.q_table_file):
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), np.load(self.q_table_file, allow_pickle=True).item())


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

    def get_action(self, state):
        """ Wählt eine Aktion basierend auf der geladenen Q-Tabelle """
        action_index = np.argmax(self.q_table[state])
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
