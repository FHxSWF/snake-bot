import torch
import numpy as np
import pygame

import environment_AI
from agent import Agent
from environment_AI import SnakeEnvironment, Point, Direction


def load_model(model_path, input_size, hidden_size, output_size):
    """
    Lädt das gespeicherte neuronale Netzmodell aus einer Datei.
    :param model_path: Pfad zur gespeicherten Modelldatei (.pth)
    :param input_size: Anzahl der Eingabeneuronen (Zustandsmerkmale)
    :param hidden_size: Anzahl der Neuronen in der versteckten Schicht
    :param output_size: Anzahl der Ausgabeneuronen (mögliche Aktionen)
    :return: ein Agent-Objekt mit dem geladenen Modell
    """
    model = Agent(input_size, hidden_size, output_size)
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    return model


def get_state(game):
    """
    Erstellt den aktuellen Zustand des Spiels als numerisches Array für das neuronale Netz.
    :param game: Instanz der Snake-Umgebung
    :return: Zustandsvektor als Numpy-Array
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

def play_game(model_path):
    """
    Lässt das geladene Modell das Snake-Spiel spielen.
    :param model_path: Pfad zur gespeicherten Modelldatei (.pth)
    """
    game = SnakeEnvironment()
    model = load_model(model_path, input_size=12, hidden_size=32, output_size=4)

    while True:
        state = get_state(game)
        action = model.get_action(state)
        reward, done, score = game.play_step(action)

        if done:
            game.reset()




if __name__ == "__main__":
    play_game(r"model/modelNew.pth")