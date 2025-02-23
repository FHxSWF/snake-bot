import random
import torch
from environment_AI import SnakeEnvironment, WINDOW_WIDTH, WINDOW_HEIGHT, SNAKE_BLOCK_SIZE, Direction
from agent import Agent


def get_state(env):
    """
    Erzeugt einen Zustandsvektor aus dem aktuellen Zustand des Environments.
    Nutzt:
      - Normierte Position des Schlangenkopfs (x, y)
      - Normierte Position des Futters (x, y)
      - Aktuelle Richtungsänderung (dx, dy), normiert auf die Blockgröße
    """
    state = [
        env.head_pos.x / WINDOW_WIDTH,
        env.head_pos.y / WINDOW_HEIGHT,
        env.food.x / WINDOW_WIDTH,
        env.food.y / WINDOW_HEIGHT,
        env.snake_change.x / SNAKE_BLOCK_SIZE,
        env.snake_change.y / SNAKE_BLOCK_SIZE,
    ]
    return state


def main():
    # Environment und Agent instanziieren
    env = SnakeEnvironment()
    state_size = 6  # Wie oben definiert in get_state
    hidden_size = 128  # Beispielwert – anpassbar
    output_size = 4  # Mögliche Aktionen: LEFT, RIGHT, UP, DOWN
    agent = Agent(input_size=state_size, hidden_size=hidden_size, output_size=output_size, learning_rate=0.001)

    episodes = 1000
    for episode in range(episodes):
        env.reset()
        state = get_state(env)
        game_over = False
        total_reward = 0

        while not game_over:
            # Berechne den aktuellen Abstand zwischen Schlangenkopf und Futter (z. B. Manhattan-Distanz)
            old_distance = abs(env.head_pos.x - env.food.x) + abs(env.head_pos.y - env.food.y)
            old_length = env.snake_length

            # Aktion auswählen (epsilon-greedy)
            action_str = agent.get_action(state, epsilon=0.1)
            # Mapping von Action-String zu Direction Enum
            mapping = {
                'LEFT': Direction.LEFT,
                'RIGHT': Direction.RIGHT,
                'UP': Direction.UP,
                'DOWN': Direction.DOWN
            }
            action_enum = mapping[action_str]

            # Einen Spielschritt durchführen
            game_over, score = env.play_step(action_enum)

            # Berechne neuen Abstand nach der Aktion
            new_distance = abs(env.head_pos.x - env.food.x) + abs(env.head_pos.y - env.food.y)

            # Reward-Berechnung:
            if env.snake_length > old_length:
                reward = 10  # Apfel gegessen
            elif game_over:
                reward = -10  # Kollision
            elif new_distance < old_distance:
                reward = 1  # Näher am Futter
            else:
                reward = -1  # Entfernt sich vom Futter

            total_reward += reward
            new_state = get_state(env)

            # Training des Agenten
            agent.train_step(state, action_str, reward, new_state, game_over)
            state = new_state

        print(f"Episode {episode + 1} - Score: {score} - Total Reward: {total_reward}")


if __name__ == '__main__':
    main()
