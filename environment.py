import pygame
import random

from enum import Enum
from agent import Agent

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


#RELEVAnTE VARIABLEN
"""
snake_head: aktuelle Position der Schlange
snake_list[]: Liste von Positionen aller Körperteile
direction: String in welche richtung sich die Schlange bewegt
food_pos: koordinaten des essens
hindernis: alle koordinaten in denen man verliert
Aktionen: UP DOWN LEFT RIGHT keys


"""
def main():
    # Pygame initialisieren
    pygame.init()


    # Fenster- und Spielvariablen
    window_width = 500
    window_height = 500
    snake_block_size = 25
    game_speed = 15
    text_size = 30
    text_x_offset = window_width / 6
    text_y_offset = window_height / 3

    # Bilder der Schlange laden
    pic_head = pygame.image.load('assets/snake_head.png')
    pic_body = pygame.image.load('assets/snake_body.png')
    pic_apple = pygame.image.load('assets/apple.png')
    pic_background = pygame.image.load('assets/snake_background.png')


    # Farben (R, G, B)
    color_background = (0, 0, 0)
    color_snake = (0, 255, 0)
    color_food = (255, 0, 0)
    color_text = (255, 255, 255)

    # Fenster erstellen
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Snake environment")
    clock = pygame.time.Clock()

    # Schriftart
    font_style = pygame.font.SysFont(None, text_size)

    # Startposition und -geschwindigkeit der Schlange
    snake_x = window_width / 2
    snake_y = window_height / 2
    snake_x_change = 0
    snake_y_change = snake_block_size

    snake_list = []    # Enthält alle Segmente (Positionen) der Schlange
    snake_length = 1   # Startlänge der Schlange

    #Umrandung der map
    map_border = []
    for x in range(window_width):
        if x % snake_block_size == 0:
            map_border.append((x , -snake_block_size))
            map_border.append((x , window_width+snake_block_size))
    for y in range(window_height):
        if y % snake_block_size == 0:
            map_border.append((-snake_block_size, y))
            map_border.append((window_height+snake_block_size, y))

    # Zufällige Position des Futters (auf einem Raster, das zur Schlange passt)
    all_positions = set()
    for x in range(window_width):
        if x % snake_block_size == 0:
            for y in range(window_height):
                if y % snake_block_size == 0:
                    all_positions.add((x, y))

    valid_positions = list(all_positions - set(snake_list))
    food_pos = random.choice(valid_positions) if valid_positions else None

    game_over = False
    game_close = False

    direction = "DOWN" #Aktuelle Richtung

    agent = Agent(input_size=4, hidden_size=24, output_size=4, learning_rate=0.01)

    while not game_over:


        # Ereignisschleife
        already_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        state = [snake_x, snake_y, food_pos[0], food_pos[1]]
        action = agent.get_action(state)  # Hier wählt der Agent die Aktion

        # Ändere die Richtung basierend auf der vom Agenten gewählten Aktion
        snake_x_change, snake_y_change = agent.get_direction_change(action, snake_block_size)

        # Position der Schlange aktualisieren
        snake_x += snake_x_change
        snake_y += snake_y_change

        next_state = [snake_x, snake_y, food_pos[0], food_pos[1]]

        reward = 0 # für den Agent Belohnung
        done = False
        if snake_x >= window_width or snake_x < 0 or snake_y >= window_height or snake_y < 0:
            reward = -1
            done = True
            game_close = True
        for segment in snake_list[:-1]:
            if segment == (int(snake_x), int(snake_y)):
                reward = -1
                done = True
                game_close = True
                break
        if snake_x == food_pos[0] and snake_y == food_pos[1]:
            reward = 1

        agent.train_step(state, direction, reward, next_state, done)

        screen.blit(pic_background, (0, 0))
        # Futter zeichnen
        #pygame.draw.rect(screen, color_food, [food_x, food_y, snake_block_size, snake_block_size])
        screen.blit(pic_apple, (food_pos[0], food_pos[1]))

        # Aktuelle Position als Kopf der Schlange definieren
        snake_head = (int(snake_x), int(snake_y))
        snake_list.append(snake_head)
        if len(snake_list) > snake_length:
            del snake_list[0]

        #Hindernisse(rand und eigener Körper außer kopf)
        hindernis = map_border.copy()
        hindernis.extend(snake_list[:-1])
        # Kollision Hindernis
        for segment in hindernis:
            if segment == snake_head:
                game_close = True


        # Schlange zeichnen
        for segment in snake_list:
            #pygame.draw.rect(screen, color_snake, [segment[0], segment[1], snake_block_size, snake_block_size])
            if segment == snake_head:
                screen.blit(pic_head, (segment[0], segment[1]))
            else:
                screen.blit(pic_body, (segment[0], segment[1]))

        # Punkteanzeige (Score)
        score = snake_length - 1
        score_text = font_style.render("Score: " + str(score), True, color_text)
        screen.blit(score_text, (0, 0))

        pygame.display.update()

        # Prüfen, ob Futter gefressen wurde
        if snake_x == food_pos[0] and snake_y == food_pos[1]:
            valid_positions = list(all_positions - set(snake_list))
            food_pos = random.choice(valid_positions) if valid_positions else None
            snake_length += 1

        clock.tick(game_speed)

        if game_close:
            print(f"Game Over! Your Score: {score}")

            #Neustart
            game_over = False
            game_close = False
            snake_x = window_width / 2
            snake_y = window_height / 2
            snake_x_change = 0
            snake_y_change = snake_block_size
            snake_list = []
            snake_length = 1
            food_pos = random.choice(valid_positions)

    pygame.quit()


if __name__ == "__main__":
    main()
