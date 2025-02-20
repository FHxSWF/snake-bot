import pygame
import random




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

    while not game_over:

        # Schleife für den Game-Over-Zustand
        while game_close:
            # Startposition und -geschwindigkeit der Schlange
            snake_x = window_width / 2
            snake_y = window_height / 2
            snake_x_change = 0
            snake_y_change = snake_block_size

            snake_list = []  # Enthält alle Segmente (Positionen) der Schlange
            snake_length = 1  # Startlänge der Schlange

            game_over = False
            game_close = False

            direction = "DOWN"  # Aktuelle Richtung

            """
            screen.fill(color_background)
            message = font_style.render("Game Over! Drücke Q zum Beenden oder C zum Neustarten", True, color_text)
            screen.blit(message, (text_x_offset, text_y_offset))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        
            """
        # Ereignisschleife
        already_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if already_pressed:
                    break
                already_pressed = True
                if event.key == pygame.K_LEFT and direction != "RIGHT": #Wenn Schlange nach links geht, dann kann sie nicht nach rechts (in entgegengesetzte Richtung)
                    snake_x_change, snake_y_change = -snake_block_size, 0
                    direction = "LEFT"
                elif event.key == pygame.K_RIGHT and direction != "LEFT":
                    snake_x_change, snake_y_change = snake_block_size, 0
                    direction = "RIGHT"
                elif event.key == pygame.K_UP and direction != "DOWN":
                    snake_x_change, snake_y_change = 0, -snake_block_size
                    direction = "UP"
                elif event.key == pygame.K_DOWN and direction != "UP":
                    snake_x_change, snake_y_change = 0, snake_block_size
                    direction = "DOWN"



        # Position der Schlange aktualisieren
        snake_x += snake_x_change
        snake_y += snake_y_change

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

    pygame.quit()




if __name__ == "__main__":
    main()
