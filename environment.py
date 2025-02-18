import pygame
import random

def main():
    # Pygame initialisieren
    pygame.init()


    #Bilder der Schlange laden
    pic_head = pygame.image.load('assets/snake_head.png')
    pic_body = pygame.image.load('assets/snake_body.png')
    pic_apple = pygame.image.load('assets/apple.png')
    pic_background = pygame.image.load('assets/snake_background.png')

    # Fenster- und Spielvariablen
    window_width = 500
    window_height = 500
    snake_block_size = 25
    game_speed = 15
    text_size = 30
    text_x_offset = window_width / 6
    text_y_offset = window_height / 3

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
    snake_y_change = 0

    snake_list = []    # Enthält alle Segmente (Positionen) der Schlange
    snake_length = 1   # Startlänge der Schlange

    # Zufällige Position des Futters (auf einem Raster, das zur Schlange passt)
    food_x = round(random.randrange(0, window_width - snake_block_size) / snake_block_size) * snake_block_size
    food_y = round(random.randrange(0, window_height - snake_block_size) / snake_block_size) * snake_block_size

    game_over = False
    game_close = False

    while not game_over:
        # Schleife für den Game-Over-Zustand
        while game_close:
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
                        main()  # Neustart des Spiels
                        return

        # Ereignisschleife
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                # Richtungsänderung: jeweils nur in eine Richtung (kein diagonaler Move)
                if event.key == pygame.K_LEFT:
                    snake_x_change = -snake_block_size
                    snake_y_change = 0
                elif event.key == pygame.K_RIGHT:
                    snake_x_change = snake_block_size
                    snake_y_change = 0
                elif event.key == pygame.K_UP:
                    snake_y_change = -snake_block_size
                    snake_x_change = 0
                elif event.key == pygame.K_DOWN:
                    snake_y_change = snake_block_size
                    snake_x_change = 0

        # Kollision mit den Spielfenstergrenzen
        if snake_x >= window_width or snake_x < 0 or snake_y >= window_height or snake_y < 0:
            game_close = True

        # Position der Schlange aktualisieren
        snake_x += snake_x_change
        snake_y += snake_y_change

        screen.fill(color_background)
        # Futter zeichnen
        #pygame.draw.rect(screen, color_food, [food_x, food_y, snake_block_size, snake_block_size])
        screen.blit(pic_apple, (food_x, food_y))

        # Aktuelle Position als Kopf der Schlange definieren
        snake_head = [snake_x, snake_y]
        snake_list.append(snake_head)
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Kollision mit sich selbst
        for segment in snake_list[:-1]:
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
        if snake_x == food_x and snake_y == food_y:
            food_x = round(random.randrange(0, window_width - snake_block_size) / snake_block_size) * snake_block_size
            food_y = round(random.randrange(0, window_height - snake_block_size) / snake_block_size) * snake_block_size
            snake_length += 1

        clock.tick(game_speed)

    pygame.quit()

if __name__ == "__main__":
    main()