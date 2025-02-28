from collections import namedtuple
from enum import Enum


import pygame
import random


# Pygame initialisieren
pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Window and Game variables
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
SNAKE_BLOCK_SIZE = 25
GAME_SPEED = 10000
TEXT_SIZE = 30
TEXT_X_OFFSET = WINDOW_WIDTH / 6
TEXT_Y_OFFSET = WINDOW_HEIGHT / 3

# Schriftart
FONT_STYLE = pygame.font.SysFont(None, TEXT_SIZE)

# Farben (R, G, B)
COLOR_TEXT = (255, 255, 255)

# Bilder der Schlange laden
PIC_HEAD = pygame.image.load('../assets/snake_head.png')
PIC_BODY = pygame.image.load('../assets/snake_body.png')
PIC_APPLE = pygame.image.load('../assets/apple.png')
PIC_BACKGROUND = pygame.image.load('../assets/snake_background.png')

#All Positions
ALL_POSITIONS = set()
for x in range(WINDOW_WIDTH):
    if x % SNAKE_BLOCK_SIZE == 0:
        for y in range(WINDOW_HEIGHT):
            if y % SNAKE_BLOCK_SIZE == 0:
                ALL_POSITIONS.add(Point(x, y))
#Border
MAP_BORDER = []
for x in range(WINDOW_WIDTH):
    if x % SNAKE_BLOCK_SIZE == 0:
        MAP_BORDER.append(Point(x , -SNAKE_BLOCK_SIZE))
        MAP_BORDER.append(Point(x , WINDOW_WIDTH + SNAKE_BLOCK_SIZE))
for y in range(WINDOW_HEIGHT):
    if y % SNAKE_BLOCK_SIZE == 0:
        MAP_BORDER.append(Point(-SNAKE_BLOCK_SIZE, y))
        MAP_BORDER.append(Point(WINDOW_HEIGHT + SNAKE_BLOCK_SIZE, y))

class SnakeEnvironment:
    def __init__(self):

        # init display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake environment")
        self.clock = pygame.time.Clock()

        # init game state
        self.already_pressed = False
        self.head_pos = Point(int(WINDOW_WIDTH/2), int(WINDOW_HEIGHT / 2))
        self.snake_change = Point(0, SNAKE_BLOCK_SIZE)


        self.snake_list = []
        self.snake_list.append(self.head_pos)# Enthält alle Segmente (Positionen) der Schlange
        self.snake_length = 1  # Startlänge der Schlange


        self.direction = Direction.DOWN

        self.score = 0
        self.food = None
        self._place_food()

    def reset(self):
        # init game state
        self.already_pressed = False
        self.head_pos = Point(int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2))
        self.snake_change = Point(0, SNAKE_BLOCK_SIZE)

        self.snake_list = []
        self.snake_list.append(self.head_pos)  # Enthält alle Segmente (Positionen) der Schlange
        self.snake_length = 1  # Startlänge der Schlange

        self.direction = Direction.DOWN

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        valid_positions: set[Point] = ALL_POSITIONS - set(self.snake_list)
        x,y = random.choice(tuple(valid_positions)) if valid_positions else None
        self.food = Point(x,y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head_pos
        hindernis = MAP_BORDER.copy()
        hindernis.extend(self.snake_list[:-1])
        # Kollision Hindernis
        for segment in hindernis:
            if segment == pt:
                return True
        return False

    def _update_ui(self):
        self.screen.blit(PIC_BACKGROUND, (0, 0))
        # Futter zeichnen
        self.screen.blit(PIC_APPLE, self.food)
        # Schlange zeichnen
        for segment in self.snake_list:
            # pygame.draw.rect(screen, color_snake, [segment[0], segment[1], snake_block_size, snake_block_size])
            if segment == self.head_pos:
                self.screen.blit(PIC_HEAD, (segment[0], segment[1]))
            else:
                self.screen.blit(PIC_BODY, (segment[0], segment[1]))
        # Punkteanzeige (Score)
        self.score = self.snake_length - 1
        score_text = FONT_STYLE.render("Score: " + str(self.score), True, COLOR_TEXT)
        self.screen.blit(score_text, (0, 0))
        pygame.display.update()

    def play_step(self, action):
        self.already_pressed = False

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if self.already_pressed:
                    break
                self.already_pressed = True
                if event.key == pygame.K_LEFT and self.direction != "RIGHT":
                    self.snake_change = Point(-SNAKE_BLOCK_SIZE, 0)
                    self.direction = "LEFT"
                elif event.key == pygame.K_RIGHT and self.direction != "LEFT":
                    self.snake_change = Point(SNAKE_BLOCK_SIZE, 0)
                    self.direction = "RIGHT"
                elif event.key == pygame.K_UP and self.direction != "DOWN":
                    self.snake_change = Point(0, -SNAKE_BLOCK_SIZE)
                    self.direction = "UP"
                elif event.key == pygame.K_DOWN and self.direction != "UP":
                    self.snake_change = Point(0, SNAKE_BLOCK_SIZE)
                    self.direction = "DOWN"

        # 2. move
        self._move(action)

        # 3. check if game over
        game_over = False
        reward = -0.1  # Jede Bewegung kostet -1 Punkt

        if self.is_collision():
            game_over = True
            reward = -100  # Höhere Bestrafung für Kollision
            return reward, game_over, self.score

        # Strafe für Nähe zur Wand
        MARGIN = 2 * SNAKE_BLOCK_SIZE  # Kritischer Bereich am Rand
        if (self.head_pos.x < MARGIN or self.head_pos.x > WINDOW_WIDTH - MARGIN or
                self.head_pos.y < MARGIN or self.head_pos.y > WINDOW_HEIGHT - MARGIN):
            reward -= 5  # Stärkere Strafe für Nähe zur Wand

        # Strafe für Nähe zum eigenen Körper
        for segment in self.snake_list[:-1]:  # Letztes Segment (Kopf) ignorieren
             if abs(self.head_pos.x - segment.x) < SNAKE_BLOCK_SIZE * 2 and abs(
                    self.head_pos.y - segment.y) < SNAKE_BLOCK_SIZE * 2:
                reward -= 2  # Mäßige Strafe für Nähe zum eigenen Körper
                break  # Nur einmal bestrafen pro Schritt

        # 4. Nahrung gefressen?
        if self.head_pos == self.food:
            reward = 100  # Bonus für Essen
            self._place_food()
            self.snake_length += 1
        else:
            # Belohnung für Annäherung an die Nahrung
            old_distance = abs(self.head_pos.x - self.food.x) + abs(self.head_pos.y - self.food.y)
            new_distance = abs((self.head_pos.x + self.snake_change.x) - self.food.x) + abs(
                (self.head_pos.y + self.snake_change.y) - self.food.y)
            if new_distance < old_distance:
                reward += 10  # Kleine Belohnung für Annäherung
            else:
                reward -= 0.5  # Kleine Bestrafung für Entfernung

        self._update_ui()
        self.clock.tick(GAME_SPEED)

        return reward, game_over, self.score


    def _move(self, action):
        x,y = self.head_pos.x + self.snake_change.x, self.head_pos.y + self.snake_change.y
        self.head_pos = Point(x,y)
        self.snake_list.append(self.head_pos)
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]
        if self.already_pressed:
            return
        self.already_pressed = True
        #LEFT
        if action == [1,0,0,0] and self.direction != Direction.RIGHT:  # Wenn Schlange nach links geht, dann kann sie nicht nach rechts (in entgegengesetzte Richtung)
            self.snake_change = Point(-SNAKE_BLOCK_SIZE, 0)
            self.direction = Direction.LEFT
        #RIGHT
        elif action == [0,1,0,0] and self.direction != Direction.LEFT:
            self.snake_change = Point(SNAKE_BLOCK_SIZE, 0)
            self.direction = Direction.RIGHT
        #UP
        elif action == [0,0,1,0] and self.direction != Direction.DOWN:
            self.snake_change = Point(0, -SNAKE_BLOCK_SIZE)
            self.direction = Direction.UP
        #DOWN
        elif action == [0,0,0,1] and self.direction != Direction.UP:
            self.snake_change = Point(0, SNAKE_BLOCK_SIZE)
            self.direction = Direction.DOWN

    def ate(self):
        # Prüfen, ob Futter gefressen wurde
        if self.head_pos == self.food:
            self._place_food()
            self.snake_length += 1
            self



if __name__ == "__main__":

        game = SnakeEnvironment()

        # game loop
        while True:
            game_over, score = game.play_step(None)

            if game_over == True:
                break

        print('Final Score', score)

        pygame.quit()
