import pygame
import numpy as np

# Window size
WIDTH, HEIGHT = 800, 600

# Cell size
CELL_SIZE = 10

# Colors
ALIVE_COLOR = (255, 255, 255)
DEAD_COLOR = (0, 0, 0)

class GameOfLife:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = np.random.choice([0,1], size=(height // cell_size, width // cell_size), p=[0.5, 0.5])

    def draw(self, screen):
        screen.fill(DEAD_COLOR)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:
                    pygame.draw.rect(screen, ALIVE_COLOR, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

    def update(self):
        new_grid = np.copy(self.grid)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                live_neighbors = self.count_live_neighbors(i, j)
                if self.grid[i, j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                    new_grid[i, j] = 0
                elif self.grid[i, j] == 0 and live_neighbors == 3:
                    new_grid[i, j] = 1
        self.grid = new_grid

    def count_live_neighbors(self, i, j):
        count = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                col = (j + y + self.grid.shape[1]) % self.grid.shape[1]
                row = (i + x + self.grid.shape[0]) % self.grid.shape[0]
                count += self.grid[row, col]
        count -= self.grid[i, j]
        return count

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game = GameOfLife(WIDTH, HEIGHT, CELL_SIZE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.draw(screen)
        pygame.display.flip()
        game.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
