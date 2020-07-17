#!/usr/bin/env python3

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool)


c10orthogonal = np.array(
    [
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    dtype=bool,
)


def gamegrid(w, h, entities):
    grid = np.zeros((h, w), dtype=bool)
    for (entity, x, y) in entities:
        add_entity(grid, entity, x, y)
    return grid


def add_entity(grid, entity, y, x):
    for i in range(len(entity)):
        for j in range(len(entity[i])):
            grid[y + j, x + i] = entity[i][j]
    return grid


def next_step(grid):
    # print(grid.nonzero())
    delta_array = np.zeros(grid.shape)
    n, m = grid.shape
    bevore = grid.copy()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbour = 0
            neighbour += grid[(i - 1) % n, j]
            neighbour += grid[(i + 1) % n, j]
            neighbour += grid[i, (j - 1) % m]
            neighbour += grid[i, (j + 1) % m]
            # corners
            neighbour += grid[(i - 1) % n, (j + 1) % m]
            neighbour += grid[(i + 1) % n, (j + 1) % m]
            neighbour += grid[(i - 1) % n, (j - 1) % m]
            neighbour += grid[(i + 1) % n, (j - 1) % m]
            assert neighbour < 9
            delta_array[i, j] = (
                1
                if neighbour == 3
                and not bevore[i, j]
                or neighbour == 3
                and bevore[i, j]
                or neighbour == 2
                and bevore[i, j]
                else 0
            )
    return delta_array


def gameoflife(grid, steps=100):
    fig, ax = plt.subplots()
    mat = ax.matshow(grid, cmap=cm.gray_r)
    ani = animation.FuncAnimation(
        fig,
        lambda _: mat.set_data(next_step(grid)),
        frames=100,
        interval=50,
        blit=False,
    )
    plt.show()


def main():
    grid = gamegrid(40, 40, [(glider, 13, 4), (c10orthogonal, 25, 25)])
    gameoflife(grid)


if __name__ == "__main__":
    main()
