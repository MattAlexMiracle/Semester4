#!/usr/bin/env python3

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

weights = np.array(
    [
        1.0 / 36,
        1.0 / 9,
        1.0 / 36,
        1.0 / 9,
        4.0 / 9,
        1.0 / 9,
        1.0 / 36,
        1.0 / 9,
        1.0 / 36,
    ]
)

directions = np.array(
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
)

inverse_dir = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])


def density(f):
    return f.sum(0)


def velocity(f):
    rho = density(f)
    u = 1 / rho * np.einsum("ab,acd->bcd", directions, f)
    return (u[0], u[1])


def f_eq(f):
    ux, uy = velocity(f)
    dir_vector = np.reshape(directions[:, 0], (9, 1, 1)) * np.expand_dims(
        ux, 0
    ) + np.reshape(directions[:, 1], (9, 1, 1)) * np.expand_dims(uy, 0)
    feq = (
        weights.reshape(9, 1, 1)
        * density(f)
        * (1 + 3 * dir_vector + 9 / 2 * dir_vector ** 2 - 3 / 2 * (ux * ux + uy * uy))
    )
    return feq


def collide(f, omega):
    return f - omega * (f - f_eq(f))


def stream(f):
    _, n, m = f.shape
    # masking
    mask = np.zeros((n, m))
    mask[1:-1, 1:-1] = 1
    print(n, m)
    fcop = f.copy()
    for i in range(n):
        for j in range(m):
            for k in range(9):
                # es = filter(lambda x : 0 < x[0] < n and 0 < x[1] < m, directions+[i,j])
                e = directions[k] + [i, j]
                if 0 < e[0] < n and 0 < e[1] < m:
                    f[k, e[0], e[1]] = fcop[k, i, j]
    f = f * mask
    f = f + fcop * (1 - mask)

    return f


def noslip(f, masklist):
    pass  # TODO
    return f


flow = np.array(
    [
        1.0 / 36,
        1.0 / 9,
        1.6 / 36,
        1.0 / 9,
        4.0 / 9,
        1.6 / 9,
        1.0 / 36,
        1.0 / 9,
        1.6 / 36,
    ]
)


def lbm(W, H, timesteps=1000, omega=1.85):
    fig, ax = plt.subplots()
    f = weights.reshape((9, 1, 1)) * np.ones((9, W, H))
    f[:, 0, :] = flow.reshape(9, 1)
    # create the list of obstacles
    mask = np.zeros((W, H), dtype=bool)
    mask[int(0.2 * W) : int(0.22 * W), int(0.4 * H) : int(0.6 * H)] = 1
    mask[1:-1, 1] = 1
    mask[1:-1, -2] = 1
    masklist = np.argwhere(mask)

    (ux, uy) = velocity(f)
    mat = ax.matshow((np.sqrt(ux * ux + uy * uy)).transpose(), cmap=cm.GnBu)

    def update_velocity_field():
        nonlocal f
        for rep in range(10):
            # inflow
            f[:, 0, :] = flow.reshape(9, 1)
            # copying outflow
            f[:, -2, :] = f[:, -1, :]
            f = stream(noslip(collide(f, omega), masklist))
        (ux, uy) = velocity(f)
        mat.set_data((np.sqrt(ux * ux + uy * uy)).transpose())

    ani = animation.FuncAnimation(
        fig, lambda _: update_velocity_field(), frames=10000, interval=10, blit=False
    )

    plt.show()


def main():
    lbm(300, 60)


if __name__ == "__main__":
    main()
