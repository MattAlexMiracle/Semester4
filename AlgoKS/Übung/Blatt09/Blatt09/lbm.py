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
    fcop = f.copy()
    tmp = max(n, m)
    coords = [(x, y) for x in range(tmp) for y in range(tmp) if x < n and y < m]
    big_e = np.expand_dims(directions, 1) + np.expand_dims(coords, 0)
    big_filter = np.stack(
        (
            (0 < big_e[:, :, 0]) * (big_e[:, :, 0] < n),
            (0 < big_e[:, :, 1]) * (big_e[:, :, 1] < m),
        ),
        2,
    )  # das hier geht, weil (0,0) immer ausmaskiert wird
    e2_big = np.where(big_filter, big_e, 0)
    coords2_big = np.where(big_filter, coords, 0,)
    # die schleife bringt man sicher auch nocht los, hab jetzt aber nicht die gedult dazu
    for k in range(9):
        f[k, e2_big[k, :, 0], e2_big[k, :, 1]] = fcop[
            k, coords2_big[k, :, 0], coords2_big[k, :, 1]
        ]

    f = f * mask
    f = f + fcop * (1 - mask)

    return f


def noslip(f, masklist):
    """ masklist ist eine liste von (x,y) coordinaten """
    for (x, y) in masklist:
        f[:, x, y] = f[::-1, x, y]
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
    # custom random obstacle
    randomX = np.random.choice(W, 50)
    randomY = np.random.choice(H, 50)
    #mask[randomX, randomY] = 1
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
        fig, lambda _: update_velocity_field(), frames=100, interval=10, blit=False
    )

    plt.show()


def main():
    lbm(300, 60)


if __name__ == "__main__":
    main()
