#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def de_casteljau_step(P, t):
    """For a given control polygon P of length n, return a control polygon of
    length n-1 by performing a single de Casteljau step with the given
    floating point number t."""
    assert len(P) > 1, 0 <= t <= 1
    return np.stack([(1-t)*p+t*q for p,q in zip(P[:-1],P[1:])])



def de_casteljau(P, t):
    """Evaluate the Bezier curve specified by the control polygon P at a single
    point corresponding to the given t in [0,1]. Returns a one-dimensional
    NumPy array contining the x and y coordinate of the Point,
    respectively."""
    assert len(P) != 0
    control =P
    for _ in range(len(P)-1):
        control = de_casteljau_step(control, t)
    return control.squeeze(0)


def bezier1(P, m):
    """Return a polygon with m points that approximates the Bezier curve
    specified by the control polygon P."""
    assert len(P) > 1, m > 1
    return np.stack([de_casteljau(P,x) for x in np.linspace(0,1,m)])
    pass # TODO


def add_control_point(P):
    """For the given Bezier curve control polygon P of length n, return a new
    control polygon with n+1 points that describes the same curve."""
    assert len(P) > 1
    Q0 = [P[0]]
    Qi =[ ((i+1)/len(P)) * P[i]+(1-((i+1)/len(P)))*P[i+1] for i, p in enumerate(P[:-1])]
    Qn=[P[-1]]
    return np.stack(Q0+Qi+Qn)


def split_curve(P):
    """Split a Bezier curve, specified by a control polynomial P. Return a
    tuple (L, R), where L and R are control polygons with the same
    length as P, that describe the left and the right half of the original
    curve, respectively."""
    L,R = [],[]
    P1 =P
    for _ in range( len(P1)-1):
        print("step", len(P1))
        L +=[P1[0]]
        R =[P1[-1]]+R
        print(L,R)
        P1 = de_casteljau_step(P1,0.5)  
    L +=[P1[0]]
    R=[P1[-1]]+R
    return (np.stack(L),np.stack(R))
    


def bezier2(P, depth):
    """Return a polygon that approximates the Bezier curve specified by the
    control polygon P by depth recursive subdivisions."""
    if depth<=0:
        return P
    L,R = split_curve(P)
    L1 = bezier2(L,depth-1)
    R1 = bezier2(R,depth-1)
    return np.concatenate([L1[:-1],R1],0)


def de_casteljau_plot(P):
    """Draw all polygons in the de Casteljau pyramid of P for varying t."""
    n = len(P)
    t = 0.3
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    lines = ax.plot(P[:,0], P[:,1], 'o-')
    Q = P.copy()
    for i in range(n-1):
        Q = de_casteljau_step(Q,t)
        [line] = ax.plot(Q[:,0], Q[:,1], 'o-')
        lines.append(line)
    plt.grid(True)

    def redraw(t):
        Q = P.copy()
        for i in range(n-1):
            Q = de_casteljau_step(Q,t)
            lines[i+1].set_xdata(Q[:,0])
            lines[i+1].set_ydata(Q[:,1])

    fig.subplots_adjust(left=0.25, bottom=0.25)
    fig.canvas.draw_idle()
    t_slider_ax  = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    t_slider = Slider(t_slider_ax, 't', 0., 1., valinit=t)
    t_slider.on_changed(redraw)
    plt.show()


def bezier_plot(P):
    """Draw different bezier curve approximations for the given P."""
    n = len(P)
    depth = 1
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    B2 = bezier2(P.copy(), depth)
    B1 = bezier1(P.copy(), len(B2))
    [line0] = ax.plot( P[:,0],  P[:,1], 'o-', label="P")
    [line1] = ax.plot(B1[:,0], B1[:,1], 'o-', label="bezier1")
    [line2] = ax.plot(B2[:,0], B2[:,1], 'o-', label="bezier2")
    plt.legend(shadow=True)
    plt.grid(True)

    def redraw(depth):
        depth = int(depth)
        B2 = bezier2(P.copy(), depth)
        line2.set_xdata(B2[:,0])
        line2.set_ydata(B2[:,1])
        B1 = bezier1(P.copy(), len(B2))
        line1.set_xdata(B1[:,0])
        line1.set_ydata(B1[:,1])

    fig.subplots_adjust(left=0.25, bottom=0.25)
    fig.canvas.draw_idle()
    depth_slider_ax  = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    depth_slider = Slider(depth_slider_ax, 'depth', 0, 7, valinit=depth)
    depth_slider.on_changed(redraw)
    plt.show()


def main():
    P = np.array([[3., 2.], [2., 5.], [7., 6.], [8., 1.]])
    P2 = np.array([[0., 0.],[1., 0.]])
    de_casteljau(P,0.5)
    print(split_curve(P))
    print(split_curve(P2))
    de_casteljau_plot(P)
    # bezier_plot(P)


if __name__ == "__main__": main()