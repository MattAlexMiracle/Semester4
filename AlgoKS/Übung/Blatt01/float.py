#!/usr/bin/env python3

from math import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def myexp(x):
    sum = 1.0
    inc = 1.0
    n = 1
    while sum != sum + inc:
        # TODO inc = ?
        inc=inc*x/n
        
        sum += inc
        n += 1
    return sum


def diff1(f, x, h=1e-8):
    return (f(x+h)-f(x))/h


def diff2(f, x, h=1e-8):
    return (f(x+h)-f(x-h))/(2*h)


def make_exp_plot():
    x = np.arange(-40.0, 0.0, 0.1)
    plt.figure()
    plt.grid(True)
    plt.semilogy(x, np.vectorize(exp)(x), 'o-', label='exp(x)')
    plt.semilogy(x, np.vectorize(myexp)(x), 'o-', label='myexp(x)')
    plt.legend(loc='upper center', shadow=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def make_diff_plot():
    a, h = 1.0, 0.05
    def f(x): return np.sin(a*x*pi) * np.exp(-x)
    def df(x): return -np.exp(-x) * (np.sin(pi*a*x) - pi*a*np.cos(pi*a*x))
    def df1(x): return np.vectorize(lambda x: diff1(f, x, h))(x)
    def df2(x): return np.vectorize(lambda x: diff2(f, x, h))(x)

    # Draw the plot
    fig = plt.figure()
    plt.grid(True)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    x = np.arange(-5.0, 5.0, 0.03)
    [line1] = ax.plot(x,  f(x), 'o-', label="f(x) = sin(aπx)")
    [line2] = ax.plot(x, df1(x), 'o-', label="f'(x) mit diff1")
    [line3] = ax.plot(x, df2(x), 'o-', label="f'(x) mit diff2")
    [line4] = ax.plot(x, df(x), 'o-', label="f'(x) exakt")
    ax.set_xlim([0, 5.0])
    ax.set_ylim([-5, 5.0])
    plt.legend(loc='upper center', shadow=True)
    fig.canvas.draw_idle()

    # Add two sliders for tweaking the parameters
    a_slider_ax  = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    h_slider_ax  = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    a_slider = Slider(a_slider_ax, 'a', 0.0, 3.0, valinit=a)
    h_slider = Slider(h_slider_ax, 'h', 0.01, 0.1, valinit=h)

    def update(new_a, new_h):
        nonlocal a,h
        a, h = new_a, new_h
        line1.set_ydata(f(x))
        line2.set_ydata(df1(x))
        line3.set_ydata(df2(x))
        line4.set_ydata(df(x))
    a_slider.on_changed(lambda a: update(a,h))
    h_slider.on_changed(lambda h: update(a,h))

    plt.show()


def main():
    # Hier kann beliebiger Testcode stehen, der bei der Korrektur ignoriert wird

    # z.B. ein Plot, der myexp und math.exp vergleicht
    make_exp_plot()

    # oder ein interaktiver Plot über numerisches Integrieren
    make_diff_plot()


if __name__ == "__main__": main()