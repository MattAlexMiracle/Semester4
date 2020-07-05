#!/usr/bin/env python3

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


def interpolate_linearly(a, b):
    """Return an object of type numpy.poly1d withe degree 1 that passes through a and b."""
    m=(a[1]-b[1])/(a[0]-b[0]) 
    t= (a[0]*b[1]-b[0]*a[1])/(a[0]-b[0])
    return np.poly1d([ m,t])


def newton_matrix(X):
    """Setup the matrix of the LSE which is used to determine the coefficients
    of the Newton-basis.  X are the x-coordinates of the nodes which are
    used for interpolation."""
    A = np.zeros((X.shape[0],X.shape[0])) 
    if len(X)==0:
        return A
    A[:,0]=1
    for i in range(1,X.shape[0]):
        A[i:,i]= A[i:,i-1]*(X[i:]-X[i-1])
    return A


def newton_polynomial(C, X):
    """Take coefficients and interpolation point x-coordinates of the
Newton-polynomial and determine the corresponding interpolation polynomial."""
    assert len(C) == len(X)
    if len(C)==0:
        return np.poly1d([0])
    polys=np.poly1d([C[0]])
    for i in range(1,len(C)):
        polys = polys+C[i]*np.poly1d(X[:i],True)

    return polys


def interpolating_polynomial(X,Y):
    """Determine the interpolating polynomial for the given NumPy arrays of x and y coordinates."""
    assert len(X) == len(Y)
    A = newton_matrix(X)
    C = np.linalg.solve(A,Y)
    return newton_polynomial(C,X)
    


def interpolation_plot(X,Y):
    p = interpolating_polynomial(X, Y)
    px = np.arange(min(X)-0.1, max(X)+0.11, 0.01)
    plt.grid(True)
    plt.plot(X, Y, "o")
    plt.plot(px, p(px))
    plt.show()


def main():
    X = np.array([0, 1, 2,3])
    Y = np.array([-2.,3.,1.,2.])
    print(newton_matrix(X))
    print(newton_polynomial(np.array([1,2,3,4]), np.array([1,2,3,4])))
    print(newton_polynomial(np.array([1., 2., 3.]), np.array([0., 1., 1.])))
    interpolation_plot(X, Y)

if __name__ == "__main__": main()
