#!/usr/bin/env python3

import numpy as np
import math


def givens_rotation(A, i, j):
    """Return the rotation matrix J, such that (J @ A)[i,j] = 0.0"""
    J = np.eye(A.shape[0])
    sigma = np.sign(A[j][j])
    #np.sign returns 0 for 0
    sigma = sigma if sigma!=0 else 1
    length = sigma*np.sqrt( A[i,j]**2+A[j,j]**2)
    c = A[j][j]/length
    s = A[i][j]/length
    J[i][i]= c
    J[j][j]= c
    J[j][i]= s
    J[i][j]=-s
    return J


def qr_decompose(A):
    """Return (Q, R), such that A = Q @ R"""
    Q = np.eye(A.shape[0])
    for i in range(A.shape[0]-1):
        for j in range(i+1,A.shape[0]):
            qn = givens_rotation(A, j,i)
            Q = Q@(qn.T)
            A = qn@A
    return (Q,A)


def backward_substitute(R, y):
    """Return x such that R @ x = y"""
    vec = np.zeros(R.shape[0])
    #vec[-1] = y[-1]/R[-1][-1]
    #print("v", vec)
    for i in range(R.shape[0]-1,-1,-1):
        vec[i] = (y[i]-((vec*R[i])).sum())/R[i][i]

    return vec.reshape(-1,1)


def linsolve(A, *bs):
    """Return (x1, ..., xn), such that A @ xk = bs[k]"""
    Q,R = qr_decompose(A)
    x = []
    for b in bs:
        y = Q.T@b
        x+=[backward_substitute(R,y)]
    return tuple(x)


def main():
    # Hier kann beliebiger Testcode stehen, der bei der Korrektur ignoriert wird
    """givens_rotation(np.array([[0., 1.],
                   [1., 1.]]), 1, 0)"""
    print(qr_decompose(np.array([[3.],
       [4.]])))
    pass


if __name__ == "__main__": main()