#!/usr/bin/env python3

import numpy as np


def swap_rows(A, row1, row2):
    """Swap the rows row1 and row2 of A"""
    v_1 = A[row1].copy()
    A[row1] = A[row2]
    A[row2] = v_1
    return A


def subtract_scaled(A, dst, src, scale):
    """Subtract scale times the row src from the row dst of A"""
    A[dst] -= scale * A[src]
    return A


def pivot_index(A, column):
    """Compute the row index of the maximum element of A in this column,
       excluding all elements above the diagonal of A."""
    return np.abs(A[column:, column]).argmax()+column


def lu_decompose(A):
    """Return (P, L, U), such that A = P @ L @ U"""
    P = np.eye(A.shape[0])
    L = np.zeros(A.shape)
    for column in range(A.shape[0]):
        max_diag = pivot_index(A, column)
        swap_rows(A, column, max_diag)
        swap_rows(P, column, max_diag)
        swap_rows(L, column, max_diag)
        factor = 1 / A[column][column]
        for down in range(column+1,A.shape[0]):
            l = A[down][column] * factor
            L[down][column] = l
            subtract_scaled(A, down, column, l)
    L=L+np.eye(A.shape[0])
    return P.T, L, A

def forward_substitute(L, b):
    """Return y such that L @ y = b"""
    vec = np.zeros(L.shape[0])
    vec[0] = b[0]/L[0][0]
    for i in range(1,L.shape[0]):
        vec[i] = (b[i]-(vec*L[i]).sum())/L[i][i]
    return vec.reshape(-1,1)

def backward_substitute(R, y):
    vec = np.zeros(R.shape[0])
    vec[-1] = y[-1]/R[-1][-1]
    for i in reversed(range(R.shape[0]-1)):
        vec[i] = (y[i]-((vec*R[i])).sum())/R[i][i]

    return vec.reshape(-1,1)


def linsolve(A, *bs):
    """Return (x1, ..., xn), such that A @ xk = bs[k]"""
    # first decomp
    P,L,U = lu_decompose(A)
    res = []
    for b in bs:
        y = forward_substitute(L,P.T@b)
        res+=[backward_substitute(U,y)]
    return tuple(res)


def main():
    # Hier kann beliebiger Testcode stehen, der bei der Korrektur ignoriert wird
    A = np.array([[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 0.]])
    #print(A)
    #print("pivotzeile1", pivot_index(A, 0))
    #A = swap_rows(A, 0, 2)
    #print("pivotzeile1", pivot_index(A, 0))
    #print(subtract_scaled(A, 0, 2, 1))
    #print("A in\n",A)

    P,L,R = lu_decompose(A)
    assert np.array_equal(P@L@R, A), P@L@R
    print("done")
    print(backward_substitute(np.array([[42]])
    , np.array([[42]])))
    pass


if __name__ == "__main__":
    main()
    [[1., 0.],
     [2., 1.]]
    [[1.],
    [1.]]
    1,
