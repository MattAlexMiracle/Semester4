import numpy as np
import scipy.linalg as la
import qr

def generate_matrix(size, low_high=(1,10), use_integer=True):
    assert size>1
    R = np.random.uniform(*low_high,size=(size,size))
    if use_integer:
        R = R.round()
    L = np.eye(size)
    for i in range(size):
        for j in range(0,i):
            L[i][j] = R[i][j]
            R[i][j] = 0
    #print(L)
    #print(R)
    A = L@R
    return A

if __name__=='__main__':
    #generate_matrix(3)
    for x in range(2,51):
        print(x)
        for _ in range(1000):
            A=generate_matrix(x)
            v = np.random.randint(3, size=x)
            sola = qr.linsolve(A, v)[0].reshape(x)
            solb = la.solve(A,v).reshape(x)
            print(sola, solb)
            assert np.array_equal(sola,solb)
    