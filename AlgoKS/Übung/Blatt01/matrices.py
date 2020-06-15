#!/usr/bin/env python3


from math import *
import numpy as np

def rotation_matrix(omega):
    return np.array([
        [np.cos(omega),-np.sin(omega)],
        [np.sin(omega),np.cos(omega)]])


def stretch_matrix(s, d):
    return np.eye(d)*s


def compose(*matrices):
    if not matrices:
        raise Exception("Expected at least one Matrix")
    m = np.eye(len(matrices[0]))
    for i in matrices:
        if len(i)!=m.shape[1]:
            raise Exception("dimension mismatch")
        m=m@i
    return m


def main():
    # Hier kann beliebiger Testcode stehen, der bei der Korrektur ignoriert wird
    for omega in np.arange(0,3.14,0.01):
        mm = compose(rotation_matrix(omega), rotation_matrix(-omega))
        print(mm[0][0])
        assert np.array_equal(mm.astype(np.int32),np.eye(2).astype(np.int32)),print(mm, omega)
    assert np.array_equal(compose(rotation_matrix(2), rotation_matrix(-2)),stretch_matrix(1,2))
    for i in np.arange(1,10,1):
        for j in range(10):
            assert np.array_equal(compose(stretch_matrix(i,j), stretch_matrix(1/i,j)), np.eye(j)), print(j,i)
    #composition tests:
    a = np.random.rand(12,3)
    b = np.random.rand(3,4)
    c = np.random.rand(4,5)
    d = np.random.rand(5,10)
    e = np.random.rand(10,1)
    ls = [a,b,c,d,e]
    #should be unifiable
    for (i,j) in zip(ls,ls[1:]):
        compose(i,j)
    for (i,j,k) in zip(ls,ls[1:],ls[2:]):
        compose(i,j)
    compose(*ls)
    print("finished non-destructive testing")
    #shouldn't be
    try:
        compose(b,a)
    except Exception as e:
        print(e)



if __name__ == "__main__": main()