from sys import stderr

import lu
import qr
import numpy as np


def generateSample(factorNonIntegers=1.9, dim=None, scale=1000):
    def genM(dim1, dim2):
        if dim2 is not None:
            A = (np.random.rand(dim1, dim2) * scale).astype(np.int64).astype(np.float64)
            A += (np.random.rand(dim1, dim2) * factorNonIntegers).astype(np.int64) * np.random.rand(dim1, dim2)
        else:
            A = (np.random.rand(dim1) * scale).astype(np.int64).astype(np.float64)
            A += (np.random.rand(dim1) * factorNonIntegers).astype(np.int64) * np.random.rand(dim1)
        return A

    while True:
        try:
            if dim is None:
                dim = np.random.randint(1, 20)
            A = genM(dim, dim)
            b = genM(dim, None)
            return A, np.array([b]).transpose(), np.array([np.linalg.solve(A, b)]).transpose()
        except np.linalg.LinAlgError as err:
            # print(err)
            pass


largestDelta = 0


def compDelta(A, B):
    global largestDelta
    maximum = np.max(np.abs(A.flatten() - B.flatten()))
    if maximum > largestDelta:
        largestDelta = maximum
        return True
    return False


np.random.seed(0)
np.seterr('raise')
for i in range(100000):

    A, b, x = generateSample()
    try:
        xLU, = lu.linsolve(np.copy(A), b)
        xQR, = qr.linsolve(np.copy(A), b)

        if compDelta(x, xLU):
            # print("===========")
            # print(A)
            # print(b)
            # print(A)
            print(f'LU caused new largestDelta: {largestDelta}')
        if compDelta(x, xQR):
            # print("===========")
            # print(A)
            # print(b)
            # print(x)
            print(f'QR caused new largestDelta: {largestDelta}')
    except Exception as err:
        print(err, file=stderr)
        print('solving')
        print(A)
        print(b)
        print(x)

    if i % 2000 == 0:
        print(f"#{i}")
    # print(x)
    # print(lu.linsolve(A, b))

print(largestDelta)
