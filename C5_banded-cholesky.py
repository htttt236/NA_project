import numpy as np


# def make_A_in_band(n):
#     q = n -1
#     m = q*q
#     A = np.zeros((m,n))

#     for r in range(q):
#         for c in range(q):
#             i = r*q + c
#             A[i,0] = 4

#             if c>0:
#                 A[i,1] = -1
#             if r>0:
#                 A[i,q] = -1
#     return A


def make_A_in_band(n):              # A의 대각 아래 band만 생성
    A = np.zeros((n-1, n-1, n))
    
    A[:, :, 0] = 4
    A[:, 1:, 1] = -1
    A[1:, :, n-1] = -1
    
    return A.reshape((n-1)**2, n)


def cholesky_decomposition(L, b):
    m, n = L.shape

    # Ly = b 계산 (전진 대입)
    y = np.zeros(m)
    for i in range(m):
        s = 0
        for k in range(max(0, i-n+1), i):
            s += L[i, i-k] * y[k]

        y[i] = (b[i] - s) / L[i, 0]

    # L^T u = y 계산 (후진 대입)
    u = np.zeros(m)
    for i in range(m-1, -1, -1):
        s = 0
        for k in range(i+1, min(m, i+n)):
            s += L[k, k-i] * u[k]

        u[i] = (y[i] - s) / L[i, 0]

    return u