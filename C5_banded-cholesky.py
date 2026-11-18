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


def cholesky_decomposition(A):
    m, n = A.shape
    L = np.zeros_like(A)

    for j in range(m):
        # 대각 원소 계산
        s = 0
        for k in range(max(0, j-n+1), j):
            s += L[j, j-k]**2

        L[j, 0] = np.sqrt(A[j, 0] - s)

        # 대각선 아래 원소 계산
        for i in range(j+1, min(m, j+n)):

            s = 0
            for k in range(max(0, i-n+1, j-n+1), j):
                s += L[i, i-k] * L[j, j-k]

            L[i, i-j] = (A[i, i-j] - s) / L[j, 0]

    return L



def solve_with_cholesky(L, b):
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

def make_dense_A(A):
    m, n = A.shape
    A_ = np.zeros((m, m))

    for i in range(m):
        for d in range(min(n-1, i)+1):
            j = i-d
            A_[i, j] = A[i, d]
            A_[j, i] = A[i, d]

    return A_

for n in [3, 4, 8]:

    A = make_A_in_band(n)
    L = cholesky_decomposition(A)

    A_ = make_dense_A(A)

    m = (n-1)**2
    b = np.ones(m)

    # 직접 구현한 방법
    u = solve_with_cholesky(L, b)

    # NumPy 정답과 비교
    u_numpy = np.linalg.solve(A_, b)

    # 상대 잔차
    residual = np.linalg.norm(A_ @ u -b) / np.linalg.norm(b)

    # 두 해의 차이
    difference = np.linalg.norm(u-u_numpy) / np.linalg.norm(u_numpy)

    # fill-in 계산
    A_count = np.count_nonzero(np.tril(A_))
    L_count = np.count_nonzero(L)

    print("n =", n)
    print("residual =", residual)
    print("differnce =", difference)
    print("nonzeros in A =", A_count)
    print("nonzeros in L =", L_count)
    print("fill-in =", L_count - A_count)
    print("======================================")