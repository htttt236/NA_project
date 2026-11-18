import numpy as np

#============================== C4_matrix-free ===============================
def _check_n_size(n):
    if n<3:
        raise ValueError(f"error: too small n. (n: {n})")

def _make_A(n):
    # m = (n-1)**2
    # I = np.eye(n-1)

    # # make T
    # I_up = np.roll(I, -1, axis=0)
    # I_down = np.roll(I, 1, axis=0)
    # I_up[n-2, 0]   = 0
    # I_down[0, n-2] = 0

    # T = 4*I - I_up - I_down

    # # make A
    # A = np.zeros((m, m))
    # for i in range(n-1):
    #     for j in range(n-1):
    #         if i==j:
    #             A[(n-1)*i:(n-1)*i+n-1, (n-1)*j : (n-1)*j+n-1] = T
    #         elif abs(i-j) == 1:
    #             A[(n-1)*i:(n-1)*i+n-1, (n-1)*j : (n-1)*j+n-1] = -I

    I = np.eye(n-1)
    T_1d = 2 * I - np.eye((n-1), k=1) - np.eye((n-1), k=-1)
    A = np.kron(I, T_1d) + np.kron(T_1d, I)

def apply_A(u, n):
    _check_n_size(n)

    reshaped_u = u.reshape(n-1,n-1)         # (n-1) by (n-1) reshape
    padded_u = np.pad(reshaped_u, 1)        # 0-padding
    b = np.array([])

    for i in range(1, n):
        for j in range(1, n):
            b = np.append(b, 4*padded_u[i, j] - padded_u[i-1, j]
                          - padded_u[i+1, j] - padded_u[i, j-1] - padded_u[i, j+1])
    return b

def apply_dense_A(u, n):
    _check_n_size(n)

    return _make_A(n) @ u

#============================== C5_banded-cholesky ===============================