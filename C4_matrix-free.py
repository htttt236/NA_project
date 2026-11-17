import numpy as np

rng = np.random.default_rng(seed=0)


def check_n_size(n):
    if n<3:
        raise ValueError(f"error: too small n. (n: {n})")

def apply_A(u, n):
    check_n_size(n)

    reshaped_u = u.reshape(n-1,n-1)         # (n-1) by (n-1) reshape
    padded_u = np.pad(reshaped_u, 1)        # 0-padding
    b = np.array([])
    #print(padded_u)

    for i in range(1, n):
        for j in range(1, n):
            b = np.append(b, 4*padded_u[i][j] - padded_u[i-1][j]
                          - padded_u[i+1][j] - padded_u[i][j-1] - padded_u[i][j+1])
    return b

def apply_dense_A(u, n):
    check_n_size(n)

    m = (n-1)**2
    I = np.eye(n-1)

    # make T
    I_up = np.roll(I, -1, axis=0)
    I_down = np.roll(I, 1, axis=0)
    I_up[n-2][0]   = 0
    I_down[0][n-2] = 0

    T = 4*I - I_up - I_down
    #print(T)

    # make A
    A = np.zeros((m, m))
    for i in range(n-1):
        for j in range(n-1):
            if i==j:
                A[(n-1)*i:(n-1)*i+n-1, (n-1)*j : (n-1)*j+n-1] = T
            elif abs(i-j) == 1:
                A[(n-1)*i:(n-1)*i+n-1, (n-1)*j : (n-1)*j+n-1] = -I
    #print(A)

    return A @ u


for i in range(3):
    n = rng.integers(low=3, high=8, size=1).item()
    u = rng.integers(low=0, high=10, size=(n-1)**2)
    print("u:")
    print(u)
    print("matrix-free method:")
    print(apply_A(u, n))
    print("dense matrix method:")
    print(apply_dense_A(u, n))