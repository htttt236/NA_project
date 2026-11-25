import numpy as np

#============================== C4: matrix-free ===============================
def _check_n_size(n):
    if n<3:
        raise ValueError(f"error: too small n. (n: {n})")

def make_A(n):
    I = np.eye(n-1)
    T_1d = 2 * I - np.eye((n-1), k=1) - np.eye((n-1), k=-1)
    return np.kron(I, T_1d) + np.kron(T_1d, I)

def apply_A(u, n):
    _check_n_size(n)

    reshaped_u = u.reshape(n-1,n-1)         # (n-1) by (n-1) reshape
    padded_u = np.pad(reshaped_u, 1)        # 0-padding
    return (
            4 * reshaped_u
            - padded_u[1:-1, :-2]   #left
            - padded_u[1:-1, 2:]    #right
            - padded_u[:-2, 1:-1]   #up
            - padded_u[2:, 1:-1]    #down
        ).reshape(-1)

def apply_dense_A(u, n):
    _check_n_size(n)

    return make_A(n) @ u

#============================== C5: banded-cholesky ===============================

def make_banded_A(n):               # A의 대각 아래 band만 생성
    n_ = n -1
    m = n_ * n_
    A = np.zeros((m, n))

    for r in range(n_):
        for c in range(n_):
            i = r*n_ + c
            A[i,0] = 4

            if c>0:
                A[i,1] = -1
            if r>0:
                A[i,n_] = -1
    return A

def cholesky_decomposition(banded_A):
    m, n = banded_A.shape
    L = np.zeros_like(banded_A)

    for j in range(m):
        # 대각 원소 계산
        s = 0
        for k in range(max(0, j-n+1), j):
            s += L[j, j-k]**2

        L[j, 0] = np.sqrt(banded_A[j, 0] - s)

        # 대각선 아래 원소 계산
        for i in range(j+1, min(m, j+n)):

            s = 0
            for k in range(max(0, i-n+1, j-n+1), j):
                s += L[i, i-k] * L[j, j-k]

            L[i, i-j] = (banded_A[i, i-j] - s) / L[j, 0]

    return L

def solve_banded_cholesky(L, b):
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

#============================== C6: iterative-solvers ===============================

def jacobi(b, n, tol=1e-8, maxit=10**6):
    def apply_N(u, n):   
        reshaped_u = u.reshape(n-1, n-1)
        padded_u = np.pad(reshaped_u, 1)
        return (
            padded_u[1:-1, :-2]     #left
            + padded_u[1:-1, 2:]    #right
            + padded_u[:-2, 1:-1]   #up
            + padded_u[2:, 1:-1]    #down
        ).reshape(-1)

    _check_n_size(n)

    u = np.zeros_like(b)
    history = [1.0]
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return u, True, [0.0]

    for i in range(maxit):
        u = 0.25 * (apply_N(u, n) + b)
        residual = np.linalg.norm(apply_A(u, n) - b) / b_norm
        history.append(residual)

        if residual < tol:
            return u, True, history

    return u, False, history


def sor_lex(b, n, w=None, tol=1e-8, maxit=10**6):
    _check_n_size(n)
    if w is None:
        w = 2.0 / (1.0 + np.sin(np.pi/n))

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros_like(b), True, [0.0]
    history = [1.0]
    b = b.reshape(n-1,n-1)
    u = np.zeros_like(b).reshape(n-1,n-1)
    u = np.pad(u, 1)

    for k in range(maxit):
        for j in range(1, n):
            for i in range(1, n):
                u[j,i] = (1-w)*u[j, i] + w*0.25 * (
                    b[j-1,i-1]          # b is not padded
                    + u[j-1,i]          # up(new)
                    + u[j+1,i]          # down(old)
                    + u[j,i-1]          # left(new)
                    + u[j,i+1])         # right(old)

        residual = np.linalg.norm(apply_A(u[1:-1, 1:-1].reshape(-1), n) - b.reshape(-1)) / b_norm
        history.append(residual)
        if residual < tol:
            return u[1:-1, 1:-1].reshape(-1), True, history
        
    return u[1:-1, 1:-1].reshape(-1), False, history

def gs_lex(b, n, tol=1e-8, maxit=10**6):
    return sor_lex(b, n, w=1.0, tol=tol, maxit=maxit)

def sor_rb(b, n, w=None, tol=1e-8, maxit=10**6):
    _check_n_size(n)
    if w is None:
        w = 2.0 / (1.0 + np.sin(np.pi/n))

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros_like(b), True, [0.0]
    history = [1.0]
    b = b.reshape(n-1,n-1)
    u = np.zeros_like(b).reshape(n-1,n-1)
    u = np.pad(u, 1)

    for i in range(maxit):
        # update red
        u[1:-1:2, 1:-1:2] = (1-w)*u[1:-1:2, 1:-1:2] + w*0.25*(
            b[::2, ::2]             # b is not padded
            + u[:-2:2, 1:-1:2]      # up
            + u[2::2, 1:-1:2]       # down
            + u[1:-1:2, :-2:2]      # left
            + u[1:-1:2, 2::2]       # right
        )
        u[2:-1:2, 2:-1:2] = (1-w)*u[2:-1:2, 2:-1:2] + w*0.25*(
            b[1::2, 1::2]       # b is not padded
            + u[1:-2:2, 2:-1:2]     # up
            + u[3::2, 2:-1:2]     # down
            + u[2:-1:2, 1:-2:2]     # left
            + u[2:-1:2, 3::2]     # right
        )
        # update black
        u[1:-1:2, 2:-1:2] = (1-w)*u[1:-1:2, 2:-1:2] + w*0.25*(
            b[::2, 1::2]          # b is not padded
            + u[:-2:2, 2:-1:2]      # up
            + u[2::2, 2:-1:2]       # down
            + u[1:-1:2, 1:-2:2]     # left
            + u[1:-1:2, 3::2]     # right
        )
        u[2:-1:2, 1:-1:2] = (1-w)*u[2:-1:2, 1:-1:2] + w*0.25*(
            b[1::2, ::2]          # b is not padded
            + u[1:-2:2, 1:-1:2]     # up
            + u[3::2, 1:-1:2]     # down
            + u[2:-1:2, :-2:2]      # left
            + u[2:-1:2, 2::2]       # right
        )

        residual = np.linalg.norm(apply_A(u[1:-1, 1:-1].reshape(-1), n) - b.reshape(-1)) / b_norm
        history.append(residual)
        if residual < tol:
            return u[1:-1, 1:-1].reshape(-1), True, history

    return u[1:-1, 1:-1].reshape(-1), False, history

def gs_rb(b, n, tol=1e-8, maxit=10**6):
    return sor_rb(b, n, w=1.0, tol=tol, maxit=maxit)

def cg(b, n, tol=1e-8, maxit=10**6):
    _check_n_size(n)

    u = np.zeros_like(b)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return u, True, [0.0]
    history = [1.0]

    r = b - apply_A(u, n)
    p = np.copy(r)
    rho = r @ r                 # r^T r

    for i in range(maxit):
        q = apply_A(p, n)
        alpha = rho / (p@q)
        u = u + alpha*p
        r = r - alpha*q
        rho_new = r @ r
        residual = np.linalg.norm(r) / b_norm
        history.append(residual)
        if residual < tol:
            return u, True, history
        p = r + rho_new/rho * p
        rho = rho_new

    return u, False, history

#============================== C7: preconditioned-cg ===============================

def pcg_ssor(b, n, w=1.0, tol=1e-8, maxit=10**6):
    def solve_lu_ssor(b, n, w):
        m = (n-1)**2
        y = np.zeros_like(b)
        for i in range(m):      # Ly = b 풀기
            if i < (n-1):
                y_up = 0
            else:
                y_up = y[i-(n-1)]
            if (i % (n-1)) == 0:
                y_left = 0
            else:
                y_left = y[i-1]
            y[i] = 0.25 * (b[i] + w*(y_up + y_left))

        x = np.zeros_like(b)
        for i in range(m):      # Ux = y 풀기
            idx = -(i+1)
            if i < (n-1):
                x_down = 0
            else:
                x_down = x[idx+n-1]
            if (i % (n-1)) == 0:
                x_right = 0
            else:
                x_right = x[idx+1]
            x[idx] = 0.25 * (y[idx] + w*(x_down + x_right))

        return x * 4*w*(2-w)

    _check_n_size(n)

    u = np.zeros_like(b)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return u, True, [0.0]
    history = [1.0]
    r = b - apply_A(u, n)
    z = solve_lu_ssor(r, n , w)
    p = np.copy(z)

    for i in range(maxit):
        Ap = apply_A(p, n)
        alpha = (r@z) / (p@Ap)
        u = u + alpha*p
        r_new = r - alpha*Ap
        z_new = solve_lu_ssor(r_new, n, w)
        beta = (r_new@z_new) / (r@z)
        p = z_new + beta*p

        r = r_new
        z = z_new
        residual = np.linalg.norm(r) / b_norm
        history.append(residual)
        if residual < tol:
            return u, True, history

    return u, False, history