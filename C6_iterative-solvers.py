import numpy as np

def rhs_manufactured(n):
    h = 1.0 / n
    x = np.arange(1, n) * h
    X, Y = np.meshgrid(x, x)                      # X[j,i] = x_i, Y[j,i] = y_j
    uex = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return h * h * 2 * np.pi ** 2 * uex, uex


def rhs_random(n, seed):
    return np.random.default_rng(seed).standard_normal((n - 1, n - 1)) / n ** 2


def omega_opt(n):
    return 2.0 / (1.0 + np.sin(np.pi / n))


# ---------------------------------------------------------------- C4: matrix-free product
def apply_A(U):
    P = np.pad(U, 1)
    return 4 * U - P[:-2, 1:-1] - P[2:, 1:-1] - P[1:-1, :-2] - P[1:-1, 2:]


def dense_A(n):
    N = n - 1
    T = 4 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    return np.kron(np.eye(N), T) - np.kron(np.eye(N, k=1) + np.eye(N, k=-1), np.eye(N))


# ---------------------------------------------------------------- C5: banded Cholesky
def banded_cholesky(n):
    N = n - 1
    m, b = N * N, N
    W = np.zeros((m, b + 1))
    W[:, 0] = 4.0
    idx = np.arange(m)
    W[idx[idx % N != 0], 1] = -1.0                # A[i, i-1]
    W[N:, b] = -1.0                               # A[i, i-N]
    tri = {}
    for j in range(m):
        q = min(b, m - 1 - j)
        W[j, 0] = np.sqrt(W[j, 0])
        if q == 0:
            break
        d = np.arange(1, q + 1)
        c = W[j + d, d] / W[j, 0]
        W[j + d, d] = c                           # column j of L
        if q not in tri:
            tri[q] = np.tril_indices(q)
        I, K = tri[q]                             # rank-1 update of the trailing band
        W[j + 1 + I, I - K] -= c[I] * c[K]
    nnzL = int(np.count_nonzero(W))
    return W, nnzL


def banded_solve(W, rhs):
    m, b = W.shape[0], W.shape[1] - 1
    y = np.zeros(m)
    for i in range(m):                            # L y = rhs
        q = min(b, i)
        y[i] = (rhs[i] - (W[i, 1:q + 1] @ y[i - q:i][::-1] if q else 0.0)) / W[i, 0]
    x = np.zeros(m)
    for j in range(m - 1, -1, -1):                # L^T x = y
        q = min(b, m - 1 - j)
        d = np.arange(1, q + 1)
        s = (W[j + d, d] @ x[j + 1:j + q + 1]) if q else 0.0
        x[j] = (y[j] - s) / W[j, 0]
    return x


def direct_solve(B):
    n = B.shape[0] + 1
    W, nnzL = banded_cholesky(n)
    return banded_solve(W, B.ravel()).reshape(B.shape), nnzL


# C6----------------------------------------------
def jacobi(B, tol=1e-8, maxit=10 ** 6):
    U = np.zeros_like(B); R = B.copy(); bn = np.linalg.norm(B); hist = [1.0]
    for k in range(1, maxit + 1):
        U += R / 4
        R = B - apply_A(U)
        hist.append(np.linalg.norm(R) / bn)
        if hist[-1] < tol:
            return U, hist, k, True
    return U, hist, maxit, False


def sor_rb(B, omega=1.0, tol=1e-8, maxit=10 ** 6):
    N = B.shape[0]
    I, J = np.indices((N, N)); masks = [(I + J) % 2 == c for c in (0, 1)]
    P = np.zeros((N + 2, N + 2)); inner = P[1:-1, 1:-1]      
    bn = np.linalg.norm(B); hist = [1.0]
    for k in range(1, maxit + 1):
        for msk in masks:                                   
            nb = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]
            inner[msk] = (1 - omega) * inner[msk] + omega * ((B + nb) / 4)[msk]
        hist.append(np.linalg.norm(B - apply_A(inner)) / bn)
        if hist[-1] < tol:
            return inner.copy(), hist, k, True
    return inner.copy(), hist, maxit, False


def sor_lex(B, omega=1.0, tol=1e-8, maxit=10 ** 5):
    N = B.shape[0]; P = np.zeros((N + 2, N + 2)); bn = np.linalg.norm(B); hist = [1.0]
    for k in range(1, maxit + 1):
        for j in range(1, N + 1):
            for i in range(1, N + 1):
                g = (B[j - 1, i - 1] + P[j - 1, i] + P[j + 1, i] + P[j, i - 1] + P[j, i + 1]) / 4
                P[j, i] += omega * (g - P[j, i])
        hist.append(np.linalg.norm(B - apply_A(P[1:-1, 1:-1])) / bn)
        if hist[-1] < tol:
            return P[1:-1, 1:-1].copy(), hist, k, True
    return P[1:-1, 1:-1].copy(), hist, maxit, False


def gauss_seidel(B, **kw):
    return sor_rb(B, 1.0, **kw)


def cg(B, tol=1e-8, maxit=10 ** 5):
    X = np.zeros_like(B); R = B.copy(); Pd = R.copy()
    rs = np.sum(R * R); bn = np.sqrt(np.sum(B * B)); hist = [1.0]
    for k in range(1, maxit + 1):
        AP = apply_A(Pd)
        a = rs / np.sum(Pd * AP)
        X += a * Pd; R -= a * AP
        rs_new = np.sum(R * R)
        hist.append(np.sqrt(rs_new) / bn)
        if hist[-1] < tol:
            return X, hist, k, True
        Pd = R + (rs_new / rs) * Pd; rs = rs_new
    return X, hist, maxit, False


# ---------------------------------------------------------------- verification / demo
def verify(ns=(8, 16, 32), seed=0):
    # C4
    for n in (4, 6, 8):
        U = np.random.default_rng(1).standard_normal((n - 1, n - 1))
        d = np.abs(dense_A(n) @ U.ravel() - apply_A(U).ravel()).max()
        print(f"C4 n={n}: max|dense - stencil| = {d:.1e}")
    # C5 (numpy.linalg.solve is a labelled REFERENCE only)
    for n in (4, 8):
        B = rhs_random(n, seed); u, nnzL = direct_solve(B)
        ref = np.linalg.solve(dense_A(n), B.ravel())
        print(f"C5 n={n}: resid={np.linalg.norm(apply_A(u)-B)/np.linalg.norm(B):.1e}, "
              f"|u-ref|={np.abs(u.ravel()-ref).max():.1e}, "
              f"nnz(A)={np.count_nonzero(dense_A(n))}, nnz(L)={nnzL}")
# C6
    print(f"\n{'n':>4} {'method':<12}{'iters':>7}  true_res   err_vs_direct  bound=cond*tol  ok")
    for n in ns:
        B = rhs_random(n, seed); ud, _ = direct_solve(B)
        c = (4 + 4 * np.cos(np.pi / n)) / (4 - 4 * np.cos(np.pi / n))
        runs = {"Jacobi": lambda: jacobi(B), "GS (RB)": lambda: gauss_seidel(B),
                f"SOR w={omega_opt(n):.3f}": lambda: sor_rb(B, omega_opt(n)), "CG": lambda: cg(B)}
        for name, f in runs.items():
            U, hist, it, ok = f()
            tr = np.linalg.norm(B - apply_A(U)) / np.linalg.norm(B)
            er = np.linalg.norm(U - ud) / np.linalg.norm(ud)
            print(f"{n:>4} {name:<12}{it:>7}  {tr:.2e}   {er:.2e}      {c*1e-8:.1e}     {ok and er <= c*1e-8}")


verify()