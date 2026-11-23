import methods
import numpy as np

rng = np.random.default_rng(seed=0)

print("#============================== C4 ===============================")
print(f"{'n':>3}   {'diff(vs dense)':>15}")
for i in range(3):
    n = rng.integers(low=3, high=8, size=1).item()
    m = (n-1)**2
    u = rng.uniform(low=0, high=10, size=m)
    b1 = methods.apply_A(u, n)
    b2 = methods.apply_dense_A(u, n)
    diff = np.linalg.norm(b1-b2) / np.linalg.norm(b2)
    print(f"{n:>3d}   {diff:>15.4e}")

print("#============================== C5 ===============================")
print(f"{'n':>3} {'residual':>10} {'diff(vs numpy)':>15} {'nnz-in-A(low)':>14} {'nnz-in-L':>14} {'fill-in':>14}")
for i in [3,6,9]:
    n = i
    m = (n-1)**2
    b = rng.uniform(low=0, high=10, size=m)
    banded_A = methods.make_banded_A(n)
    banded_L = methods.cholesky_decomposition(banded_A)
    A = methods.make_A(n)
    u1 = methods.solve_banded_cholesky(banded_L, b)
    u2 = np.linalg.solve(A, b)
    residual = np.linalg.norm(A@u1 - b) / np.linalg.norm(b)
    diff = np.linalg.norm(u1-u2) / np.linalg.norm(u2)
    nnz_A = np.count_nonzero(banded_A)
    nnz_L = np.count_nonzero(banded_L)
    fill_in = (nnz_L - nnz_A) * 2
    print(f"{n:>3d} {residual:>10.4e} {diff:>15.4e} {nnz_A:>14d} {nnz_L:>14d} {fill_in:>14d}")

print("#============================== C6 ===============================")
print(f"{'case':>5} {'n':>3} {'method':>8} {'residual':>12} {'diff(vs numpy)':>15} {'it':>10} {'converged':>12}")
for i in range(3):
    n = rng.integers(low=8, high=32, size=1).item()
    m = (n-1)**2
    b = rng.uniform(low=0, high=10, size=m)

    answer = np.linalg.solve(methods.make_A(n), b)
    answer_norm = np.linalg.norm(answer)
    u, converged_jacobi, history_jacobi  = methods.jacobi(b, n);    diff_jacobi = np.linalg.norm(answer - u) / answer_norm
    u, converged_gs, history_gs = methods.gs_lex(b, n);             diff_gs = np.linalg.norm(answer - u) / answer_norm
    u, converged_sor, history_sor= methods.sor(b, n);               diff_sor = np.linalg.norm(answer - u) / answer_norm
    u, converged_cg, history_cg = methods.cg(b, n);                 diff_cg = np.linalg.norm(answer - u) / answer_norm
    print(f"{i+1:>5d}", end="")
    print(f" {n:>3d} {'jacobi':>8} {history_jacobi[-1]:>12.4e} {diff_jacobi:>15.4e} {len(history_jacobi)-1:>10d} {str(converged_jacobi):>12}")
    print(f"          {'gs':>8} {history_gs[-1]:>12.4e} {diff_gs:>15.4e} {len(history_gs)-1:>10d} {str(converged_gs):>12}")
    print(f"          {'sor':>8} {history_sor[-1]:>12.4e} {diff_sor:>15.4e} {len(history_sor)-1:>10d} {str(converged_sor):>12}")
    print(f"          {'cg':>8} {history_cg[-1]:>12.4e} {diff_cg:>15.4e} {len(history_cg)-1:>10d} {str(converged_cg):>12}")

print("#============================== C7 ===============================")
print(f"{'case':>5} {'n':>3} {'method':>8} {'residual':>12} {'it':>10} {'converged':>12}")
for i in range(3):
    n = rng.integers(low=8, high=32, size=1).item()
    m = (n-1)**2
    b = rng.uniform(low=0, high=10, size=m)

    u_cg, converged_cg, history_cg  = methods.cg(b, n)
    u_pcg, converged_pcg, history_pcg = methods.pcg_ssor(b, n)
    diff = np.linalg.norm(u_cg - u_pcg) / np.linalg.norm(u_cg)
    print(f"{i+1:>5d}", end="")
    print(f" {n:>3d} {'cg':>8} {history_cg[-1]:>12.4e} {len(history_cg)-1:>10d} {str(converged_cg):>12}")
    print(f"          {'pcg':>8} {history_pcg[-1]:>12.4e} {len(history_pcg)-1:>10d} {str(converged_pcg):>12}")
    print(f"              pcg-cg-difference: {diff:.4e}")