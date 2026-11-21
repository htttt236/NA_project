import methods
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)
n = 64
m = (n-1)**2
b = rng.uniform(low=0, high=50, size=m)

u, converged_jacobi, history_jacobi  = methods.jacobi(b, n)
u, converged_gs, history_gs = methods.gs_lex(b, n)
u, converged_sor, history_sor= methods.sor(b, n)
u, converged_cg, history_cg = methods.cg(b, n)
u, converged_pcg, history_pcg = methods.pcg_ssor(b, n)

histories = {
    "Jacobi": history_jacobi,
    "Gauss-Seidel": history_gs,
    "SOR": history_sor,
    "CG": history_cg,
    "PCG": history_pcg,
}

line = {
    "Jacobi": '-',
    "Gauss-Seidel": '--',
    "SOR": '-',
    "CG": '-',
    "PCG": '--',
}

for name, history in histories.items():
    iterations = range(len(history))
    plt.semilogy(iterations, history, line[name], label=name)

plt.xlabel("iteration")
plt.ylabel("relative residual")
plt.legend()
plt.savefig('fig_C10.png', dpi=300)
plt.show()