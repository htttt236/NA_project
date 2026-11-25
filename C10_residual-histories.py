import methods
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)
n = 64
m = (n-1)**2
b = rng.uniform(low=0, high=50, size=m)

u, converged_jacobi, history_jacobi  = methods.jacobi(b, n)
u, converged_gs, history_gs = methods.gs_rb(b, n)
u, converged_sor, history_sor= methods.sor_rb(b, n)
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

fig, ax = plt.subplots(1, 2, figsize=(10,3.5))

for name, history in histories.items():
    iterations = range(len(history))
    ax[0].semilogy(iterations, history, line[name], label=name)

ax[0].set_xlabel("iteration")
ax[0].set_ylabel("relative residual")
ax[0].legend()

for name in ["SOR", "CG", "PCG"]:
    ax[1].semilogy(range(len(histories[name])), histories[name], line[name], label=name)

ax[1].legend()
fig.savefig('fig_C10.png', dpi=300)
plt.show()