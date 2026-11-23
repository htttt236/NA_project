import methods
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)

n_list = [8, 16, 32, 64, 128]
jacobi_it = []
gs_it = []
sor_it = []
cg_it = []
pcg_it = []

# print table
print(f"      {'jacobi':>10} {'gs':>10} {'sor':>10} {'cg':>10} {'pcg':>10}")
for n in n_list:
    m = (n-1)**2
    b = rng.uniform(low=0, high=50, size=m)

    print(f"n={n:<3d} ", end="", flush=True)
    u, converged, history_jacobi = methods.jacobi(b, n)     ; jacobi_it.append(len(history_jacobi) - 1)
    print(f"{jacobi_it[-1]:>10d} ", end="", flush=True)
    u, converged, history_gs = methods.gs_lex(b,n)          ; gs_it.append(len(history_gs) - 1)
    print(f"{gs_it[-1]:>10d} ", end="", flush=True)
    u, converged, history_sor = methods.sor(b,n)            ; sor_it.append(len(history_sor) - 1)
    print(f"{sor_it[-1]:>10d} ", end="", flush=True)
    u, converged, history_cg = methods.cg(b,n)              ; cg_it.append(len(history_cg) - 1)
    print(f"{cg_it[-1]:>10d} ", end="", flush=True)
    u, converged, history_pcg = methods.pcg_ssor(b,n)            ; pcg_it.append(len(history_cg) - 1)
    print(f"{pcg_it[-1]:>10d} ", flush=True)

iterations ={
    "Jacobi": jacobi_it,
    "Gauss-Seidel": gs_it,
    "SOR": sor_it,
    "CG": cg_it,
    "PCG": pcg_it
}

line = {
    "Jacobi": 'o-',
    "Gauss-Seidel": '^-',
    "SOR": 'v--',
    "CG": 's-',
    "PCG": 'x--',
}

for name, it in iterations.items():
    plt.loglog(n_list, it, line[name], label=name)

plt.xlabel("n")
plt.ylabel("iteration")
plt.legend()
plt.savefig('fig_C9_1.png', dpi=300)
plt.show()