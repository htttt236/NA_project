import methods
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)
n = 32
m = (n-1)**2
b = rng.uniform(low=0, high=50, size=m)
w_step = 1 / 32
w = 1.0
iterations = []

while w < 2.0:
    u, converged, history = methods.sor(b, n, w=w)
    iterations.append(len(history) - 1)

    w_new = w + w_step
    if 
