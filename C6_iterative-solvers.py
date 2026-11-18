import numpy as np
import methods


def jacobi(B, tol=1e-8, maxit=10 ** 6):
    U = np.zeros_like(B); R = B.copy(); bn = np.linalg.norm(B); hist = [1.0]
    for k in range(1, maxit + 1):
        U += R / 4
        R = B - methods.apply_A(U)
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
        hist.append(np.linalg.norm(B - methods.methods.apply_A(inner)) / bn)
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
        hist.append(np.linalg.norm(B - methods.apply_A(P[1:-1, 1:-1])) / bn)
        if hist[-1] < tol:
            return P[1:-1, 1:-1].copy(), hist, k, True
    return P[1:-1, 1:-1].copy(), hist, maxit, False


def gauss_seidel(B, **kw):
    return sor_rb(B, 1.0, **kw)


def cg(B, tol=1e-8, maxit=10 ** 5):
    X = np.zeros_like(B); R = B.copy(); Pd = R.copy()
    rs = np.sum(R * R); bn = np.sqrt(np.sum(B * B)); hist = [1.0]
    for k in range(1, maxit + 1):
        AP = methods.apply_A(Pd)
        a = rs / np.sum(Pd * AP)
        X += a * Pd; R -= a * AP
        rs_new = np.sum(R * R)
        hist.append(np.sqrt(rs_new) / bn)
        if hist[-1] < tol:
            return X, hist, k, True
        Pd = R + (rs_new / rs) * Pd; rs = rs_new
    return X, hist, maxit, False
