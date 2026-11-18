"""Set C의 Poisson 문제를 일반 CG와 SSOR-PCG로 푸는 예제."""

import numpy as np


def make_rhs(n, f):
    """내부 격자에서 b[j-1, i-1] = h² f(ih, jh)를 계산한다."""
    h = 1 / n
    points = np.arange(1, n) * h
    return h**2 * np.array([[f(x, y) for x in points] for y in points])


def apply_A(v):
    """A의 원소를 저장하지 않고 5점 차분식으로 Av를 구한다."""
    padded = np.pad(v, 1)
    return (
        4 * v
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
    )


def apply_ssor(r):
    """ω=1인 SSOR에서 Mz=r을 전진 대입과 후진 대입으로 푼다."""
    s = r.shape[0]
    y = np.zeros((s + 2, s + 2))
    z = np.zeros_like(y)

    # 배열의 행 번호 j가 커지는 쪽이 물리적 격자의 위쪽이다.
    # 바깥쪽 한 겹은 경계 이웃을 나타내며 계속 0으로 둔다.
    for j in range(1, s + 1):
        for i in range(1, s + 1):
            y[j, i] = (r[j - 1, i - 1] + y[j, i - 1] + y[j - 1, i]) / 4

    for j in range(s, 0, -1):
        for i in range(s, 0, -1):
            z[j, i] = (4 * y[j, i] + z[j, i + 1] + z[j + 1, i]) / 4

    return z[1:-1, 1:-1]


def solve_poisson(n, f, tol=1e-8, x0=None, *, use_ssor=True, max_iter=10000):
    """격자 위의 해와 반복 횟수를 반환한다. use_ssor=False이면 일반 CG다."""
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n은 2 이상의 정수여야 합니다.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol은 유한한 양수여야 합니다.")

    b = make_rhs(n, f)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != b.shape:
        raise ValueError(f"x0의 크기는 {b.shape}여야 합니다.")

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return np.zeros_like(b), 0
    target = tol * norm_b

    r = b - apply_A(x)
    if np.linalg.norm(r) < target:
        return x, 0

    # 일반 CG에서는 M=I이므로 z=r이다.
    precondition = apply_ssor if use_ssor else np.copy
    z = precondition(r)
    p = z
    rz = np.sum(r * z)

    for iteration in range(1, max_iter + 1):
        q = apply_A(p)
        alpha = rz / np.sum(p * q)
        x = x + alpha * p
        r = r - alpha * q
        if np.linalg.norm(r) < target:
            return x, iteration

        z = precondition(r)
        rz_new = np.sum(r * z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    raise RuntimeError(f"{max_iter}회 안에 상대 잔차 기준에 도달하지 못했습니다.")


def manufactured_f(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


if __name__ == "__main__":
    n = 4
    tol = 1e-8
    x0 = np.zeros((n - 1, n - 1))
    x_cg, steps_cg = solve_poisson(n, manufactured_f, tol, x0, use_ssor=False)
    x_pcg, steps_pcg = solve_poisson(n, manufactured_f, tol, x0)

    b = make_rhs(n, manufactured_f)
    norm_b = np.linalg.norm(b)
    residual_cg = np.linalg.norm(b - apply_A(x_cg)) / norm_b
    residual_pcg = np.linalg.norm(b - apply_A(x_pcg)) / norm_b
    difference = np.linalg.norm(x_pcg - x_cg) / np.linalg.norm(x_cg)

    print(f"n = {n}, 상대 잔차 기준 = {tol:g}")
    print(f"CG        {steps_cg:>2}회, 최종 상대 잔차 {residual_cg:.3e}")
    print(f"SSOR-PCG  {steps_pcg:>2}회, 최종 상대 잔차 {residual_pcg:.3e}")
    print(f"두 최종 해의 상대 차이 {difference:.3e}")
