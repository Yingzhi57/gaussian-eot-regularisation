# RBPB decision rule for the full-matrix Gaussian plug-in transport map.
# The rule estimates fitted-Gaussian risk by Rao--Blackwellised parametric
# bootstrap and selects a multiplier from a fixed grid that includes zero.
# Common bootstrap draws are used across candidates, and numerical ties are
# resolved in favour of the smallest multiplier.
# Run ``python rbpb_selector.py`` to execute the implementation checks.

import time
import numpy as np

MULTIPLIERS = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        16, 18, 24], dtype=float)
SENTINEL = 24.0
B_SIM = 250       # bootstrap draws used in the report simulations
B_APP = 4000      # optional higher-budget calls
CHUNK = 250       # bootstrap draws processed per block


def make_rng(seed):
    # Build an independent random stream from an integer tuple.
    return np.random.default_rng(np.random.SeedSequence(list(seed)))


def sqrt_and_inv(M, name="matrix"):
    # Return the symmetric square root and inverse square root of an SPD matrix.
    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    w, V = np.linalg.eigh(M)
    if w.min() <= 0:
        raise ValueError(f"{name} is not positive definite")
    Vt = np.swapaxes(V, -1, -2)
    return (V * np.sqrt(w)[..., None, :]) @ Vt, \
           (V / np.sqrt(w)[..., None, :]) @ Vt


def sample_wishart(d, nu, B, rng):
    # Draw W_d(I, nu) matrices by the Bartlett construction.
    if nu < d:
        raise ValueError(f"need nu >= d for an SPD Wishart (nu={nu}, d={d})")
    T = np.zeros((B, d, d))
    for i in range(d):
        T[:, i, i] = np.sqrt(rng.chisquare(nu - i, size=B))
        if i:
            T[:, i, :i] = rng.normal(size=(B, i))
    return T @ np.swapaxes(T, -1, -2)


def sample_cov(Sigma, nu, B, rng):
    # Draw bootstrap covariances satisfying nu * Sigma_hat ~ W_d(Sigma, nu).
    R, _ = sqrt_and_inv(Sigma, "fitted covariance")
    return (R @ sample_wishart(Sigma.shape[0], nu, B, rng) @ R) / nu


def prep_eig(S1, S2):
    # Precompute the eigendecomposition shared by all epsilon candidates.
    sq, isq = sqrt_and_inv(S1, "source covariance")
    C = sq @ S2 @ sq
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    w, V = np.linalg.eigh(C)
    # Clip round-off-scale negative eigenvalues and reject material negatives.
    if w.min() < -1e-12 * max(float(np.abs(w).max()), 1.0):
        raise ValueError("covariance product materially negative")
    return isq, np.where(w < 0, 0.0, w), V


def A_from(prep, eps):
    # Evaluate A_epsilon with a rationalised form that avoids cancellation.
    isq, w, V = prep
    c = eps / 4.0
    coeff = np.sqrt(w) if eps == 0 else w / (np.sqrt(w + c * c) + c)
    mid = (V * coeff[..., None, :]) @ np.swapaxes(V, -1, -2)
    return isq @ mid @ isq


def criterion_matrix(S1, S2, A0, n1, n2, nu1, nu2, eps_list, B, rng):
    # Return Rao--Blackwellised losses: candidates by bootstrap draw.
    d = S1.shape[0]
    if nu1 <= d + 3:
        raise ValueError(f"need n1 >= d+5 for the paired MCSE (n1={n1})")
    if nu2 < d:
        raise ValueError(f"need n2 >= d+1 for an SPD target draw (n2={n2})")
    out = np.empty((len(eps_list), B))
    mean_term = np.trace(S2) / n2
    for lo in range(0, B, CHUNK):
        hi = min(lo + CHUNK, B)
        S1b = sample_cov(S1, nu1, hi - lo, rng)
        S2b = sample_cov(S2, nu2, hi - lo, rng)
        prep = prep_eig(S1b, S2b)
        for k, eps in enumerate(eps_list):
            A = A_from(prep, float(eps))
            D = A - A0
            out[k, lo:hi] = (mean_term
                             + np.einsum("bij,jk,bik->b", A, S1, A) / n1
                             + np.einsum("bij,jk,bik->b", D, S1, D))
    return out


def fit_moments(X, Y):
    # Fit sample means and unbiased sample covariances.
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    m1, m2 = X.mean(axis=0), Y.mean(axis=0)
    S1 = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    S2 = np.atleast_2d(np.cov(Y, rowvar=False, ddof=1))
    n1, n2 = len(X), len(Y)
    A0 = A_from(prep_eig(S1[None], S2[None]), 0.0)[0]
    tau = (np.trace(A0) / np.trace(np.linalg.inv(S1))) * (1 / n1 + 1 / n2)
    return m1, m2, S1, S2, A0, n1, n2, tau


def pick_smallest(q, multipliers):
    # Select the smallest multiplier within the numerical tie tolerance.
    tied = np.flatnonzero(q <= q.min() + 1e-12 * np.abs(q).max())
    return int(tied[np.argmin(multipliers[tied])])


def select_epsilon(X, Y, purpose, seed):
    # Apply the RBPB decision rule to one source--target dataset.
    if purpose not in ("sim", "app"):
        raise ValueError("purpose must be 'sim' or 'app'")
    B = B_SIM if purpose == "sim" else B_APP
    t0 = time.perf_counter()
    m1, m2, S1, S2, A0, n1, n2, tau = fit_moments(X, Y)
    eps = tau * MULTIPLIERS
    L = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1, eps, B,
                         make_rng(seed))

    q = L.mean(axis=1)
    delta = (L - L[0]).mean(axis=1)
    mcse = (L - L[0]).std(axis=1, ddof=1) / np.sqrt(B)

    j = pick_smallest(q, MULTIPLIERS)

    # Flag selection at the upper grid point and a still-decreasing upper tail.
    tail = L[-1] - L[-2]
    falling = tail.mean() < -2 * tail.std(ddof=1) / np.sqrt(B)
    warnings = []
    if MULTIPLIERS[j] == SENTINEL:
        warnings.append("sentinel 24 selected: coverage warning, "
                        "grid is not extended automatically")
    if falling:
        warnings.append("criterion still falling into the sentinel")

    A_hat = A_from(prep_eig(S1[None], S2[None]), float(eps[j]))[0]
    return {"multiplier": float(MULTIPLIERS[j]), "eps": float(eps[j]),
            "tau": tau, "grid": MULTIPLIERS.copy(),
            "q": q, "delta": delta, "mcse": mcse,
            "mcse_at_pick": float(mcse[j]),
            "zero_selected": bool(MULTIPLIERS[j] == 0),
            "sentinel_selected": bool(MULTIPLIERS[j] == SENTINEL),
            "sentinel_falling": bool(falling),
            "warnings": warnings,
            "A": A_hat, "m1": m1, "m2": m2, "A0": A0,
            "n1": n1, "n2": n2, "d": S1.shape[0],
            "cond_source": float(np.linalg.cond(S1)),
            "cond_target": float(np.linalg.cond(S2)),
            "B": B, "seed": tuple(seed), "purpose": purpose,
            "runtime_s": time.perf_counter() - t0}


def apply_map(res, X):
    # Evaluate the fitted affine map b(x) = m2 + A(x - m1).
    X = np.atleast_2d(np.asarray(X, float))
    return res["m2"] + (X - res["m1"]) @ res["A"].T


# Implementation checks.

def chi2_nodes(nu, n):
    # Integrate over a chi-square law using mapped Gauss--Legendre quadrature.
    from math import lgamma
    u, w = np.polynomial.legendre.leggauss(n)
    u, w = (u + 1) / 2, w / 2
    x = nu * u / (1 - u)
    logpdf = (nu / 2 - 1) * np.log(x) - x / 2 - (nu / 2) * np.log(2) \
        - lgamma(nu / 2)
    return x, w * nu / (1 - u) ** 2 * np.exp(logpdf)


def run_checks():
    print("rbpb_selector checks")

    # Fixed-grid configuration.
    assert len(MULTIPLIERS) == 18 and 18 in MULTIPLIERS and 24 in MULTIPLIERS
    assert (B_SIM, B_APP) == (250, 4000)
    print("[ok] frozen grid and budgets")

    # Check the Riccati equation and the one-dimensional closed form.
    rng = make_rng((1,))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    S1 = np.diag(np.geomspace(1, 6, 4)); S1 *= 4 / np.trace(S1)
    S2 = Q @ np.diag(np.geomspace(1, 3, 4)) @ Q.T; S2 *= 4 / np.trace(S2)
    for eps in (0.0, 0.7, 5.0):
        A = A_from(prep_eig(S1[None], S2[None]), eps)[0]
        assert np.linalg.norm(A @ S1 @ A + eps / 2 * A - S2) \
            <= 1e-10 * np.linalg.norm(S2)
    a, b, e = 1.3, 0.8, 0.7
    got = A_from(prep_eig(np.array([[[a]]]), np.array([[[b]]])), e)[0, 0, 0]
    assert np.isclose(got, (np.sqrt(a * b + e * e / 16) - e / 4) / a,
                      rtol=1e-12)
    print("[ok] Riccati residual and scalar formula")

    # Compare the Rao--Blackwellised criterion with explicit bootstrap means.
    rng = make_rng((2,))
    X = rng.normal(size=(120, 2)) @ np.linalg.cholesky(
        np.array([[1.3, .2], [.2, .9]])).T
    Y = 0.4 + rng.normal(size=(120, 2)) @ np.linalg.cholesky(
        np.array([[.8, -.1], [-.1, 1.1]])).T
    m1, m2, S1, S2, A0, n1, n2, tau = fit_moments(X, Y)
    Bt = 4000
    S1b = sample_cov(S1, n1 - 1, Bt, make_rng((3,)))
    S2b = sample_cov(S2, n2 - 1, Bt, make_rng((4,)))
    prep = prep_eig(S1b, S2b)
    e1 = make_rng((5,)).normal(size=(Bt, 2)) @ np.linalg.cholesky(S1 / n1).T
    e2 = make_rng((6,)).normal(size=(Bt, 2)) @ np.linalg.cholesky(S2 / n2).T
    for eps in tau * np.array([0.0, 3.0, 18.0]):
        A = A_from(prep, float(eps))
        D = A - A0
        rb = (np.trace(S2) / n2
              + np.einsum("bij,jk,bik->b", A, S1, A) / n1
              + np.einsum("bij,jk,bik->b", D, S1, D))
        u = e2 - np.einsum("bij,bj->bi", A, e1)
        full = (u ** 2).sum(1) + np.einsum("bij,jk,bik->b", D, S1, D)
        diff = full - rb
        assert abs(diff.mean()) <= 4 * diff.std(ddof=1) / np.sqrt(Bt)
    print("[ok] Rao-Blackwell loss formula (paired, 4 MCSE)")

    # Check fixed-seed reproducibility and scale equivariance.
    r1 = select_epsilon(X, Y, purpose="sim", seed=(7, 0))
    r2 = select_epsilon(X, Y, purpose="sim", seed=(7, 0))
    assert r1["multiplier"] == r2["multiplier"]
    assert np.array_equal(r1["q"], r2["q"])
    r4 = select_epsilon(2 * X, 2 * Y, purpose="sim", seed=(7, 0))
    assert r4["multiplier"] == r1["multiplier"]
    assert np.isclose(r4["eps"], 4 * r1["eps"], rtol=1e-9)
    assert np.allclose(apply_map(r4, 2 * X[:5]), 2 * apply_map(r1, X[:5]),
                       rtol=1e-9)
    print("[ok] reproducibility and scale equivariance")

    # Check sample-size requirements and rejection of non-SPD input.
    try:
        select_epsilon(X[:5], Y, purpose="sim", seed=(8, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("small source sample should have raised")
    
    Xs = X.copy()
    Xs[:, 1] = Xs[:, 0]  # singular covariance
    try:
        select_epsilon(Xs, Y, purpose="sim", seed=(8, ))
    except ValueError:
        pass
    else:
        raise AssertionError("singular source covariance should have raised")
    print("[ok] sample-size gates; non-PD input raises, never repaired")

    # Check the one-dimensional criterion against deterministic quadrature.
    rng = make_rng((9,))
    X1 = 1.4 * rng.normal(size=(100, 1))
    Y1 = 0.2 + 0.9 * rng.normal(size=(100, 1))
    m1, m2, S1, S2, A0, n1, n2, tau = fit_moments(X1, Y1)
    Bq = 50_000
    Lq = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1,
                          tau * np.array([0.0, 5.0, 18.0]), Bq,
                          make_rng((10,)))
    x, wx = chi2_nodes(n1 - 1, 240)
    y, wy = chi2_nodes(n2 - 1, 240)
    for k, eps in enumerate(tau * np.array([0.0, 5.0, 18.0])):
        s1 = S1[0, 0] * x[:, None] / (n1 - 1)
        s2 = S2[0, 0] * y[None, :] / (n2 - 1)
        c = eps / 4
        Ag = np.sqrt(s2 / s1) if eps == 0 else s2 / (np.sqrt(s1 * s2 + c * c) + c)
        loss = S2[0, 0] / n2 + Ag ** 2 * S1[0, 0] / n1 \
            + (Ag - A0[0, 0]) ** 2 * S1[0, 0]
        quad = float(wx @ loss @ wy)
        assert abs(Lq[k].mean() - quad) \
            <= 4 * Lq[k].std(ddof=1) / np.sqrt(Bq)
    print("[ok] d=1 quadrature agreement (4 MCSE)")

    # Run end-to-end checks in dimensions one and five.
    res1 = select_epsilon(X1, Y1, purpose="sim", seed=(11,))
    rng = make_rng((12,))
    Q5, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    S = np.diag(np.geomspace(1, 4, 5)); S *= 5 / np.trace(S)
    X5 = rng.normal(size=(100, 5)) @ np.linalg.cholesky(S).T
    Y5 = 0.3 + rng.normal(size=(100, 5)) @ np.linalg.cholesky(Q5 @ S @ Q5.T).T
    res5 = select_epsilon(X5, Y5, purpose="app", seed=(13,))
    for r in (res1, res5):
        assert r["multiplier"] in MULTIPLIERS and np.all(np.isfinite(r["q"]))
    print(f"[ok] end-to-end: d=1 pick {res1['multiplier']:g} "
          f"(B={res1['B']}, {res1['runtime_s']:.2f}s), "
          f"d=5 pick {res5['multiplier']:g} "
          f"(B={res5['B']}, {res5['runtime_s']:.2f}s)")
    print("all checks passed; scripts must import this module, "
          "not re-implement it")


if __name__ == "__main__":
    run_checks()
