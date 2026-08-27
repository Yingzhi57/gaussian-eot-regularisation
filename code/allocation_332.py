# Gaussian evaluation across source--target allocations and covariance
# geometries. The 21 designs combine three covariance pairs with seven ratios
# n1/n2 at fixed total sample size 240. For each design, independent reference
# datasets estimate the best common multiplier on a dense mesh. Fresh datasets
# compare no regularisation, the local known-basis comparator, the dense-mesh
# reference and RBPB using exact affine population loss. Geometry, data and
# bootstrap streams are independent; failures are recorded rather than repaired.
# Run: python allocation_332.py          (about 35-55 minutes)

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from rbpb_selector import (MULTIPLIERS, make_rng, fit_moments, prep_eig,
                           A_from, pick_smallest, select_epsilon)

# Fixed geometry, reference, evaluation and bootstrap roots.
GEOM_ROOT = 20260825
REF_ROOT, CONF_ROOT, BOOT_ROOT = 20260910, 20260911, 20260912
R_REF = 5000
R_CONFIRM = 5000
R_EXTENDED = 10000
MCSE_GATE = 0.002
M_REF = np.arange(0, 257) / 8.0          # dense mesh on [0, 32]
I24 = int(np.where(M_REF == 24)[0][0])
I32 = 256
GRID_IN_REF = [int(np.where(M_REF == m)[0][0]) for m in MULTIPLIERS]


def D_matrix(kappa):
    # Return a trace-normalised geometric spectrum with condition number kappa.
    v = kappa ** (np.arange(5) / 4.0)
    return np.diag(5.0 * v / v.sum())


Q, _ = np.linalg.qr(make_rng((GEOM_ROOT, 900, 5)).normal(size=(5, 5)))
P_REV = np.eye(5)[::-1]
D4, D16 = D_matrix(4.0), D_matrix(16.0)
GEOMETRIES = [
    ("G4-C", D4, P_REV @ D4 @ P_REV.T),      # commuting control
    ("G4-N", D4, Q @ D4 @ Q.T),              # moderate noncommuting
    ("G4-K", D16, Q @ D16 @ Q.T),            # conditioning stress
]
ALLOCATIONS = [(0.25, 48, 192), (1.0, 120, 120), (3.0, 180, 60),
               (7.0, 210, 30), (9.0, 216, 24), (11.0, 220, 20),
               (15.0, 225, 15)]


def m_kb(rho):
    # Local known-basis multiplier for the estimated-means regime.
    return max(0.0, (11.0 - rho) / (1.0 + rho))


def exact_loss(m1h, m2h, A, Sigma1, A0_true):
    # Exact Gaussian population L2 loss relative to the Brenier map.
    v = m2h - A @ m1h
    D = A - A0_true
    return float(v @ v + np.einsum("ij,jk,ik->", D, Sigma1, D))


def ratio_and_se(Lv, L0v):
    # Paired delta-method MCSE for mean(Lv) / mean(L0v).
    rho = Lv.mean() / L0v.mean()
    infl = Lv - rho * L0v
    return rho, infl.std(ddof=1) / np.sqrt(len(L0v)) / L0v.mean()


def eta(S1, S2):
    return np.linalg.norm(S1 @ S2 - S2 @ S1) \
        / (np.linalg.norm(S1) * np.linalg.norm(S2))


def main():
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("3.3.2 allocation study (frozen 332-G1..G5)")
    emit(f"roots: geometry {GEOM_ROOT}; reference/confirmatory/bootstrap "
         f"{REF_ROOT}/{CONF_ROOT}/{BOOT_ROOT}")
    emit(f"R_ref={R_REF}, R_confirm={R_CONFIRM} (extend to {R_EXTENDED} "
         f"per cell if normalized paired MCSE > {MCSE_GATE})")
    for g, S1t, S2t in GEOMETRIES:
        emit(f"   {g}: cond {np.linalg.cond(S1t):.0f}, "
             f"noncommutativity eta = {eta(S1t, S2t):.6f}")

    ref_store = {"M_REF": M_REF, "Q": Q,
                 "n1": np.array([a[1] for a in ALLOCATIONS]),
                 "n2": np.array([a[2] for a in ALLOCATIONS]),
                 "rho": np.array([a[0] for a in ALLOCATIONS])}
    fieldnames = ["geometry", "alloc_index", "n1", "n2", "rho", "rep",
                  "L0", "L_KB", "L_dense", "L_RBPB",
                  "m_KB", "m_dense", "m_rbpb", "eps_hat", "tau",
                  "zero", "sentinel", "sentinel_falling",
                  "mcse_at_pick", "cond_source", "cond_target",
                  "runtime_s", "B", "boot_seed", "n_warnings", "failure"]
    csv_fh = open("allocation_332_records.csv", "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    for gi, (gname, S1_true, S2_true) in enumerate(GEOMETRIES):
        C1 = np.linalg.cholesky(S1_true)
        C2 = np.linalg.cholesky(S2_true)
        A0_true = A_from(prep_eig(S1_true[None], S2_true[None]), 0.0)[0]
        ref_store[f"{gname}_Sigma1"] = S1_true
        ref_store[f"{gname}_Sigma2"] = S2_true
        ref_store[f"{gname}_A0"] = A0_true

        for ai, (rho, n1, n2) in enumerate(ALLOCATIONS):
            cell = f"{gname} | rho={rho:g} ({n1},{n2})"
            emit(f"\n-- {cell} --")

            # Estimate the design-level risk curve on the dense mesh.
            t0 = time.perf_counter()
            ref_losses = np.full((R_REF, len(M_REF)), np.nan)
            ref_failures = 0
            for rep in range(R_REF):
                try:
                    X = make_rng((REF_ROOT, gi, ai, rep, 0)).normal(
                        size=(n1, 5)) @ C1.T
                    Y = make_rng((REF_ROOT, gi, ai, rep, 1)).normal(
                        size=(n2, 5)) @ C2.T
                    m1h, m2h, S1, S2, _, _, _, tau = fit_moments(X, Y)
                    prep = prep_eig(S1[None], S2[None])
                    for k, m in enumerate(M_REF):
                        A = A_from(prep, float(m * tau))[0]
                        ref_losses[rep, k] = exact_loss(m1h, m2h, A,
                                                        S1_true, A0_true)
                except (ValueError, np.linalg.LinAlgError):
                    ref_failures += 1      # recorded NA, never repaired
            valid_ref = np.all(np.isfinite(ref_losses), axis=1)
            ref_ok = ref_losses[valid_ref]
            if len(ref_ok) < 2:
                raise RuntimeError(f"{cell}: fewer than two valid "
                                   "reference runs")
            curve = ref_ok.mean(axis=0)
            curve_mcse = ref_ok.std(axis=0, ddof=1) / np.sqrt(len(ref_ok))
            j_dense = pick_smallest(curve, M_REF)
            m_dense = float(M_REF[j_dense])
            j_grid = GRID_IN_REF[pick_smallest(curve[GRID_IN_REF],
                                               MULTIPLIERS)]
            m_grid = float(M_REF[j_grid])
            tail = ref_ok[:, I32] - ref_ok[:, I24]
            falling = tail.mean() < -2 * tail.std(ddof=1) / np.sqrt(len(ref_ok))
            unresolved = (j_dense == I32) or bool(falling)
            emit(f"   reference ({time.perf_counter() - t0:.0f}s): valid "
                 f"{len(ref_ok)}/{R_REF}, failures {ref_failures}; "
                 f"m*_dense = {m_dense:g}, grid best = {m_grid:g}, range "
                 f"{'UNRESOLVED' if unresolved else 'resolved'} "
                 f"(top pick {j_dense == I32}, 24->32 falling "
                 f"{bool(falling)})")
            key = f"{gname}_{ai}"
            ref_store[f"{key}_curve_mean"] = curve
            ref_store[f"{key}_curve_mcse"] = curve_mcse
            ref_store[f"{key}_m_dense"] = m_dense
            ref_store[f"{key}_m_grid"] = m_grid
            ref_store[f"{key}_tail_mean"] = tail.mean()
            ref_store[f"{key}_tail_mcse"] = tail.std(ddof=1) / np.sqrt(len(ref_ok))
            ref_store[f"{key}_reference_failures"] = ref_failures
            ref_store[f"{key}_unresolved"] = unresolved

            # Evaluate all decision rules on fresh datasets.
            t0 = time.perf_counter()
            mkb = m_kb(rho)
            rows, failures = [], 0
            rep, target = 0, R_CONFIRM
            while rep < target:
                base = {"geometry": gname, "alloc_index": ai, "n1": n1,
                        "n2": n2, "rho": rho, "rep": rep,
                        "m_KB": mkb, "m_dense": m_dense,
                        "boot_seed": str((BOOT_ROOT, gi, ai, rep))}
                try:
                    X = make_rng((CONF_ROOT, gi, ai, rep, 0)).normal(
                        size=(n1, 5)) @ C1.T
                    Y = make_rng((CONF_ROOT, gi, ai, rep, 1)).normal(
                        size=(n2, 5)) @ C2.T
                    res = select_epsilon(X, Y, "sim",
                                         (BOOT_ROOT, gi, ai, rep))
                    m1h, m2h, S1, S2, A0h, _, _, tau = fit_moments(X, Y)
                    prep = prep_eig(S1[None], S2[None])
                    row = dict(base,
                               L0=exact_loss(m1h, m2h, A0h, S1_true,
                                             A0_true),
                               L_KB=exact_loss(
                                   m1h, m2h,
                                   A_from(prep, float(mkb * tau))[0],
                                   S1_true, A0_true),
                               L_dense=exact_loss(
                                   m1h, m2h,
                                   A_from(prep, float(m_dense * tau))[0],
                                   S1_true, A0_true),
                               L_RBPB=exact_loss(m1h, m2h, res["A"],
                                                 S1_true, A0_true),
                               m_rbpb=res["multiplier"],
                               eps_hat=res["eps"], tau=res["tau"],
                               zero=res["zero_selected"],
                               sentinel=res["sentinel_selected"],
                               sentinel_falling=res["sentinel_falling"],
                               mcse_at_pick=res["mcse_at_pick"],
                               cond_source=res["cond_source"],
                               cond_target=res["cond_target"],
                               runtime_s=res["runtime_s"], B=res["B"],
                               n_warnings=len(res["warnings"]),
                               failure="")
                    rows.append(row)
                except (ValueError, np.linalg.LinAlgError) as err:
                    failures += 1
                    rows.append(dict(base, failure=str(err)))
                rep += 1
                if rep == R_CONFIRM and target == R_CONFIRM:
                    ok = [r for r in rows if r["failure"] == ""]
                    L0v = np.array([r["L0"] for r in ok])
                    dif = np.array([r["L_RBPB"] for r in ok]) - L0v
                    crit = dif.std(ddof=1) / np.sqrt(len(dif)) / L0v.mean()
                    if crit > MCSE_GATE:
                        target = R_EXTENDED
                        emit(f"   extension: normalized paired MCSE "
                             f"{crit:.4f} > {MCSE_GATE} -> "
                             f"R_confirm = {R_EXTENDED}")
            writer.writerows(rows)
            csv_fh.flush()

            ok = [r for r in rows if r["failure"] == ""]
            L0v = np.array([r["L0"] for r in ok])
            emit(f"   confirmatory ({time.perf_counter() - t0:.0f}s): "
                 f"{len(ok)} ok, {failures} failures")
            parts = []
            for name, key2 in (("KB", "L_KB"), ("dense", "L_dense"),
                               ("RBPB", "L_RBPB")):
                r, se = ratio_and_se(np.array([x[key2] for x in ok]), L0v)
                parts.append(f"{name} {r:.4f}+/-{se:.4f}")
            emit("   ratios vs m=0: " + "; ".join(parts)
                 + f"   [m_KB = {mkb:g}]")
            U = np.array([r["L_RBPB"] - r["L_dense"] for r in ok])
            gap = U.mean() / L0v.mean()
            gap_se = (U - gap * L0v).std(ddof=1) / np.sqrt(len(ok)) \
                / L0v.mean()
            Lrb = np.array([r["L_RBPB"] for r in ok])
            m_hats = np.array([r["m_rbpb"] for r in ok])
            emit(f"   RBPB gap vs dense {gap:+.4f}+/-{gap_se:.4f}; "
                 f"improve {np.mean(Lrb < L0v):.3f}; zero "
                 f"{np.mean([r['zero'] for r in ok]):.3f}; sentinel "
                 f"{np.mean([r['sentinel'] for r in ok]):.3f}; "
                 f"falling-to-sentinel "
                 f"{np.mean([r['sentinel_falling'] for r in ok]):.3f}")
            emit(f"   m_rbpb median {np.median(m_hats):g}, IQR "
                 f"[{np.quantile(m_hats, .25):g}, "
                 f"{np.quantile(m_hats, .75):g}]")

    csv_fh.close()
    np.savez("allocation_332_reference.npz", **ref_store)
    emit("\ncode hashes:")
    for f in (Path(__file__).name, "rbpb_selector.py"):
        emit(f"   {f}: "
             f"{hashlib.sha256(Path(f).read_bytes()).hexdigest()}")
    with open("allocation_332_report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwritten: allocation_332_report.txt, allocation_332_records.csv, "
          "allocation_332_reference.npz")


if __name__ == "__main__":
    main()
