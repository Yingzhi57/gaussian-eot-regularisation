# Supplementary sensitivity analysis under covariance-matched multivariate
# Student-t sampling. It applies the unchanged Gaussian RBPB rule to t5 and t3
# samples under one covariance geometry and two allocations. The common radial
# family keeps the affine Brenier map known, so exact population loss remains
# available. Independent reference and evaluation datasets are used, and
# failures are recorded rather than repaired. The t3 analysis uses quantile
# summaries because fourth moments, and hence variance-based MCSE conditions,
# are unavailable.
# Run: python heavy_333.py          (about 10-14 minutes)

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from rbpb_selector import (make_rng, fit_moments, prep_eig, A_from,
                           pick_smallest, select_epsilon)

# Fixed roots. Stage index 0 denotes reference and 1 denotes evaluation.
GEOM_ROOT = 20260825
REF_DATA_ROOT, CONFIRM_DATA_ROOT, INNER_BOOT_ROOT = 20260920, 20260921, 20260922
R_REF = 5000
R_CONFIRM = 10000
M_REF = np.arange(0, 257) / 8.0
I24 = int(np.where(M_REF == 24)[0][0])
I32 = 256
DISTRIBUTIONS = [(0, "t5", 5.0), (1, "t3", 3.0)]
ALLOCATIONS = [(0, 0.25, 48, 192), (1, 9.0, 216, 24)]


def D_matrix(kappa):
    v = kappa ** (np.arange(5) / 4.0)
    return np.diag(5.0 * v / v.sum())


Q, _ = np.linalg.qr(make_rng((GEOM_ROOT, 900, 5)).normal(size=(5, 5)))
S1_TRUE = D_matrix(4.0)
S2_TRUE = Q @ D_matrix(4.0) @ Q.T
C1 = np.linalg.cholesky(S1_TRUE)
C2 = np.linalg.cholesky(S2_TRUE)


def t_sample(n, nu, C, rng):
    # Draw covariance-matched multivariate t_nu observations.
    Z = rng.standard_normal((n, 5))
    W = rng.chisquare(df=nu, size=n)
    return np.sqrt((nu - 2.0) / W)[:, None] * (Z @ C.T)


def exact_loss(m1h, m2h, A, Sigma1, A0_true):
    # Exact population L2 loss for an affine map with zero population means.
    v = m2h - A @ m1h
    D = A - A0_true
    return float(v @ v + np.einsum("ij,jk,ik->", D, Sigma1, D))


def main():
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("3.3.3 heavy-tail sensitivity of the frozen Gaussian RBPB")
    emit(f"geometry: 3.3.2 G4-N pair (root {GEOM_ROOT}); data roots "
         f"{REF_DATA_ROOT}/{CONFIRM_DATA_ROOT}; inner boot "
         f"{INNER_BOOT_ROOT} with stage index")
    emit(f"R_ref={R_REF}, R_confirm={R_CONFIRM} (fixed, no extension gate)")

    A0_true = A_from(prep_eig(S1_TRUE[None], S2_TRUE[None]), 0.0)[0]
    ref_store = {"M_REF": M_REF, "Q": Q, "Sigma1_true": S1_TRUE,
                 "Sigma2_true": S2_TRUE, "A0_true": A0_true}
    fieldnames = ["distribution", "nu", "alloc_index", "n1", "n2", "rho",
                  "rep", "L0", "L_dense", "L_RBPB",
                  "m_dense", "m_rbpb", "eps_hat", "tau",
                  "zero", "sentinel", "sentinel_falling", "mcse_at_pick",
                  "cond_source", "cond_target", "runtime_s", "B",
                  "boot_seed", "n_warnings", "failure"]
    csv_fh = open("heavy_333_records.csv", "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    for di, dname, nu in DISTRIBUTIONS:
        for ai, rho, n1, n2 in ALLOCATIONS:
            cell = f"{dname} | rho={rho:g} ({n1},{n2})"
            emit(f"\n-- {cell} --")

            # Estimate the dense-mesh loss curves; retain failures as missing.
            t0 = time.perf_counter()
            ref_losses = np.full((R_REF, len(M_REF)), np.nan)
            ref_failures = 0
            for rep in range(R_REF):
                try:
                    X = t_sample(n1, nu, C1,
                                 make_rng((REF_DATA_ROOT, di, ai, rep, 0)))
                    Y = t_sample(n2, nu, C2,
                                 make_rng((REF_DATA_ROOT, di, ai, rep, 1)))
                    m1h, m2h, S1, S2, _, _, _, tau = fit_moments(X, Y)
                    prep = prep_eig(S1[None], S2[None])
                    for k, m in enumerate(M_REF):
                        A = A_from(prep, float(m * tau))[0]
                        ref_losses[rep, k] = exact_loss(m1h, m2h, A,
                                                        S1_TRUE, A0_true)
                except (ValueError, np.linalg.LinAlgError):
                    ref_failures += 1      # recorded NA, never repaired
            valid_mask = np.all(np.isfinite(ref_losses), axis=1)
            ok = ref_losses[valid_mask]
            ref_failures = int(R_REF - valid_mask.sum())
            if len(ok) < 2:
                raise RuntimeError(f"{cell}: fewer than two valid "
                                   "reference runs")
            mean_curve = ok.mean(axis=0)
            ratio = ok / ok[:, :1]
            med_curve = np.median(ratio, axis=0)
            j_mean = pick_smallest(mean_curve, M_REF)
            j_med = pick_smallest(med_curve, M_REF)
            # Use mean loss for t5 and median loss ratio for t3.
            j_sel = j_mean if nu > 4 else j_med
            m_dense = float(M_REF[j_sel])
            curve_sel = mean_curve if nu > 4 else med_curve
            falling = bool(curve_sel[I32] < curve_sel[I24])
            unresolved = (j_sel == I32) or falling
            emit(f"   reference ({time.perf_counter() - t0:.0f}s): valid "
                 f"{len(ok)}/{R_REF}, failures {ref_failures}; "
                 f"m* = {m_dense:g} ({'mean' if nu > 4 else 'median-ratio'}"
                 f" rule; mean-rule {M_REF[j_mean]:g}, median-rule "
                 f"{M_REF[j_med]:g}); range "
                 f"{'UNRESOLVED' if unresolved else 'resolved'}")
            key = f"{dname}_{ai}"
            ref_store[f"{key}_mean_curve"] = mean_curve
            ref_store[f"{key}_median_ratio_curve"] = med_curve
            ref_store[f"{key}_m_dense"] = m_dense
            ref_store[f"{key}_m_mean_rule"] = float(M_REF[j_mean])
            ref_store[f"{key}_m_median_rule"] = float(M_REF[j_med])
            ref_store[f"{key}_unresolved"] = unresolved
            ref_store[f"{key}_reference_failures"] = ref_failures

            # Evaluate the fixed reference and RBPB on 10,000 fresh datasets.
            t0 = time.perf_counter()
            rows, failures = [], 0
            for rep in range(R_CONFIRM):
                base = {"distribution": dname, "nu": nu, "alloc_index": ai,
                        "n1": n1, "n2": n2, "rho": rho, "rep": rep,
                        "m_dense": m_dense,
                        "boot_seed": str((INNER_BOOT_ROOT, 1, di, ai, rep))}
                try:
                    X = t_sample(n1, nu, C1,
                                 make_rng((CONFIRM_DATA_ROOT, di, ai, rep, 0)))
                    Y = t_sample(n2, nu, C2,
                                 make_rng((CONFIRM_DATA_ROOT, di, ai, rep, 1)))
                    res = select_epsilon(X, Y, "sim",
                                         (INNER_BOOT_ROOT, 1, di, ai, rep))
                    m1h, m2h, S1, S2, A0h, _, _, tau = fit_moments(X, Y)
                    prep = prep_eig(S1[None], S2[None])
                    rows.append(dict(
                        base,
                        L0=exact_loss(m1h, m2h, A0h, S1_TRUE, A0_true),
                        L_dense=exact_loss(
                            m1h, m2h,
                            A_from(prep, float(m_dense * tau))[0],
                            S1_TRUE, A0_true),
                        L_RBPB=exact_loss(m1h, m2h, res["A"],
                                          S1_TRUE, A0_true),
                        m_rbpb=res["multiplier"], eps_hat=res["eps"],
                        tau=res["tau"], zero=res["zero_selected"],
                        sentinel=res["sentinel_selected"],
                        sentinel_falling=res["sentinel_falling"],
                        mcse_at_pick=res["mcse_at_pick"],
                        cond_source=res["cond_source"],
                        cond_target=res["cond_target"],
                        runtime_s=res["runtime_s"], B=res["B"],
                        n_warnings=len(res["warnings"]), failure=""))
                except (ValueError, np.linalg.LinAlgError) as err:
                    failures += 1
                    rows.append(dict(base, failure=str(err)))
            writer.writerows(rows)
            csv_fh.flush()

            ok_rows = [r for r in rows if r["failure"] == ""]

            if len(ok_rows) < 2:
                raise RuntimeError(f"{cell}: fewer than two valid "
                                   "confirmatory runs")
            L0v = np.array([r["L0"] for r in ok_rows])
            Lrb = np.array([r["L_RBPB"] for r in ok_rows])
            Lde = np.array([r["L_dense"] for r in ok_rows])
            rr = Lrb / L0v
            m_hats = np.array([r["m_rbpb"] for r in ok_rows])
            emit(f"   confirmatory ({time.perf_counter() - t0:.0f}s): "
                 f"{len(ok_rows)} ok, {failures} failures")
            if nu > 4:
                # t5 supports mean summaries and paired ratio-of-means MCSEs.
                mr = Lrb.mean() / L0v.mean()
                infl = Lrb - mr * L0v
                se = infl.std(ddof=1) / np.sqrt(len(ok_rows)) / L0v.mean()

                dense_mean_ratio = Lde.mean() / L0v.mean()
                gd = (Lrb - Lde).mean() / L0v.mean()

                emit(f"   mean ratio {mr:.4f} (paired MCSE {se:.4f}); "
                     f"dense mean ratio {dense_mean_ratio:.4f}; "
                     f"gap vs dense {gd:+.4f}")


            else:
                dense_rr = Lde / L0v
                gap_rr = (Lrb - Lde) / L0v

                emit(f"   dense median ratio {np.median(dense_rr):.4f}; "
                     f"median RBPB-minus-dense normalized gap "
                     f"{np.median(gap_rr):+.4f}")


            emit(f"   paired ratio: median {np.median(rr):.4f}, IQR "
                 f"[{np.quantile(rr, .25):.4f}, {np.quantile(rr, .75):.4f}]; "
                 f"q90/q95/q99 {np.quantile(rr, .9):.3f}/"
                 f"{np.quantile(rr, .95):.3f}/{np.quantile(rr, .99):.3f}; "
                 f"worst {rr.max():.2f}")
            
            emit(f"   improve P(L_RBPB < L0) = {np.mean(Lrb < L0v):.3f}; "
                 f"zero {np.mean([r['zero'] for r in ok_rows]):.3f}; "
                 f"sentinel {np.mean([r['sentinel'] for r in ok_rows]):.3f}; "
                 f"falling-to-sentinel "
                 f"{np.mean([r['sentinel_falling'] for r in ok_rows]):.3f}")
            
            emit(f"   m_rbpb median {np.median(m_hats):g}, IQR "
                 f"[{np.quantile(m_hats, .25):g}, "
                 f"{np.quantile(m_hats, .75):g}]")
            

    csv_fh.close()
    np.savez("heavy_333_reference.npz", **ref_store)
    emit("\ncode hashes:")
    for f in (Path(__file__).name, "rbpb_selector.py"):
        emit(f"   {f}: "
             f"{hashlib.sha256(Path(f).read_bytes()).hexdigest()}")
    with open("heavy_333_report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwritten: heavy_333_report.txt, heavy_333_records.csv, "
          "heavy_333_reference.npz")


if __name__ == "__main__":
    main()
