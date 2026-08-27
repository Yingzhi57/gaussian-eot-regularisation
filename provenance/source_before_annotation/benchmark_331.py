# benchmark_331.py
# Gaussian benchmark for the theory-guided full-covariance RBPB selector
# (Section 3.3.1). Two stages on independent outer datasets:
#   reference stage: R_ref datasets build the scenario-level true-risk
#     curve over the dense mesh M_REF = {k/8 : k = 0..256}; its smallest
#     minimiser is the best common multiplier policy, with the
#     production-grid restriction kept as a discretisation diagnostic;
#   confirmatory stage: R_confirm fresh datasets evaluate four policies
#     by the exact affine Gaussian L2 loss: m = 0, the balanced theory
#     reference m = 5, the dense-reference best fixed multiplier, and
#     the RBPB selection from rbpb_selector (production grid, B = 250).
# The population geometry is seeded by GEOM_ROOT, separate from all
# simulation roots: G1 is the frozen pilot covariance pair (design root
# 20260825), and changing simulation seeds must never move the
# population scenario. Datasets and bootstrap streams reuse nothing
# from the pilots (fresh roots below).
# Ratio and gap uncertainties use the paired delta-method (influence)
# form; the pre-authorised extension criterion stays the frozen paired
# RBPB-vs-zero MCSE with gate 0.002.
# Run: python benchmark_331.py

import csv
import hashlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbpb_selector import (MULTIPLIERS, make_rng, fit_moments, prep_eig,
                           A_from, pick_smallest, select_epsilon)

# frozen design (G1, dense reference, G3)
GEOM_ROOT = 20260825       # population geometry: the frozen pilot pair
REF_ROOT, CONF_ROOT, BOOT_ROOT = 20260830, 20260831, 20260832
R_REF = 5000
R_CONFIRM = 5000
R_EXTENDED = 10000
MCSE_GATE = 0.002          # normalized paired RBPB-vs-zero MCSE
M_REF = np.arange(0, 257) / 8.0          # dense mesh on [0, 32]
M_THEORY = 5.0             # balanced estimated-means anchor, 10G/(nH) = 5 tau
I24 = int(np.where(M_REF == 24)[0][0])
I32 = 256
GRID_IN_REF = [int(np.where(M_REF == m)[0][0]) for m in MULTIPLIERS]


def geom_cov(d, kappa, rot=None):
    S = np.diag(np.geomspace(1.0, kappa, d))
    if rot is not None:
        S = rot @ S @ rot.T
    return S * d / np.trace(S)


def rotation(d):
    Q, _ = np.linalg.qr(make_rng((GEOM_ROOT, 900, d)).normal(size=(d, d)))
    return Q


# G1: principal balanced non-commuting benchmark (pilot covariance pair);
# G2 secondary cells (smaller-sample balanced, d=10 stress) join once frozen
CELLS = [
    (0, "G1 d=5 (100,100) balanced kappa=4 noncommuting", 5, 100, 100,
     geom_cov(5, 4.0), geom_cov(5, 4.0, rotation(5))),
]


def exact_loss(m1h, m2h, A, Sigma1, A0_true):
    # exact affine Gaussian L2 loss against the population map (m1=m2=0)
    v = m2h - A @ m1h
    D = A - A0_true
    return float(v @ v + np.einsum("ij,jk,ik->", D, Sigma1, D))


def ratio_and_se(Lv, L0v):
    # paired delta-method SE for the risk ratio mean(Lv)/mean(L0v)
    rho = Lv.mean() / L0v.mean()
    infl = Lv - rho * L0v
    return rho, infl.std(ddof=1) / np.sqrt(len(L0v)) / L0v.mean()


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("3.3.1 Gaussian benchmark (frozen G1 + dense reference + G3)")
    emit(f"geometry root {GEOM_ROOT} (frozen pilot pair); simulation roots "
         f"{REF_ROOT}/{CONF_ROOT}/{BOOT_ROOT}")
    emit(f"R_ref={R_REF}, R_confirm={R_CONFIRM} (extension to {R_EXTENDED} "
         f"if normalized paired MCSE > {MCSE_GATE})")

    all_rows = []
    for ci, label, d, n1, n2, S1_true, S2_true in CELLS:
        C1 = np.linalg.cholesky(S1_true)
        C2 = np.linalg.cholesky(S2_true)
        A0_true = A_from(prep_eig(S1_true[None], S2_true[None]), 0.0)[0]
        comm = np.linalg.norm(S1_true @ S2_true - S2_true @ S1_true) \
            / (np.linalg.norm(S1_true) * np.linalg.norm(S2_true))
        emit(f"\n-- {label} --")
        emit(f"   noncommutativity |[S1,S2]|/(|S1||S2|) = {comm:.5f}")

        # reference stage: scenario-level true-risk curve on the dense mesh
        t0 = time.perf_counter()
        ref_losses = np.empty((R_REF, len(M_REF)))
        for rep in range(R_REF):
            X = make_rng((REF_ROOT, ci, rep, 0)).normal(size=(n1, d)) @ C1.T
            Y = make_rng((REF_ROOT, ci, rep, 1)).normal(size=(n2, d)) @ C2.T
            m1h, m2h, S1, S2, _, _, _, tau = fit_moments(X, Y)
            prep = prep_eig(S1[None], S2[None])
            for k, m in enumerate(M_REF):
                A = A_from(prep, float(m * tau))[0]
                ref_losses[rep, k] = exact_loss(m1h, m2h, A, S1_true, A0_true)
        curve = ref_losses.mean(axis=0)
        curve_mcse = ref_losses.std(axis=0, ddof=1) / np.sqrt(R_REF)
        j_dense = pick_smallest(curve, M_REF)
        m_dense = float(M_REF[j_dense])
        j_grid = GRID_IN_REF[pick_smallest(curve[GRID_IN_REF], MULTIPLIERS)]
        m_grid = float(M_REF[j_grid])
        tail = ref_losses[:, I32] - ref_losses[:, I24]
        tail_mean = tail.mean()
        tail_mcse = tail.std(ddof=1) / np.sqrt(R_REF)
        falling = tail_mean < -2 * tail_mcse
        unresolved = (j_dense == I32) or falling
        emit(f"   reference curve ({time.perf_counter() - t0:.0f}s): "
             f"dense best m* = {m_dense:g}; production-grid best "
             f"(diagnostic) = {m_grid:g}; range "
             f"{'UNRESOLVED' if unresolved else 'resolved'} "
             f"(top pick {j_dense == I32}, 24->32 falling {bool(falling)})")
        np.savez("benchmark_331_reference.npz",
                 M_REF=M_REF, curve_mean=curve, curve_mcse=curve_mcse,
                 Sigma1_true=S1_true, Sigma2_true=S2_true,
                 m_dense=m_dense, m_grid=m_grid,
                 tail_mean=tail_mean, tail_mcse=tail_mcse)

        # confirmatory stage: four policies on fresh independent datasets
        t0 = time.perf_counter()
        rows = []
        failures = 0
        rep = 0
        target = R_CONFIRM
        while rep < target:
            try:
                X = make_rng((CONF_ROOT, ci, rep, 0)).normal(size=(n1, d)) @ C1.T
                Y = make_rng((CONF_ROOT, ci, rep, 1)).normal(size=(n2, d)) @ C2.T
                res = select_epsilon(X, Y, "sim", (BOOT_ROOT, ci, rep))
                m1h, m2h, S1, S2, A0h, _, _, tau = fit_moments(X, Y)
                prep = prep_eig(S1[None], S2[None])
                L0 = exact_loss(m1h, m2h, A0h, S1_true, A0_true)
                Lth = exact_loss(m1h, m2h,
                                 A_from(prep, float(M_THEORY * tau))[0],
                                 S1_true, A0_true)
                Lde = exact_loss(m1h, m2h,
                                 A_from(prep, float(m_dense * tau))[0],
                                 S1_true, A0_true)
                Lgr = exact_loss(m1h, m2h,
                                 A_from(prep, float(m_grid * tau))[0],
                                 S1_true, A0_true)
                Lrb = exact_loss(m1h, m2h, res["A"], S1_true, A0_true)
                rows.append({"cell": label.split()[0], "rep": rep,
                             "L0": L0, "L_theory": Lth, "L_dense": Lde,
                             "L_grid": Lgr, "L_rbpb": Lrb,
                             "m_hat": res["multiplier"],
                             "eps_hat": res["eps"], "tau": res["tau"],
                             "zero": res["zero_selected"],
                             "sentinel": res["sentinel_selected"],
                             "sentinel_falling": res["sentinel_falling"],
                             "mcse_at_pick": res["mcse_at_pick"],
                             "cond_source": res["cond_source"],
                             "cond_target": res["cond_target"],
                             "runtime_s": res["runtime_s"],
                             "B": res["B"],
                             "boot_seed": str(res["seed"]),
                             "n_warnings": len(res["warnings"])})
            except ValueError:
                failures += 1          # recorded NA, raw discipline
            rep += 1
            if rep == R_CONFIRM and target == R_CONFIRM:
                L0v = np.array([r["L0"] for r in rows])
                dif = np.array([r["L_rbpb"] for r in rows]) - L0v
                crit = dif.std(ddof=1) / np.sqrt(len(dif)) / L0v.mean()
                if crit > MCSE_GATE:
                    target = R_EXTENDED
                    emit(f"   pre-authorised extension: normalized paired "
                         f"MCSE {crit:.4f} > {MCSE_GATE} -> "
                         f"R_confirm = {R_EXTENDED}")
        all_rows += rows

        L0v = np.array([r["L0"] for r in rows])
        base = L0v.mean()
        emit(f"   confirmatory stage ({time.perf_counter() - t0:.0f}s): "
             f"{len(rows)} datasets, {failures} failures")
        emit("   risk ratios vs m=0 (paired delta-method SE):")
        for name, key in (("theory m=5", "L_theory"),
                          ("dense-ref best", "L_dense"),
                          ("grid-ref best", "L_grid"),
                          ("RBPB", "L_rbpb")):
            Lv = np.array([r[key] for r in rows])
            rho, se = ratio_and_se(Lv, L0v)
            emit(f"      {name:<17s} {rho:.4f}  (+/- {se:.4f})")
        U = np.array([r["L_rbpb"] - r["L_dense"] for r in rows])
        gap = U.mean() / base
        gap_se = (U - gap * L0v).std(ddof=1) / np.sqrt(len(rows)) / base
        emit(f"   RBPB gap vs dense-ref policy: {gap:+.4f} (+/- {gap_se:.4f})")
        Lrb = np.array([r["L_rbpb"] for r in rows])
        m_hats = np.array([r["m_hat"] for r in rows])
        emit(f"   improvement P(L_RBPB < L0) = {np.mean(Lrb < L0v):.3f}; "
             f"zero rate {np.mean([r['zero'] for r in rows]):.3f}; "
             f"sentinel rate {np.mean([r['sentinel'] for r in rows]):.3f}; "
             f"sentinel-falling rate "
             f"{np.mean([r['sentinel_falling'] for r in rows]):.3f}")
        emit(f"   selected multiplier: median {np.median(m_hats):g}, "
             f"IQR [{np.quantile(m_hats, .25):g}, "
             f"{np.quantile(m_hats, .75):g}]; inner MCSE at pick median "
             f"{np.median([r['mcse_at_pick'] for r in rows]):.2e}; "
             f"selector runtime median "
             f"{np.median([r['runtime_s'] for r in rows]):.3f}s at "
             f"B={rows[0]['B']}")

        # composite figure for the principal cell
        if ci == 0:
            fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(13.2, 3.8))
            p1.plot(M_REF, curve / curve[0], "-", color="C0", alpha=0.9,
                    label="reference true risk")
            p1.axvline(M_THEORY, color="C2", ls=":", alpha=0.8,
                       label="theory m=5")
            p1.axvline(m_dense, color="C3", ls="--", alpha=0.8,
                       label=f"dense best m*={m_dense:g}")
            ax1b = p1.twinx()
            vals, cnts = np.unique(m_hats, return_counts=True)
            ax1b.bar(vals, cnts / len(m_hats), width=0.35, color="C1",
                     alpha=0.4)
            ax1b.set_ylabel("RBPB selection frequency")
            p1.set_xlabel("multiplier m")
            p1.set_ylabel("true risk / risk at m = 0")
            p1.set_title("reference curve and RBPB selections")
            p1.legend(fontsize=7, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18), ncol=3)
            names = ["theory\nm=5", "dense-ref\nbest", "grid-ref\nbest",
                     "RBPB"]
            keys = ["L_theory", "L_dense", "L_grid", "L_rbpb"]
            stats = [ratio_and_se(np.array([r[k] for r in rows]), L0v)
                     for k in keys]
            p2.bar(names, [s[0] for s in stats],
                   yerr=[s[1] for s in stats],
                   color=["C2", "C3", "C4", "C1"], alpha=0.8,
                   capsize=3, width=0.6)
            p2.axhline(1.0, color="k", ls=":", lw=1)
            p2.set_ylim(0.90, 1.01)
            p2.set_ylabel("risk ratio vs m = 0")
            p2.set_title("policy comparison")
            p3.bar(vals, cnts, width=0.35, color="C1", alpha=0.85)
            p3.axvline(M_THEORY, color="C2", ls=":", alpha=0.8)
            p3.axvline(m_dense, color="C3", ls="--", alpha=0.8)
            p3.set_xlabel("selected multiplier (grid values)")
            p3.set_ylabel("count")
            p3.set_title(f"selections (zero "
                         f"{np.mean([r['zero'] for r in rows]):.1%}, "
                         f"improve {np.mean(Lrb < L0v):.1%})")
            fig.tight_layout(w_pad=2.5)
            fig.savefig("benchmark_331.pdf", bbox_inches="tight")
            fig.savefig("benchmark_331.png", dpi=150, bbox_inches="tight")
            emit("   figure written: benchmark_331.pdf / .png")

    emit("\ncode hashes:")
    for f in ("benchmark_331.py", "rbpb_selector.py"):
        h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        emit(f"   {f}: {h}")

    with open("benchmark_331_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("benchmark_331_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print("\nwritten: benchmark_331_report.txt, benchmark_331_records.csv, "
          "benchmark_331_reference.npz")


if __name__ == "__main__":
    main()
