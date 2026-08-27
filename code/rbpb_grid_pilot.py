# Pilot used to define the fixed RBPB candidate grid. It compares the proposed
# grid with a finer and wider reference grid. When their selections differ, an
# independent bootstrap stream compares the selected candidates, separating
# selection noise from evaluation noise.
# Run: python rbpb_grid_pilot.py

import csv
import numpy as np
from rbpb_selector import make_rng, fit_moments, criterion_matrix, pick_smallest

DATA_ROOT, SELECT_ROOT, EVAL_ROOT = 20260825, 20260826, 20260827
B_SELECT, B_EVAL, REPS = 2000, 4000, 30
MEDIAN_GATE, P90_GATE = 1e-3, 5e-3

BASE = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16.])
pos = BASE[BASE > 0]
MIDS = (pos[:-1] + pos[1:]) / 2
REF = np.array(sorted(set(BASE) | {1 / 16, 1 / 8} | set(MIDS)
                      | {18, 20, 22, 24, 32}))
IB = [int(np.where(REF == m)[0][0]) for m in BASE]
I24 = int(np.where(REF == 24)[0][0])
I32 = int(np.where(REF == 32)[0][0])


def geom_cov(d, kappa, rot=None):
    # Return a trace-normalised geometric spectrum with condition number kappa.
    S = np.diag(np.geomspace(1.0, kappa, d))
    if rot is not None:
        S = rot @ S @ rot.T
    return S * d / np.trace(S)


def rotation(d):
    Q, _ = np.linalg.qr(make_rng((DATA_ROOT, 900, d)).normal(size=(d, d)))
    return Q


R5, R10 = rotation(5), rotation(10)
CELLS = [
    ("P0 d=1 (100,100) scalar", 1, 100, 100,
     np.array([[1.0]]), np.array([[1.0]])),
    ("P1 d=5 (50,50) kappa=4 noncommuting", 5, 50, 50,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    ("P2 d=5 (100,100) kappa=4 reference", 5, 100, 100,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    ("P3 d=5 (50,200) source-scarce", 5, 50, 200,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    ("P4 d=5 (320,20) source-rich", 5, 320, 20,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    ("P5 d=10 (162,144) kappa=150 stress", 10, 162, 144,
     geom_cov(10, 150.0), geom_cov(10, 10.0, R10)),
]


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"RBPB grid pilot (internal design audit; roots "
         f"{DATA_ROOT}/{SELECT_ROOT}/{EVAL_ROOT})")
    emit(f"base grid ({len(BASE)}): {BASE.tolist()}")
    emit(f"reference grid ({len(REF)}): base + 1/16, 1/8, midpoints, "
         f"18, 20, 22, 24, 32")


    rows = []
    for ci, (label, d, n1, n2, S1t, S2t) in enumerate(CELLS):
        C1, C2 = np.linalg.cholesky(S1t), np.linalg.cholesky(S2t)
        cell = []
        for rep in range(REPS):
            rng = make_rng((DATA_ROOT, ci, rep))
            X = rng.normal(size=(n1, d)) @ C1.T
            Y = 0.3 + rng.normal(size=(n2, d)) @ C2.T
            m1, m2, S1, S2, A0, _, _, tau = fit_moments(X, Y)
            L = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1,
                                 tau * REF, B_SELECT,
                                 make_rng((SELECT_ROOT, ci, rep)))
            q = L.mean(axis=1)
            jb = IB[pick_smallest(q[IB], BASE)]
            jr = pick_smallest(q, REF)

            # Check whether the criterion is still decreasing above 24.
            tail = L[I32] - L[I24]
            falling = tail.mean() < -2 * tail.std(ddof=1) / np.sqrt(B_SELECT)

            # Re-evaluate distinct selections on an independent stream.
            reg = se = 0.0
            if jb != jr:
                cols = [0.0]
                for m in (REF[jb], REF[jr]):
                    if m not in cols:
                        cols.append(float(m))
                Le = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1,
                                      tau * np.array(cols), B_EVAL,
                                      make_rng((EVAL_ROOT, ci, rep)))
                U = Le[cols.index(float(REF[jb]))] \
                    - Le[cols.index(float(REF[jr]))]
                reg = U.mean() / Le[0].mean()
                se = U.std(ddof=1) / np.sqrt(B_EVAL) / Le[0].mean()
            cell.append({"cell": label.split()[0], "rep": rep,
                         "base_pick": float(REF[jb]),
                         "ref_pick": float(REF[jr]),
                         "regret": reg, "regret_pos": max(0.0, reg),
                         "regret_se": se, "falling_24_32": bool(falling)})
        rows += cell

        regs = np.array([r["regret_pos"] for r in cell])
        med, p90 = np.median(regs), np.quantile(regs, 0.9)
        gate = med <= MEDIAN_GATE and p90 <= P90_GATE
        picks = np.array([r["base_pick"] for r in cell])
        refs = np.array([r["ref_pick"] for r in cell])
        vals, cnts = np.unique(refs, return_counts=True)
        emit(f"\n {label} ")
        emit(f"   base pick median {np.median(picks):.3g}; reference picks "
             + ", ".join(f"{v:g}:{c}" for v, c in zip(vals, cnts)))
        emit(f"   regret vs reference pick: median {med:.2e}, P90 {p90:.2e} "
             f"[{'PASS' if gate else 'FAIL'}]")
        emit(f"   endpoint 32 picks {int((refs == 32).sum())}; picks below "
             f"1/4: {int(((refs > 0) & (refs < 0.25)).sum())}; "
             f"24->32 falling {sum(r['falling_24_32'] for r in cell)}/{REPS}")
        if not gate:
            bad = {}
            for r in cell:
                if r["ref_pick"] not in BASE and r["regret_pos"] > MEDIAN_GATE:
                    bad.setdefault(r["ref_pick"], []).append(r["regret_pos"])
            emit("   off-grid points implicated: "
                 + ", ".join(f"{k:g} ({len(v)} reps, median "
                             f"{np.median(v):.2e})" for k, v in bad.items()))

    emit("\n" )
    emit("Decision rule: keep M0 if every cell passes; otherwise review only")
    emit("the implicated off-grid points (documented amendment). Endpoint and")
    emit("falling counts are diagnostics. Internal pilot, not report evidence.")


    with open("grid_pilot_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("grid_pilot_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwritten: grid_pilot_report.txt, grid_pilot_records.csv")


if __name__ == "__main__":
    main()
