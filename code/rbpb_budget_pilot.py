# Pilot used to choose the RBPB bootstrap budget. Candidate budgets are nested
# prefixes of one 8,000-draw selection stream. An independent stream compares
# each selected multiplier with the 8,000-draw reference, so selection and
# evaluation noise remain separate. Datasets match the grid-pilot designs.
# Run: python rbpb_budget_pilot.py

import csv
import time
import numpy as np
from rbpb_selector import (MULTIPLIERS, make_rng, fit_moments,
                           criterion_matrix, pick_smallest)

DATA_ROOT, SELECT_ROOT, EVAL_ROOT = 20260825, 20260828, 20260829
B_CANDS = [250, 500, 1000, 2000, 4000]
B_REF = 8000
B_EVAL = 8000
REPS = 30
MEDIAN_GATE, P90_GATE = 1e-3, 5e-3
GRID = MULTIPLIERS
SENTINEL = GRID[-1]


def geom_cov(d, kappa, rot=None):
    S = np.diag(np.geomspace(1.0, kappa, d))
    if rot is not None:
        S = rot @ S @ rot.T
    return S * d / np.trace(S)


def rotation(d):
    Q, _ = np.linalg.qr(make_rng((DATA_ROOT, 900, d)).normal(size=(d, d)))
    return Q


R5, R10 = rotation(5), rotation(10)
# Cell indices match the grid pilot, giving identical outer datasets.
CELLS = [
    (0, "P0 d=1 (100,100) scalar", 1, 100, 100,
     np.array([[1.0]]), np.array([[1.0]])),
    (2, "P2 d=5 (100,100) kappa=4 reference", 5, 100, 100,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    (3, "P3 d=5 (50,200) source-scarce", 5, 50, 200,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    (4, "P4 d=5 (320,20) source-rich, zero-stability", 5, 320, 20,
     geom_cov(5, 4.0), geom_cov(5, 4.0, R5)),
    (5, "P5 d=10 (162,144) kappa=150 stress", 10, 162, 144,
     geom_cov(10, 150.0), geom_cov(10, 10.0, R10)),
]


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"RBPB budget pilot (internal calibration; roots "
         f"{DATA_ROOT}/{SELECT_ROOT}/{EVAL_ROOT})")
    emit(f"grid ({len(GRID)} points, sentinel {SENTINEL:g}): {GRID.tolist()}")
    emit(f"candidates {B_CANDS} as prefixes of B_ref={B_REF}; independent "
         f"evaluation at B={B_EVAL}")
    emit("prefix agreement is optimistic by construction; the binding")
    emit("criterion is the independent-evaluation regret vs the B_ref pick.")


    rows = []
    pass_all = {b: True for b in B_CANDS}
    for ci, label, d, n1, n2, S1t, S2t in CELLS:
        C1, C2 = np.linalg.cholesky(S1t), np.linalg.cholesky(S2t)
        cell = []
        for rep in range(REPS):
            rng = make_rng((DATA_ROOT, ci, rep))
            X = rng.normal(size=(n1, d)) @ C1.T
            Y = 0.3 + rng.normal(size=(n2, d)) @ C2.T
            m1, m2, S1, S2, A0, _, _, tau = fit_moments(X, Y)
            L = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1,
                                 tau * GRID, B_REF,
                                 make_rng((SELECT_ROOT, ci, rep)))
            picks = {b: pick_smallest(L[:, :b].mean(axis=1), GRID)
                     for b in B_CANDS + [B_REF]}
            jref = picks[B_REF]

            # Evaluate all distinct selections on one independent stream.
            distinct = sorted(set(picks.values()))
            if len(distinct) > 1:
                cols = [0.0] + [float(GRID[j]) for j in distinct
                                if GRID[j] != 0]
                Le = criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1,
                                      tau * np.array(cols), B_EVAL,
                                      make_rng((EVAL_ROOT, ci, rep)))
                where = {j: cols.index(float(GRID[j])) if GRID[j] != 0 else 0
                         for j in distinct}
            for b in B_CANDS:
                jb = picks[b]
                if jb == jref:
                    reg = se = 0.0
                else:
                    U = Le[where[jb]] - Le[where[jref]]
                    reg = U.mean() / Le[0].mean()
                    se = U.std(ddof=1) / np.sqrt(B_EVAL) / Le[0].mean()
                cell.append({"cell": label.split()[0], "rep": rep, "B": b,
                             "pick": float(GRID[jb]),
                             "ref_pick": float(GRID[jref]),
                             "agree": jb == jref,
                             "regret": reg, "regret_pos": max(0.0, reg),
                             "regret_se": se,
                             "zero": bool(GRID[jb] == 0),
                             "top": bool(GRID[jb] == SENTINEL)})
            cell.append({"cell": label.split()[0], "rep": rep, "B": B_REF,
                         "pick": float(GRID[jref]),
                         "ref_pick": float(GRID[jref]), "agree": True,
                         "regret": 0.0, "regret_pos": 0.0, "regret_se": 0.0,
                         "zero": bool(GRID[jref] == 0),
                         "top": bool(GRID[jref] == SENTINEL)})
        rows += cell

        # Time one fresh call per budget on replication-zero data.
        rng = make_rng((DATA_ROOT, ci, 0))
        X = rng.normal(size=(n1, d)) @ C1.T
        Y = 0.3 + rng.normal(size=(n2, d)) @ C2.T
        m1, m2, S1, S2, A0, _, _, tau = fit_moments(X, Y)
        times = {}
        for b in B_CANDS + [B_REF]:
            t0 = time.perf_counter()
            criterion_matrix(S1, S2, A0, n1, n2, n1 - 1, n2 - 1, tau * GRID,
                             b, make_rng((SELECT_ROOT, ci, 999, b)))
            times[b] = time.perf_counter() - t0

        ref_rows = [r for r in cell if r["B"] == B_REF]
        emit(f"\n {label} ")
        emit(f"   B_ref pick median "
             f"{np.median([r['pick'] for r in ref_rows]):.3g}; zero rate "
             f"{np.mean([r['zero'] for r in ref_rows]):.2f}; sentinel rate "
             f"{np.mean([r['top'] for r in ref_rows]):.2f}; "
             f"time {times[B_REF]:.2f}s")
        for b in B_CANDS:
            rb = [r for r in cell if r["B"] == b]
            regs = np.array([r["regret_pos"] for r in rb])
            med, p90 = np.median(regs), np.quantile(regs, 0.9)
            ok = med <= MEDIAN_GATE and p90 <= P90_GATE
            pass_all[b] = pass_all[b] and ok
            emit(f"   B={b:>5d}: agree {np.mean([r['agree'] for r in rb]):.2f}; "
                 f"regret med {med:.2e}, P90 {p90:.2e} "
                 f"[{'PASS' if ok else 'FAIL'}]; "
                 f"zero {np.mean([r['zero'] for r in rb]):.2f}, "
                 f"sentinel {np.mean([r['top'] for r in rb]):.2f}; "
                 f"time {times[b]:.2f}s")

    emit("\n" )
    passing = [b for b in B_CANDS if pass_all[b]]
    if passing:
        emit(f"budgets passing the gates in every cell: {passing}")
        emit(f"smallest-passing rule: B = {min(passing)} for simulations; "
             f"real-data calls stay at 4000.")
        emit("check the zero and sentinel columns against the B_ref row for")
        emit("systematic drift before freezing.")
    else:
        emit("no candidate passed; raise the budget range and rerun.")
    emit("Internal calibration only; not report evidence.")
    emit("=" * 74)

    with open("budget_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("budget_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwritten: budget_report.txt, budget_records.csv")


if __name__ == "__main__":
    main()
