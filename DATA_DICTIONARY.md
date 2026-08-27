# Data dictionary

All CSV files contain simulated results. One row is one independent evaluation
dataset unless stated otherwise.

## Common fields

- `rep`: evaluation replication index.
- `n1`, `n2`, `rho`: source size, target size and `n1/n2`.
- `L0`: exact population loss of the unregularised plug-in.
- `L_KB` or `L_theory`: loss for the local known-basis comparator.
- `L_dense`: loss for the independently estimated best common multiplier on
  the dense mesh.
- `L_RBPB` or `L_rbpb`: loss for the RBPB-selected map.
- `m_dense`, `m_rbpb` or `m_hat`: dense-reference and selected multipliers.
- `eps_hat`: selected numerical regularisation level.
- `tau`: dataset-fitted scale, so `eps_hat = m_rbpb * tau`.
- `zero`: whether RBPB selected no regularisation.
- `sentinel`: whether RBPB selected the largest candidate multiplier.
- `mcse_at_pick`: conditional bootstrap Monte Carlo standard error at the
  selected candidate.
- `B`: inner-bootstrap budget.
- `boot_seed`: inner-bootstrap stream identifier.
- `failure`: diagnostic text; blank means the replication completed.

## File groups

- `data/balanced/`: three balanced Gaussian benchmarks. The paired losses
  compare no regularisation, the known-basis comparator, the dense reference
  and RBPB.
- `data/allocation/`: 21 Gaussian designs. `geometry` identifies the commuting
  control, moderate non-commuting pair or conditioning-stress pair. The CSV is
  stored as `allocation_332_records.csv.gz`; this is lossless compression of
  the CSV written by `code/allocation_332.py`.
- `data/student_t/`: covariance-matched `t5` and `t3` sensitivity records at
  target-rich and source-heavy allocations.
- `data/pilots/`: grid and bootstrap-budget calibration summaries. `regret`
  compares a candidate selection with the corresponding higher-resolution
  reference on an independent bootstrap stream.

NPZ files contain numeric dense-mesh risk curves, selected common multipliers
and fixed population matrices used by the associated experiment.
