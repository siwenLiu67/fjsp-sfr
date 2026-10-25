# Formal experiment serial runtime estimate

The estimate uses the measured `theory11_plus3` pooled-mix timings:
S=0.18 h, M=0.39 h and L=0.90 h per 5,000-episode training run.

| Category | S runs | M runs | L runs | Training hours |
|---|---:|---:|---:|---:|
| Eight libraries, HGT-PPO, three seeds | 24 | 24 | 24 | 35.28 |
| MLP-PPO and DQN variants | 6 | 6 | 6 | 8.82 |
| Information ablations | 0 | 6 | 0 | 2.34 |
| Balanced M auxiliary runs | 0 | 6 | 0 | 2.34 |
| Training subtotal | 30 | 42 | 30 | 48.78 |
| Validation, 100-instance tests and 21 evaluation-only runs | — | — | — | about 1.2 |
| **Estimated serial total** |  |  |  | **about 50.0 h** |

There are 123 matrix entries: 102 trainable runs and 21 evaluation-only
runs. Actual wall time depends on CPU contention and external-drive latency.
