# Paper A P4.2 rebuild (2026-11-03)

## Provenance

- Frozen implementation source: `c1b676d8c9d9d077f8edfc5aa13258d97d35ca3a`
- Reconstruction branch: `chatgpt/papera-p42-rebuild-20261103`
- The user's earlier local-only P4.2 commit was not available on GitHub; this is **not** a cherry-pick or a verification of that unpushed code.
- The 2026-10-30 frozen manuscript remains unchanged. The independent candidate is `paper_sections/sec6_p42.tex`.

## Paper/implementation contract

| Requirement | Implementation | Gate |
|---|---|---|
| 14 named public features for symbolic rules | `dispatch/symbolic_rule.py`, `dispatch/rule_library.py` | Strict names/order and formal library protocol |
| Exogenous service shares | `generation/scenario_generator.py` | `a_r~U[.8,1.2]`, normalized, independent per-job allocation |
| Entity commitments and checkpoints | `generation/scenario_generator.py` | `Q_r=rho*(N0+Nnew)*E[q]*pi_r`; independent +/-10% calendar perturbations; mathematical half-up rounding |
| Compound Poisson per-entity moments | `core/forecast.py` | `lambda_r=lambda*pi_r`; no future sampled job information |
| Service calibration | `experiments/calibrate_paper_protocol.py` | 60 candidates with fixed 25100..25129 calibration seeds; SPT and ATC only |
| Offline unavoidable shortfall | `unavoidable_supply_shortfall` | Full realized supply available **only during offline calibration** |
| Global F_ref | `rl/reward_scale.py` | New manifests bind generator_id + scenario_config_hash; old manifests rejected by formal entry |
| Formal data split | `experiments/formal_experiment.py` | Guarded formal center generator and 20000..24999 training; formal runtime remains disabled |
| L9 | `generation/scenario_config.json` and `paper_sections/sec6_p42.tex` | N_new fixed at 50; service multiplier .2/1/5 with equal global TT/WSF coefficients |

## Calibration decision rule (frozen; no relaxed thresholds)

1. Compare mean F across the same 30 instances for SPT and ATC; ties select SPT.
2. Compute ratios of summed quantities: R_WSF=sum(WSF)/sum(F), P_early=count of positive first-checkpoint shortfalls/(30R), R_irr=sum(L)/sum(WSF).
3. Accept only when 0.35<=R_WSF<=0.50, 0.20<=P_early<=0.80, R_irr<=0.40.
4. Minimize absolute distance to 0.425, then R_irr, then ascending (rho,theta,omega).
5. If zero candidates pass, **stop**. Do not alter thresholds or freeze an invalid scenario.

Command from the checkout directory with the source folder importable as `fjsp_sfr`:

```bash
PYTHONPATH=.. python -m fjsp_sfr.experiments.calibrate_paper_protocol --output-dir experiments/p42_output
```

Result files:

- `paperA_service_calibration_60.csv`
- `paperA_p42_calibration_result.json`

The GitHub Actions workflow also archives these outputs after full regression tests.

## Explicitly unresolved / gated

- The rebuilt 60-cell outcome has to be read from the GitHub Actions artifact, not inferred from the **local-only** earlier report of 0/60.
- Kappa calibration can only occur if the frozen service grid succeeds. Otherwise it remains BLOCKED.
- A new global F_ref may only be frozen after successful service and kappa calibration. The previously recorded 1872.40 is obsolete.
- Formal paper14 rule generation/risk coverage/selection and remaining M5-M10 comparisons/ablations are not proven ready.
- P5 formal training stays BLOCKED. No fabricated performance numbers, silent generator fallback, or use of test seeds for calibration.

## Test discipline

- Python compilation and `tests/test_p42_rebuild_contracts.py`.
- Full repository regression before an automatic calibration run.
- Keep source branch PR in DRAFT until all required gates pass and the author approves protocol changes.
