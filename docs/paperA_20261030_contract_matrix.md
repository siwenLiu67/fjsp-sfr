# Paper A 2026-10-30 contract matrix

## Authority and integration boundary

- Base commit: `fa9e045616624813afe6a13ae69df2fbc05d10fa` on `main`.
- Authoritative sources and hashes: `docs/paperA_20261030_contract/README.md`.
- Official integration anchor: `experiments/formal_experiment.py` together
  with `rl/graph.py`, `rl/ppo_agent.py`, and `rl/ppo_trainer.py`.
- The corrected configuration is named `paper_20261030`; the historical path
  is named `legacy`. `--contract-config` is mandatory. Selecting the frozen
  corrected contract fails closed until its prerequisites exist; it never
  falls back to `legacy`.
- `agents/ca_ppo.py` is a reusable source of action-equivalence routines, not
  a second official training entry point.

Status terms are **aligned**, **partial**, **incompatible**, **missing**, and
**frozen/gated**. A later-phase item is deliberately not implemented in P0.

## Equation/table to code and test matrix

| Manuscript contract | Required behavior | Current module/function | Authoritative test | P0 status and exact incompatibility |
| --- | --- | --- | --- | --- |
| Sec.4.6, Eqs. `cert-vis`, `cert-fut-on`, `cert-fut-priv`, `cert-interval` | Deterministic online and training-privileged shortage intervals, intersected along the observed trajectory; Actor never sees privileged future attributes. | `dispatch/certificates.py` computes both constructions and their monotone history; `env/ca_rule_selection.py` exposes only online endpoints through checkpoint metrics and keeps the immutable privileged record behind an opt-in Critic API. | `tests/test_paper_certificates.py` hand examples, hidden-future invariance, snapshot, contradiction and reachable-trajectory coverage tests. | **aligned in Phase 1**; the 12-dimensional graph consumption of the two public endpoints remains Phase 2. |
| Sec.5.1 Table `tab:feat` | Job/Machine/Checkpoint widths are 14/2/12; the final two checkpoint coordinates are online interval endpoints. | `rl/graph.py::_build_paper_graph` emits 14/2/**10**; `checkpoint_rows` omits both endpoints. | `test_authoritative_paper_graph_has_14_2_12_inputs` | **incompatible**; executable test fails 10 vs 12. Phase 2 owns implementation. |
| Sec.5.1 Eq. `crg` | Relation-wise GATv2 softmax, timely/non-timely checkpoint relations and reverse relations, relation-message sum, LayerNorm, zero-message branch. | `rl/hgt.py::HGTLayer` uses ReLU scores, degree averaging, and one unsplit `job_checkpoint` relation. | `test_contract_declares_crg_gatv2` (declaration); numerical attention tests belong to Phase 2. | **missing/incompatible**; Phase 2 blocker. |
| Sec.5.3 Eq. `global-reward-scale` | One positive scalar from the independent calibration set: mean of SPT and ATC objectives; fixed across instances and matched libraries/methods. | `rl/reward_scale.py` computes and hash-verifies the calibration manifest; `PPOTrainer(reward_reference_mode="global_calibration")` and optional fixed references in matched trainers reuse the scalar. Legacy per-instance behavior remains explicit and unchanged. | `test_authoritative_streaming_ppo_does_not_recompute_f_ref_per_instance`, `test_global_reward_scale_manifest_and_trainer_path`, and the updated Group-F contract. | **aligned in Phase 1**; corrected formal runs require a manifest and checkpoints validate its hash. |
| Sec.5.3 Eqs. `rew`, `cert-gae`; Sec.6 Table `tab:hyper` | SC-PPO uses `gamma=1`; reward is the true cost increment divided by the fixed global scale. | `rl/ppo_trainer.py` accepts `gamma`; the historical formal entry hard-coded `0.99`. `experiments/paperA_contract.py::PAPER_20261030` freezes `1.0`. | `test_paper_contract_freezes_paper14_gamma_one_and_final_checkpoint` | **frozen/gated**; formal value is explicit while runnable legacy remains 0.99. Phase 1/3 must wire the scale and official trainer. |
| Sec.5.4 Eq. `equiv` | Merge rules by actual selected job; behavior probability, PPO ratio and entropy use the aggregate distribution; all-equal actions have zero Actor gradient. | `agents/ca_ppo.py::build_action_groups` and `aggregate_action_group_logits` implement reusable grouping. `rl/ppo_trainer.py` still samples rule indices. | `test_contract_declares_action_equivalence_merge` (declaration); integration tests belong to Phase 3. | **partial**; utility exists off the official path. |
| Sec.6.4 Table `tab:critic` | C1 scalar; C2 online-only decomposed; C3 privileged unconstrained decomposed; C4 privileged interval-constrained scalar heads, with matched public encoder/data/budget. | No C1--C4 variant API in the official path. | `test_contract_declares_critic_variant_blocker` | **missing**; Phase 3. |
| Sec.5.4 Eqs. `cc-ordered`--`cert-critic-loss`; Sec.6 Table `tab:ccdvl_ablation` | D1 unconstrained quantile heads and D2 certificate-constrained ordered quantiles; D2 is the main CC-DVL method. | No D1/D2 heads or pinball-loss implementation. | Contract-matrix review; numerical tests belong to Phase 4. | **missing**; Phase 4. No separate D3 is permitted. |
| Sec.6.1.2 data split | Calibration, streaming train, 20 screening, 20 holdout, 50 validation, 100 center test, 50 per generalization scenario; disjoint seeds. | Seed partitions exist across `generation/scenario_config.json`, `experiments/formal_experiment.py`, and generator tests, but a dedicated calibration split for global `F_ref` is not wired into the official RL path. | Existing seed-disjointness tests plus future Phase-5 protocol tests. | **partial**; historical S/M/L/X ranges are not evidence for the corrected center/L9 protocol. |
| Sec.6.1.2 evaluation rule | Evaluate the final training checkpoint; validation selects hyperparameters only, never an interim model. | `experiments/formal_experiment.py::train_ppo` restores `best_state` from interim validation and writes it as `best.pt`. `PAPER_20261030.evaluation_checkpoint` freezes `final_training`. | `test_paper_contract_freezes_paper14_gamma_one_and_final_checkpoint` | **incompatible but frozen**; implementation belongs to Phase 3/5. |
| Sec.6.1, 6.3--6.6 formal experiment set | GP-Best, LLM-Best, Direct-Job PPO, CRG-DDQN/A2C/Rainbow, Freq-Matched, effective disagreement, prediction mismatch, and K=1/2/3 pressure tests under matched protocol. | Some legacy baselines/reports exist, but no audited corrected formal matrix implements the complete set. | Future Phase-5 protocol and smoke tests. | **partial/missing**; historical result files are not corrected results. |
| Sec.6.7 industrial case, Table `tab:industrial_data_mapping` | External flexible-packaging reconstruction, pre-registered exogenous simulated contract, final held-out evaluation only. | No industrial dataset loader/runner identified. | Future Phase-6 mapping and legality tests. | **missing**; Phase 6. |

## Explicit formal and legacy configurations

| Field | `legacy` | `paper_20261030` |
| --- | --- | --- |
| Feature set | `theory11_plus3` | `paper14` |
| Graph | `legacy` (14/2/8) | `paper_graph` contract (14/2/12) |
| Encoder | ReLU HGT | relation-wise GATv2 CRG |
| PPO gamma | 0.99 | 1.0 |
| Reward reference | per-instance active-library mean | one calibration SPT/ATC scalar |
| Action semantics | rule index | merged actual-job equivalence classes |
| Evaluation checkpoint | best interim validation | final training checkpoint |
| Runnable after P0 | yes, historical only | no; fails closed with listed blockers |

## Baseline test evidence before P0 edits

Command: `PYTHONPATH=..:. python3 -m pytest -q`

Result at `fa9e045`: **200 passed, 5 failed** in 7.15 s. All five failures
were pre-existing Group-F contract gaps: library-dependent `F_ref`, unsafe
division, invalid Cond/Filter exception handling, incorrect all-filtered
fallback, and missing-library fallback. P0 does not alter those algorithms.

After the P0 freeze, the same command reports **208 passed, 7 failed** in
5.68 s. The two additional failures are the deliberately executable corrected
contracts for 14/2/12 graph inputs and one global calibration `F_ref`. The
historical suite excluding the new P0 file remains **200 passed, 5 failed**;
there is no new legacy regression.

The reproducible non-training smoke command is:

```bash
PYTHONPATH=..:. python3 experiments/formal_experiment.py \
  --contract-config legacy --library-id classic4 \
  --library-file experiments/classic4_library.json \
  --scale S --difficulty balanced --seed 1 --selector fixed \
  --output-dir /tmp/papera_p0_legacy_smoke --smoke
```

It completed successfully and wrote `summary.json`, `per_instance.csv`, and
`manifest.json`. Replacing `legacy` with `paper_20261030` exits before any run
with `ContractNotReadyError` and the six frozen blockers; this is the intended
P0 fail-closed behavior.

## Unresolved scientific or document decisions

These are manuscript TBDs, not values to infer from old code:

1. Sec.6 has not frozen `rho^Q`, `theta`, `omega`, `kappa`, their resulting
   diagnostics, or the independent calibration seed interval.
2. Sec.6 has no fixed SC-PPO minibatch size or selected learning rate; the
   table intentionally requires pre-experiment validation.
3. GP grammar depth, population, generation count, crossover/mutation rates,
   and effective-evaluation budget remain TBD.
4. CC-DVL quantile count `K`, quantile-loss coefficient `alpha_Q`, and the
   associated gradient-scale protocol remain TBD.
5. DDQN/A2C/Rainbow hyperparameters and selected discount factors remain TBD.
6. The K=3 checkpoint insertion time/demand proportion and pressure-test
   sample size remain TBD.
7. The industrial-case time origin, unit conversions, product grouping,
   machine-eligibility reconstruction, and exogenous contract parameters must
   be pre-registered before viewing method results.

No value above is supplied by the `legacy` configuration.

## Phase-1 evidence

- `dispatch/certificates.py` implements the online and opt-in Critic-only
  constructions, with independent monotone history and explicit contradiction
  errors.  Public checkpoint metrics contain only `online_shortfall_lower` and
  `online_shortfall_upper`; P2 remains responsible for adding them to the
  12-dimensional paper graph.
- `experiments/paperA_global_f_ref.json` was computed only from center-scenario
  seeds 25100--25129.  Its fixed value is `1872.4027777777778` and its canonical
  payload SHA-256 is
  `f41f822755a4a5ea28a2497dd91b48c81e774047a7e7e40a64a58e158c125b1b`.
- Phase-1 numerical tests cover hand-derived endpoints, forecast exclusion,
  hidden-future invariance, snapshot restoration, degenerate and contradictory
  intervals, and terminal reward telescoping.  Three additional center-scenario
  random feasible trajectories produced 10,084/10,084 online and privileged
  certificate inclusions, with only floating-point telescoping error below
  `2e-12`.
