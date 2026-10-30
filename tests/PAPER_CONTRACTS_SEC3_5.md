# Paper A executable-contract ledger

> Authority notice (2026-10-30): this file documents the earlier Sec.3--5
> contract and remains as a historical test record. The current Paper A
> contract is registered in `docs/paperA_20261030_contract/README.md`; conflicts
> are tracked in `docs/paperA_20261030_contract_matrix.md`, and the newer source
> takes precedence.

This ledger is frozen against the following manuscript files. Historical paper
copies are not admissible inputs to these tests.

| section | SHA-256 |
|---|---|
| `sec3.tex` | `256ab1a83b1dd9da27275aae17ac65657a910b7c3c57b64de5616dc80c8fb6f8` |
| `sec4.tex` | `c217c3d741839e023cca12be09541d2a3c2a3329285e3a10e96a27344e3b9790` |
| `sec5.tex` | `1a776791f74d858b94d5b6e4212e5639c8837eb00c96f2e382e5049fefd393c7` |
| `sec6.tex` | `3247bf6301c7dc3b5f7b9797336aa1c030cf98a6bdc010aab005c2d869dd7193` |

The executable contracts live in `tests/test_paper_contracts_sec3_5.py`.
They are deliberately not marked `xfail`: a failure is a visible unresolved
paper/code discrepancy and must not be converted into a pass by weakening the
paper formula.

## Contract groups

| group | manuscript basis | executable assertions |
|---|---|---|
| A | sec3 lines 79--83 | Commitments are invariant to a changed future realization under the same scenario prior; changing hidden job identity, route, quantity, due date or processing data cannot change the current legal observation. |
| B | sec4 equations (4.5)--(4.7) | Bucket means and variances accumulate before one conservative correction; short windows may validly remain zero; the prefix maximum and first coverage time are checked independently; formal scenarios expose compound-Poisson moments through every last checkpoint. |
| C | sec4 equations (4.2), (4.3), (4.9)--(4.11), (4.12) | `C^LB`, `Omega`, confirmed final-operation supply `A`, `G+`, `Omega+`, the capacity applicability guard and shortfall, and all three target-checkpoint branches are checked through the centralized Sec. 4 interfaces using hand-valued instances. |
| D | sec5 Table 5-2 | The `paper14` interface fixes the ten theory features followed by the four basic features; a pure mathematical oracle and two-candidate production fixture check every value, signed slack, target absence, zero weights, multiple overdue jobs and trigger-machine dependence. |
| E | sec5 Table 5-1 and equation (5.1) | Explicit `paper_graph` nodes are 14/2/10, settled/nonpositive-gap checkpoints are excluded, edge values use the paper compressions, empty-checkpoint/no-neighbor states remain finite, and Job rows are byte-identical at float32 precision to canonical `paper14`. |
| F | sec5 equations (5.4)--(5.6) and lines 124--131 | `F_ref` is the SPT/ATC mean and library-invariant; raw increments telescope to `F`; invalid expressions and all-filtered candidates follow the deterministic fallback; missing formal libraries fail strictly. |

## Fourteen-feature input/output contract

The order is:

1. gap pressure: `(w / w_max) * (G / Q)`;
2. timely-contribution eligibility: `1[J in Omega]`;
3. checkpoint urgency: `sigma((C_LB-t)/(tau-t))`;
4. quantity contribution: `min(q,G)/G`;
5. multi-checkpoint value: timely `sum(w*min(q,G)) / sum(w*G)`;
6. forecast deficit: `Delta_fc/G`;
7. capacity pressure: `Delta_cap/G_plus` when applicable, otherwise zero;
8. capacity applicability flag;
9. due slack: `sigma_signed((d-C_LB)/W_bar)`;
10. tardiness exposure: `sigma(max(0,C_LB-d)/W_bar)`;
11. trigger-machine processing time: `sigma(p/p_bar)`;
12. processing ratio: `p/(p+downstream minimum work)`;
13. remaining minimum work: `sigma(W/W_bar)`;
14. critical ratio: `sigma((d-t)/W)`, zero when `d<=t`.

The production identifier for this exact contract is `paper14`.  Historical
`theory11` and `theory11_plus3` remain separate compatibility interfaces so
previously serialized rules do not silently change mathematical meaning.

Here `sigma(x)=x/(1+x)` for nonnegative `x`, and
`sigma_signed(x)=x/(1+abs(x))`. Every zero denominator returns zero. If there
is no target checkpoint, features 1--8 are zero. Due slack is the only signed
feature.

## Manuscript-defined contracts versus unresolved constants

The following are mathematically defined and therefore executable now:

- exogenous commitments and hidden-job isolation;
- compound-Poisson aggregation order;
- all Sec4 quantities and target selection;
- the 14 feature formulas and ordering;
- graph dimensions and mathematical attributes;
- SPT/ATC `F_ref`, true incremental reward and rule fallbacks.

The following sec6 values remain author/TBD choices and are intentionally not
asserted as production constants:

- forecast bucket width and update interval;
- final forecast-risk `kappa`;
- the final checkpoint-calendar equation and any independent calendar jitter;
- the selected PPO discount factor from `{0.99, 0.999, 1}`;
- any result-table values, model API version, GP hyperparameters or stress-test
  sample counts.

The latest sec6 fixes `p_bar=E[U{1,...,50}]=25.5` and
`W_bar=E[U{2,...,5}]*p_bar=3.5*25.5=89.25`.  Synthetic tests may still supply
explicit `p_bar` and `W_bar` metadata to make hand arithmetic checkable; those
local numbers define only that fixture.

## Legacy-name compatibility boundary

| legacy English name | `paper14` position | compatibility classification |
|---|---:|---|
| `gap_pressure` | 1 | Same name, formula changed to weighted residual-gap pressure. |
| `forecast_deficit` | 6 | Position and formula changed; now the formal compound-Poisson shortfall ratio. |
| `checkpoint_slack_urgency` | 3 | Same position/name, target selection and `C_LB` semantics changed. |
| `multi_checkpoint_value` | 5 | Position and formula changed; denominator is weighted active positive gaps. |
| `tardiness_exposure` | 10 | Position and normalization changed from legacy horizon/tanh. |
| `due_date_slack` | 9 | Position and formula changed to signed `sigma_pm(S_d/W_bar)`. |
| `capacity_pressure` | 7 | Same position, formula changed to formal `Delta_cap/G_plus`. |
| `forecast_flag` | unsupported | Removed from the formal rule vector. |
| `capacity_flag` | 8 | Position changed; semantics are formal capacity applicability. |
| `quantity_contribution` | 4 | Position changed; formula retained subject to formal target selection. |
| `processing_ratio` | 12 | Position changed; formula retained. |
| `proc_time` | 11 | Position changed; normalization now uses fixed `p_bar`. |
| `remaining_work` | 13 | Same position, but current-operation minimum and fixed `W_bar` replace trigger-machine/instance scales. |
| `critical_ratio` | 14 | Same position, but denominator is formal minimum remaining work. |
| none | 2 | New `timely_eligibility`; it replaces no legacy feature semantically. |
| `mean_remaining_work` | unsupported | Legacy batch-only expression context, not a Table 5-2 feature. |

Consequently old `theory11`/`theory11_plus3` libraries remain historical and
must not be reported as formal `paper14` libraries without regeneration.

## Graph compatibility boundary

| graph config | Job | Machine | Checkpoint | status |
|---|---:|---:|---:|---|
| `legacy` | 11 or 14 | 2 | 8 | Historical default; preserves existing checkpoints and model artifacts. |
| `paper_graph` | 14 (`paper14`) | 2 | 10 | Formal Table 5-1 path; explicit selection only, with no shape inference or fallback. |

`paper_graph` constructs its Job matrix by calling the canonical `paper14`
builder and validates any caller-supplied matrix against that result.  The
existing HGT implementation accepts this graph when initialized explicitly
with `job_input_dim=14` and `checkpoint_input_dim=10`; its historical defaults
remain 11 and 8 and are therefore not compatible without those constructor
arguments.
