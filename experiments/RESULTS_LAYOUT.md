# Formal experiment result layout

## Scenario and seed partitions

Scenario factors are defined once in `generation/scenario_config.json`. The center scenario uses 10 machines, 20 initial jobs, 50 Poisson-arrival jobs, target load 0.80 and `ddt=1.0`; `l9_01` through `l9_09` are the fixed L9 cells in the same file.

The new-scenario partitions are pairwise disjoint and separate from all legacy S/M/L/X ranges: training stream 20000–24999, center validation 25000–25049, center calibration 25100–25129, center selection 25200–25219, center holdout 25300–25319, ten-scenario calibration overview 25400–25409, center test 26000–26099, and L9 test blocks 27000–27049, 27100–27149, …, 27800–27849.

## Directory structure

```text
results/
  center/<number_algorithm[_variant]>/seed_<k>/
  l9/<number_algorithm[_variant]>/seed_<k>/
    model_last.pt          # learning methods; trained on center only
    per_instance.csv       # scenario_id, instance_id, F, TT, WSF, solve_seconds
    train_curve.csv        # update, episodes, mean_train_F, val_F_greedy, wall_seconds
    summary.json           # aggregate quality and timing fields
    manifest.json          # immutable configuration, hashes, seeds
    run.log
    COMPLETED              # written last; the sole resume marker
```

An existing `COMPLETED` marker skips the run. A directory without that marker is incomplete and is rerun, overwriting its task-local outputs.

## Rule-library names

- `classic4`: SPT, EDD, ATC(K=2), MIN_SLACK.
- `pool_center_g`: quality-gated pooled LLM rules only.
- `pool_center_mix`: SPT and ATC plus complementary quality-gated LLM rules; default learning library.
- `handwritten_center`: the best six existing handwritten rules on center selection seeds.
- `random_center_nogate` and `random_center_nogate_mix`: random-rule source ablations without a quality gate.

`experiments/groups.json` is the single method-to-directory mapping used by comparison builders. It defines `main`, `ablation_library`, and `ablation_network`; absent or incomplete run directories are skipped rather than interpreted as failed observations.

## Commands

Train and evaluate on center:

```bash
python3 experiments/run_algo.py --scenario center --eval_set center --algo hgt_ppo --library pool_center_mix --seeds 1 2 3
```

Evaluate the trained center models on all L9 cells without retraining:

```bash
python3 experiments/run_algo.py --scenario center --eval_set l9 --algo hgt_ppo --library pool_center_mix --seeds 1 2 3
```

Rule-only methods:

```bash
python3 experiments/run_algo.py --algo spt --eval_set center
python3 experiments/run_algo.py --algo edd --eval_set center
python3 experiments/run_algo.py --algo atc --eval_set center
python3 experiments/run_algo.py --algo minslack --eval_set center
python3 experiments/run_algo.py --algo best_single --library pool_center_mix --eval_set center
python3 experiments/run_algo.py --algo random_select --library pool_center_mix --eval_set center
python3 experiments/run_algo.py --algo rolling_ga --eval_set center --ga_time_limit 3.0 --workers 4
python3 experiments/run_algo.py --algo hgt_ddqn --library pool_center_mix --eval_set center
python3 experiments/run_algo.py --algo hgt_a2c --library pool_center_mix --eval_set center
python3 experiments/run_algo.py --algo hgt_rainbow --library pool_center_mix --eval_set center
```

Rolling-GA timing probe:

```bash
python3 experiments/run_algo.py --algo rolling_ga --eval_set center --ga_time_probe --n_instances 3
```

Comparison aggregation and analysis:

```bash
python3 experiments/build_comparison.py --set center --group main
python3 experiments/analyze_comparison.py --set center --group main
python3 experiments/build_comparison.py --set l9 --group main
python3 experiments/analyze_comparison.py --set l9 --group main
```

Selector and information ablations:

```bash
python3 experiments/run_algo.py --algo mlp_ppo --library pool_center_mix --eval_set center
python3 experiments/run_algo.py --algo hgt_ppo --library pool_center_mix --mask checkpoint --eval_set center
python3 experiments/run_algo.py --algo hgt_ppo --library pool_center_mix --mask forecast --eval_set center
```

The learning-rate sweep uses center validation only and never reads either test partition:

```bash
python3 experiments/hparam_sweep.py --jobs 1
```
