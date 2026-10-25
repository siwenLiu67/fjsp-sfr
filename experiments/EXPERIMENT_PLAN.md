# Formal experiment plan

## Main comparison

The main group comprises SPT, EDD, ATC, MIN_SLACK, the validation-selected best fixed rule, rolling-horizon GA, HGT-DDQN, HGT-A2C, HGT-Rainbow and HGT-PPO. The four dispatch rules are transparent local heuristics; `best_single` isolates library quality without state-dependent selection; rolling GA is an online search baseline; DDQN, A2C and Rainbow separate the effect of the learning objective; HGT-PPO is the proposed selector.

## Rule-library ablation

The complete method is HGT-PPO with `pool_center_mix`. Its rule-source controls are `pool_center_g` (LLM only), `handwritten_center`, `random_center_nogate_mix`, and `classic4`. All use the same state, action semantics, reward and episode budget.

## Network and input ablation

The complete method is compared with MLP-PPO on `pool_center_mix`, HGT-PPO without checkpoint nodes and job–checkpoint edges, HGT-PPO with forecast inputs zeroed, and uniform random rule selection. Information masks affect only selector observations; symbolic and classic rule calculations remain unchanged.

## Generalization design

All learning methods train only on freshly generated center instances. Center validation selects hyperparameters and center selection chooses fixed rules/libraries. Final center testing uses 100 untouched instances. The nine L9 cells cross machines `{5,10,20}`, new jobs `{20,50,100}`, load `{0.65,0.80,0.95}` and due-date tightness `{0.5,1.0,1.5}`; each cell has its own untouched 50-instance block and receives the frozen center-trained model without retraining.
