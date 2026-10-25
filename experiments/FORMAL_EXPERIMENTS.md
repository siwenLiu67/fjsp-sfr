# Formal experiment protocol

All trainable selectors use 5,000 streaming episodes, 32 episodes per PPO
update, `gamma=0.99`, `gae_lambda=0.95`, four PPO epochs, reward scale 10,
learning rate `3e-4 -> 1e-4`, and entropy coefficient `0.02 -> 0.002`.
HGT and MLP actor-critics therefore differ only in their state encoder. DQN
uses the same HGT encoder, discount and episode budget, with replay memory,
a target network and linearly decaying epsilon-greedy exploration.

Selector input ablations do not alter dispatching rules. `--mask-checkpoint`
removes checkpoint nodes and job-checkpoint edges from the selector graph.
`--mask-forecast` zeros the selector's job forecast deficit/flag, checkpoint
forecast-shortfall ratio/flag, and global forecast flag. Rule expressions
continue to receive the complete, unmasked `theory11_plus3` factors.

Test seeds are S=10000--10099, M=10100--10199, L=10200--10299 and
X=10300--10399. They are disjoint from validation 100--149, selection
600--619, holdout 700--719, and streaming training 1000--5999.

Commands:

```bash
bash experiments/run_formal.sh --list
bash experiments/run_formal.sh --smoke
bash experiments/run_formal.sh
python3 experiments/aggregate_formal.py
```

Each run owns one directory under `experiments/formal/` containing a log,
manifest, checkpoint when applicable, per-instance CSV and summary JSON.
The matrix skips a run whenever its summary already exists.
