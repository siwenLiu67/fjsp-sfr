# PPO timing estimate

Each row is one 20-episode update using `theory11_plus3` and the pooled mixed library. No checkpoint is retained.

| Scale | Actions | Episodes | Seconds/episode | Seconds/update | Estimated 5000 episodes |
|---|---:|---:|---:|---:|---:|
| S | 5 | 20 | 0.130 | 2.610 | 0.18 h |
| M | 5 | 20 | 0.280 | 5.610 | 0.39 h |
| L | 4 | 20 | 0.651 | 13.016 | 0.90 h |
