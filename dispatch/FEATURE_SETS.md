# Dispatch feature sets

`theory11` is the compatibility default and retains the original eleven
columns byte-for-byte. `theory11_plus3` appends the following columns:

| Feature | Raw ratio | Stored value |
|---|---|---|
| `proc_time` | trigger-machine processing time / visible-instance mean alternative processing time | `x / (1 + x)` |
| `remaining_work` | trigger-machine current-operation time plus downstream minimum work, divided by visible-instance mean total minimum job work | `x / (1 + x)` |
| `critical_ratio` | `max(0, due_date - now) / (remaining_minimum_work + epsilon)` | `x / (1 + x)` |

The rational compression `x / (1 + x)` is strictly increasing and invertible
on nonnegative inputs. It therefore preserves dispatch order while avoiding
the loss of discrimination caused by clipping every ratio above one. Its
inverse is `x / (1 - x)` (with a small numerical guard in expressions).

Consequently SPT is exactly represented by `-proc_time`. With `K=2`, the ATC
ordering uses the batch-context scalar `mean_remaining_work` (the mean
uncompressed normalized remaining work over current candidates) and is
represented by:

```text
math.exp(-max(0.0,
  (remaining_work / max(1e-12, 1.0 - remaining_work)) *
  ((critical_ratio / max(1e-12, 1.0 - critical_ratio)) - 1.0)
) / (2.0 * mean_remaining_work)) /
max(1e-12, remaining_work / max(1e-12, 1.0 - remaining_work))
```

All instance averages use only jobs visible at the current online decision,
so the extra features do not expose unreleased jobs.
