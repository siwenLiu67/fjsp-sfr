# FJSP-SFR

Research implementation for theory-guided symbolic dispatching rules and
streaming PPO rule selection in a flexible job-shop scheduling environment.

## Contents

- `core/`: simulation models, state, events, forecasts, and objectives.
- `dispatch/`: safe expression evaluation, symbolic rules, and theory features.
- `env/`: rule-selection environment and observations.
- `rl/`: heterogeneous graph encoder and PPO implementation.
- `experiments/`: validation, tuning, and streaming-training scripts.
- `tests/`: unit and end-to-end tests.
- `paper_sections/`: manuscript Sections 4-5 and the Sections 3-5 symbol table.

## Tests

From this directory:

```bash
PYTHONPATH=.. pytest -q
```

Generated model checkpoints and experiment logs are intentionally excluded
from version control.
