# PUF Machine-Learning Attacks

`run_ml_attack_experiment()` first trains a vanilla logistic classifier. If its
validation challenge-response imitation rate is below the target, it trains
ensembles of independently initialized neural agents with one, two, and
successively more hidden nonlinear layers. It stops at the first successful
stage or at `max_depth`. Reported imitation accuracy requires every response
bit for a challenge to match.

```python
from puf_sim.puf_ml_attack import run_ml_attack_experiment

results = run_ml_attack_experiment(
    families=["arbiter", "xor_apuf"],
    n=32,
    training_samples=5000,
    validation_samples=1000,
    test_samples=1000,
    success_threshold=0.95,
    max_depth=3,
    agents=3,
    backend="auto",
    scenario="iot",
)
```

Run the same orchestration from PowerShell:

```powershell
python -m puf_sim.puf_ml_attack --families arbiter xor_apuf --n 32 --scenario iot
```

`backend` accepts `auto` (prefer Intel XPU when available), `torch_xpu`, or
`cpu`; `device` optionally selects an XPU index. PyTorch is required for
training. Each run is saved below `puf_sim/reports_ml_attacks/<scenario>/` using the
baseline report naming convention. A report contains `parameters.json`,
`results.json`, and `summary.csv`. The test set is separate from the validation
set used to choose the attack stage.