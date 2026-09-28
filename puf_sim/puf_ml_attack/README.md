# PUF Machine-Learning Attacks

`run_ml_attack_experiment()` creates `puf_instances` independent PUFs per
family (1,000 by default) and trains a separate model for each instance. The
model starts with logistic regression, then escalates through ensembles with
more hidden nonlinear layers. Each stage runs for `epochs`; the deepest stage
continues until every response vector in that instance's validation set matches
exactly, or `epochs` is reached at that depth. An instance counts as matched only
if every response vector in its held-out test set matches exactly, including all
response bits. The cohort succeeds when at least
`ceil(puf_instances * success_threshold)` instances match (95% by default).

Each family result reports `matched_instances`, `instance_match_rate`, and the
required match count. `epochs_to_threshold` and `time_to_threshold_seconds`
accumulate the training effort through the instance where the cohort threshold
was first reached. All configured instances are still evaluated for the final
cohort result. `epochs` is the per-depth limit for each instance.

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
    puf_instances=1000,
    backend="auto",
    scenario="iot",
)
```

Run the same orchestration from PowerShell:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack --families arbiter xor_apuf interpose ring_oscillator sram memristive --scenario medeqpt --report-root puf_sim/reports_ml_attacks --n 64 --training-samples 1600 --validation-samples 200 --test-samples 200 --puf-instances 1000 --seed 2026092602 --k 4 --response-bits 16 --noisiness 0.01 --success-threshold 0.95 --max-depth 3 --agents 3 --epochs 100 --batch-size 256 --learning-rate 0.001 --backend torch_xpu --device 0
```

Pass `--puf-instances` to change the independent PUF cohort size. The `--epochs`
limit applies separately at each model depth and PUF instance.

`backend` accepts `auto` (prefer Intel XPU when available), `torch_xpu`, or
`cpu`; `device` optionally selects an XPU index. PyTorch is required for
training. Each run is saved below `puf_sim/reports_ml_attacks/<scenario>/` using the
baseline report naming convention. A report contains `parameters.json`,
`results.json`, and `summary.csv`. The test set is separate from the validation
set used to choose the attack stage.