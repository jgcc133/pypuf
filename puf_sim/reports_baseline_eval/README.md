# Baseline Evaluation Reports

`run_baseline_experiment()` automatically saves each baseline run under this
folder:

```text
puf_sim/reports_baseline_eval/
└── yyyy mm dd Report x - All PUFs - parameters and values/
    ├── parameters.json
    ├── results.json
    └── summary.csv
```

For a selected family subset, the folder label is `Selected PUFs` instead of
`All PUFs`. The folder name includes the date, sequential report number, and
these parameter values:

- `instances_per_family`
- `n`
- `samples`
- `repetitions`
- `input_noise`
- `seed`
- `k`
- `response_bits`
- `noisiness`

## Automatic saving

```python
from puf_sim.puf_baseline_eval import run_baseline_experiment

results = run_baseline_experiment(
    instances_per_family=100,
    n=64,
    samples=1000,
    repetitions=17,
    seed=20260921,
)
```

The function still returns the in-memory results dictionary. The same results
are saved to the newest report folder.

## Custom report location

```python
from pathlib import Path
from puf_sim.puf_baseline_eval import run_baseline_experiment

run_baseline_experiment(
    families=["arbiter", "xor_apuf"],
    instances_per_family=100,
    report_root=Path("./study_reports"),
)
```

This produces a `Selected PUFs` report under `study_reports`.

Disable persistence when needed:

```python
run_baseline_experiment(save_report=False)
```

## Files

- `parameters.json` records the run configuration and selected families.
- `results.json` stores the complete metric arrays and values.
- `summary.csv` stores one row per family, reducing array-valued metrics to
  their mean for convenient spreadsheet analysis.
