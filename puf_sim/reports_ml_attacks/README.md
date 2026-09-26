# Machine-Learning Attack Reports

`run_ml_attack_experiment()` stores reports here by scenario, using the same
date, sequence number, and family-selection naming as baseline evaluations:

```text
puf_sim/reports_ml_attacks/
└── <scenario>/
    └── yyyy mm dd Report x - All PUFs/
        ├── parameters.json
        ├── results.json
        └── summary.csv
```

Selected-family runs use `Selected PUFs`. Each results row records the winning
attack stage, nonlinear depth, ensemble size, validation and test imitation
accuracy, and timing for each attempted stage.