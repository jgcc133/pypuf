# puf_sim

`puf_sim` provides a common workflow for creating PUF simulation instances
and evaluating their baseline performance.

The package is organized into two submodules:

- `puf_sim.puf_implementations`: native `pypuf` PUFs and analytical models for
  the requested PUF families.
- `puf_sim.puf_baseline_eval`: official `pypuf.metrics` calls plus derived
  baseline metrics such as uniformity, entropy-based randomness, bit aliasing,
  diffuseness, Hamming-distance distributions, and probability of
  misidentification.

## Use cases

### 1. Generate simulated PUF family instances

Use `create_puf()` to create one instance from a common factory interface:

```python
from puf_sim.puf_implementations import create_puf
from pypuf.io import random_inputs

puf = create_puf(
    "xor_apuf",
    n=64,
    k=4,
    seed=7,
    noisiness=0.05,
)

challenges = random_inputs(n=puf.challenge_length, N=100, seed=8)
responses = puf.eval(challenges)
print(responses.shape)
```

Supported family names include:

```text
optical
arbiter
ff_apuf
xor_apuf
interpose
permutation
ring_oscillator
sram
memristive
silicon_photonic
```

Create a population of devices by changing the device seed:

```python
from puf_sim.puf_implementations import create_puf

population = [
    create_puf("arbiter", n=64, seed=device_seed)
    for device_seed in range(10)
]
```

The native `pypuf` implementations are used for Optical, Arbiter,
Feed-Forward Arbiter, XOR Arbiter, Interpose, and Permutation PUFs. Ring
Oscillator, SRAM, Memristive, and Silicon Photonic PUFs use deterministic
analytical evaluation models because the base library does not include
hardware-level implementations for those families.

### 2. Baseline evaluation over 10,000 simulated instances per family

A family-level baseline should evaluate one common challenge set across a
population of 10,000 independently seeded instances. The population is then
summarized using the baseline criteria:

- uniqueness
- steadiness and reliability
- uniformity and randomness
- bit aliasing
- diffuseness
- probability of misidentification
- similarity, bias, influence, total influence, and noise sensitivity when
  applicable

The complete report is returned as a dictionary:

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_metrics
from puf_sim.puf_implementations import create_puf

family = "arbiter"
instance_count = 10_000
challenge_count = 1_000

population = [
    create_puf(family, n=64, seed=device_seed)
    for device_seed in range(instance_count)
]

report = evaluate_exposed_metrics(
    population[0],
    instances=population,
    samples=challenge_count,
    repetitions=17,
    input_noise=0.01,
    seed=20260921,
)

print("family:", family)
print("uniqueness:", report["uniqueness"].mean())
print("steadiness:", report["steadiness"].mean())
print("reliability:", report["reliability"].mean())
print("uniformity:", report["uniformity"])
print("randomness:", report["randomness"])
print("bit aliasing:", report["bit_aliasing"])
print("diffuseness:", report["diffuseness"])
print(
    "probability of misidentification:",
    report["probability_of_misidentification"],
)
```

### Run the 100-instance all-family baseline in PowerShell

The following command calls `run_baseline_experiment()` for all ten canonical
families, using 100 independently seeded instances per family. It prints the
main baseline metrics for each family:

```powershell
& ".\puf_env\Scripts\python.exe" -c "from puf_sim.puf_baseline_eval import run_baseline_experiment; results=run_baseline_experiment(instances_per_family=100, n=64, samples=1000, repetitions=17, seed=20260921); [print(f'{family}: uniqueness={report[\"uniqueness\"].mean():.4f} steadiness={report[\"steadiness\"].mean():.4f} reliability={report[\"reliability\"].mean():.4f} uniformity={report[\"uniformity\"]:.4f} randomness={report[\"randomness\"]:.4f} bit_aliasing={report[\"bit_aliasing\"]:.4f} diffuseness={report[\"diffuseness\"]:.4f} misidentification={report[\"probability_of_misidentification\"]:.6f}') for family, report in results.items()]"
```

For a quicker smoke run, reduce the challenge sample count and repeated
evaluations while retaining 100 instances per family:

```powershell
& ".\puf_env\Scripts\python.exe" -c "from puf_sim.puf_baseline_eval import run_baseline_experiment; results=run_baseline_experiment(instances_per_family=100, n=64, samples=32, repetitions=3, seed=20260921); [print(family, report['uniformity'], report['randomness']) for family, report in results.items()]"
```

For a full study, repeat the same operation for every family:

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_metrics
from puf_sim.puf_implementations import create_puf

families = [
    "optical",
    "arbiter",
    "ff_apuf",
    "xor_apuf",
    "interpose",
    "permutation",
    "ring_oscillator",
    "sram",
    "memristive",
    "silicon_photonic",
]

for family in families:
    population = [
        create_puf(family, n=64, k=2, response_bits=8, seed=device_seed)
        for device_seed in range(10_000)
    ]
    report = evaluate_exposed_metrics(
        population[0],
        instances=population,
        samples=1_000,
        repetitions=17,
        seed=20260921,
    )
    print(
        family,
        float(report["uniqueness"].mean()),
        float(report["reliability"].mean()),
        report["uniformity"],
        report["randomness"],
        report["bit_aliasing"],
        report["diffuseness"],
        report["probability_of_misidentification"],
    )
```

## Command line usage

The implementation package has a command-line smoke runner. From the
repository root, list available family names:

```powershell
python -m puf_sim.puf_implementations --list
```

Create and evaluate a simulated 4-XOR 64-bit Arbiter PUF:

```powershell
python -m puf_sim.puf_implementations `
    --family xor_apuf `
    --bits 64 `
    --chains 4 `
    --samples 100 `
    --seed 7
```

The baseline report can be called from a shell using Python's `-c` option:

```powershell
python -c "from puf_sim.puf_implementations import create_puf; from puf_sim.puf_baseline_eval import evaluate_exposed_metrics; p=create_puf('arbiter', n=64, seed=7); r=evaluate_exposed_metrics(p, samples=1000, repetitions=17, seed=8); print('uniformity=', r['uniformity']); print('randomness=', r['randomness']); print('reliability=', r['reliability'].mean())"
```

A 10,000-instance study is more conveniently run as a Python script so that
results can be saved and each family can be processed independently:

```powershell
python run_baseline_study.py --instances 10000 --families all --bits 64 --samples 1000 --output baseline_results.json
```

The command above assumes a study runner has been saved as
`run_baseline_study.py`. The repository APIs used by that runner are
`create_puf()` and `evaluate_exposed_metrics()` as shown in the Python
examples above.

## Metric API layers

Call only the official metrics when a raw API result is needed:

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_api

api_metrics = evaluate_exposed_api(
    puf,
    instances=population,
    samples=1000,
    repetitions=17,
    input_noise=0.01,
    seed=8,
)
```

Call individual derived response-data helpers when responses have already
been collected:

```python
from puf_sim.puf_baseline_eval import (
    binary_entropy,
    bit_aliasing,
    hamming_distance_distribution,
    probability_of_misidentification,
)

randomness = binary_entropy(responses)
aliasing = bit_aliasing(population_responses)
distances = hamming_distance_distribution(responses)
misidentification = probability_of_misidentification(population_responses)
```

Use `evaluate_exposed_metrics()` when one result dictionary containing both
layers is preferred.

## Scaling notes

A 10,000-instance study can be memory-intensive because uniqueness and
misidentification compare a population response matrix. Start with a small
pilot such as 10 or 100 devices, confirm the family configuration, then scale
to 10,000. Process one family at a time and save each report rather than
keeping all families' populations in memory simultaneously.

The `PermutationPUF` implementation relies on the supported challenge lengths
provided by the underlying `pypuf` permutation tables. A challenge length of
64 is supported by the base library and is used in the examples above.
