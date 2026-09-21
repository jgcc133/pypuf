# PUF Implementations

This submodule provides one import location for the PUF families used by
`puf_sim`. Every implementation follows the `pypuf.simulation.Simulation`
interface:

- `challenge_length`
- `response_length`
- `eval(challenges)`

The native families delegate to existing `pypuf` implementations:

- Optical PUF
- Arbiter PUF
- Feed-Forward Arbiter PUF (FF-APUF)
- XOR Arbiter PUF (XOR-APUF)
- Interpose PUF
- Permutation PUF

The following families use deterministic analytical models because the base
`pypuf` library does not provide hardware-level implementations for them:

- Ring Oscillator PUF
- Static Random Access Memory PUF
- Memristive PUF
- Silicon Photonic PUF

These analytical models are useful for software evaluation and comparison;
they are not substitutes for measurements from physical devices.

## Python API

Import the factory and create a 64-bit Arbiter PUF:

```python
from puf_sim.puf_implementations import create_puf
from pypuf.io import random_inputs

puf = create_puf("arbiter", n=64, seed=7)
challenges = random_inputs(n=puf.challenge_length, N=10, seed=8)
responses = puf.eval(challenges)

print(responses.shape)
```

Create a 4-XOR Arbiter PUF, an FF-APUF, and an Interpose PUF:

```python
from puf_sim.puf_implementations import create_puf

xor_puf = create_puf("xor_apuf", n=64, k=4, seed=1, noisiness=0.05)
ff_puf = create_puf(
    "ff_apuf",
    n=64,
    seed=2,
    ff=[(20, 40)],
)
interpose_puf = create_puf(
    "interpose",
    n=64,
    k=4,
    seed=3,
    interpose_pos=32,
)
```

Create the other families with the same factory:

```python
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
    puf = create_puf(family, n=64, seed=10, k=2, response_bits=8)
    print(family, puf.challenge_length, puf.response_length)
```

`k` controls the number of chains for XOR, Interpose, and Permutation PUFs.
`response_bits` controls the response width of the analytical families and the
number of optical response pixels. `noisiness` applies to native noisy Arbiter
PUF-based implementations.

The native classes are also available directly:

```python
from puf_sim.puf_implementations import (
    ArbiterPUF,
    FeedForwardArbiterPUF,
    InterposePUF,
    OpticalPUF,
    PermutationPUF,
    XORArbiterPUF,
)
from puf_sim.puf_implementations import (
    MemristivePUF,
    RingOscillatorPUF,
    SiliconPhotonicPUF,
    StaticRandomAccessMemoryPUF,
)

puf = RingOscillatorPUF(n=64, response_bits=8, seed=11)
```

## Command line

The package includes a small command-line smoke runner. From the repository
root, list the supported family names:

```powershell
python -m puf_sim.puf_implementations --list
```

Create and evaluate a 64-bit 4-XOR Arbiter PUF on 100 challenges:

```powershell
python -m puf_sim.puf_implementations `
    --family xor_apuf `
    --bits 64 `
    --chains 4 `
    --samples 100 `
    --seed 7
```

Create an analytical SRAM PUF with 32 response bits:

```powershell
python -m puf_sim.puf_implementations `
    --family sram `
    --bits 64 `
    --response-bits 32 `
    --samples 100
```

The command prints the selected family, challenge count, response width, and
response array shape. It is intended as a quick integration check; detailed
metric evaluation is provided by `puf_sim.eval_utils` and
`puf_sim.puf_test`.

## Permutation PUF note

The underlying `pypuf` Permutation PUF uses predefined permutation tables for
specific challenge lengths. Use a supported challenge length, such as `64`,
when calling `create_puf("permutation", ...)`.
