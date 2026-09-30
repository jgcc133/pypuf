import csv
import json
from types import SimpleNamespace

import numpy as np

import puf_sim.puf_ml_attack.experiment as experiment
from puf_sim.puf_ml_attack.experiment import _pair_accuracy
from puf_sim.puf_ml_attack.arbiter.xor_apuf import (
    XorTorchPUFModel,
    _XorParityModel,
)
from puf_sim.puf_ml_attack.arbiter.interpose import (
    InterposeCore,
    InterposeTorchPUFModel,
)
from puf_sim.puf_implementations import create_puf
from puf_sim.puf_ml_attack.writer import save_ml_attack_report


class EchoModel:
    def eval(self, challenges):
        return challenges[:, :1]


class FakeTorchModel:
    def __init__(self, depth):
        self.depth = depth

    def cpu(self):
        return self


def test_pair_accuracy_flattens_responses():
    challenges = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    dataset = SimpleNamespace(
        challenges=challenges,
        responses=np.array([[[1]], [[-1]]], dtype=np.int8),
    )

    assert _pair_accuracy(EchoModel(), dataset) == 1.0


def test_xor_inference_applies_training_feature_map():
    import torch

    puf = create_puf("xor_apuf", n=8, seed=17, k=2)
    challenges = np.array([
        [1, -1, 1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1],
    ], dtype=np.int8)
    model = _XorParityModel(8, puf.k, seed=0)
    with torch.no_grad():
        model.chain_logits.weight.copy_(
            torch.as_tensor(puf.weight_array[:, :-1], dtype=torch.float32)
        )
        model.chain_logits.bias.copy_(
            torch.as_tensor(puf.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = XorTorchPUFModel([model], 8, 1, torch.device("cpu"))

    assert np.array_equal(
        attack_model.eval(challenges).reshape(-1), puf.eval(challenges).reshape(-1)
    )


def test_interpose_model_matches_simulator_with_target_weights():
    import torch

    puf = create_puf("interpose", n=12, seed=27, k=2, interpose_pos=5)
    challenges = np.random.default_rng(28).choice((-1, 1), size=(32, 12)).astype(np.int8)
    model = InterposeCore(12, puf.down.k, puf.interpose_pos, seed=0)
    with torch.no_grad():
        model.up_weights.weight.copy_(
            torch.as_tensor(puf.up.weight_array[:, :-1], dtype=torch.float32)
        )
        model.up_weights.bias.copy_(
            torch.as_tensor(puf.up.weight_array[:, -1], dtype=torch.float32)
        )
        model.down_weights.weight.copy_(
            torch.as_tensor(puf.down.weight_array[:, :-1], dtype=torch.float32)
        )
        model.down_weights.bias.copy_(
            torch.as_tensor(puf.down.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = InterposeTorchPUFModel([model], 12, 1, torch.device("cpu"))

    assert np.array_equal(
        attack_model.eval(challenges).reshape(-1), puf.eval(challenges).reshape(-1)
    )


def test_train_until_threshold_reports_effort(monkeypatch, capsys):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

    validation_scores = iter([0.4, 0.8, 0.95])
    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_ArbiterAgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_ArbiterTorchPUFModel", AttackModel)
    monkeypatch.setattr(
        experiment,
        "_pair_accuracy",
        lambda model, dataset: next(validation_scores, 0.95)
        if dataset.seed == 11 else 0.98,
    )

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=10,
        family="arbiter",
        success_threshold=0.95,
        max_depth=1,
        agents=1,
        width=None,
        epochs=3,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
        print_progress=True,
    )

    output = capsys.readouterr().out
    assert result["threshold_reached"] is True
    assert result["epochs_to_threshold"] == 3
    assert result["epochs_trained"] == 3
    assert result["time_to_threshold_seconds"] >= 0
    assert result["test_accuracy"] == 0.98
    assert "epoch 1/3, validation accuracy 40.0000%" in output
    assert "epoch 3/3, validation accuracy 95.0000%" in output


def test_epoch_limit_applies_at_each_depth(monkeypatch):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.5)

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=10,
        family="ring_oscillator",
        success_threshold=1.0,
        max_depth=2,
        agents=2,
        width=None,
        epochs=3,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
    )

    assert [stage["depth"] for stage in result["stages"]] == [1, 2]
    assert [stage["epochs"] for stage in result["stages"]] == [3, 3]
    assert result["epochs_trained"] == 6
    assert result["epochs_per_depth_limit"] == 3


def test_cohort_counts_exact_matches():
    instance_results = [
        {"test_accuracy": 0.99, "epochs_trained": 2, "fit_elapsed_seconds": 1.0,
         "instance_matched": False},
        {"test_accuracy": 1.0, "epochs_trained": 3, "fit_elapsed_seconds": 1.5,
         "instance_matched": True},
        {"test_accuracy": 1.0, "epochs_trained": 4, "fit_elapsed_seconds": 2.0,
         "instance_matched": True},
    ]

    result = experiment._summarize_puf_instances(instance_results, success_threshold=0.66)

    assert result["puf_instances"] == 3
    assert result["matched_instances"] == 2
    assert result["required_matches"] == 2
    assert result["instance_match_rate"] == 2 / 3
    assert result["success"] is True
    assert result["instances_processed_to_threshold"] == 3
    assert result["epochs_to_threshold"] == 9
    assert result["time_to_threshold_seconds"] == 4.5


def test_instances_are_independent_and_aggregated(monkeypatch):
    created_seeds = []
    test_scores = iter([1.0, 0.99, 1.0])

    def create_target(family, **kwargs):
        created_seeds.append(kwargs["seed"])
        return SimpleNamespace(seed=kwargs["seed"])

    def attack_target(target, *args):
        assert args[5] == 0.66
        assert args[13] == "arbiter"
        return {
            "test_accuracy": next(test_scores),
            "epochs_trained": 1,
            "fit_elapsed_seconds": 0.5,
        }

    monkeypatch.setattr(experiment, "create_puf", create_target)
    monkeypatch.setattr(experiment, "_attack_one_puf", attack_target)
    monkeypatch.setattr(experiment, "_resolve_device", lambda backend, device: "cpu")

    results = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        puf_instances=3,
        success_threshold=0.66,
        backend="cpu",
        save_report=False,
    )

    result = results["arbiter"]
    assert len(set(created_seeds)) == 3
    assert result["matched_instances"] == 2
    assert result["threshold_reached"] is True
    assert result["epochs_to_threshold"] == 3


def test_report_layout(tmp_path):
    results = {
        "arbiter": {
            "attack": "nonlinear_ensemble",
            "depth": 1,
            "agents": 3,
            "test_accuracy": 0.97,
            "success": True,
            "stages": [{"depth": 0}, {"depth": 1}],
        }
    }
    report_directory = save_ml_attack_report(
        results,
        {"default_families": ["arbiter"]},
        ["arbiter"],
        scenario="iot",
        report_root=tmp_path,
    )

    assert report_directory.parent.name == "iot"
    assert report_directory.name.endswith("Report 1 - All PUFs")
    assert {path.name for path in report_directory.iterdir()} == {
        "parameters.json", "results.json", "summary.csv"
    }
    assert json.loads((report_directory / "results.json").read_text()) == results
    with (report_directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["family"] == "arbiter"
    assert row["test_accuracy"] == "0.97"
