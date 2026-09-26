import csv
import json
from types import SimpleNamespace

import numpy as np

from puf_sim.puf_ml_attack.experiment import _pair_accuracy
from puf_sim.puf_ml_attack.writer import save_ml_attack_report


class EchoModel:
    def eval(self, challenges):
        return challenges[:, :1]


def test_pair_accuracy_flattens_pypuf_response_axis():
    challenges = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    dataset = SimpleNamespace(
        challenges=challenges,
        responses=np.array([[[1]], [[-1]]], dtype=np.int8),
    )

    assert _pair_accuracy(EchoModel(), dataset) == 1.0


def test_ml_attack_report_uses_scenario_and_baseline_file_layout(tmp_path):
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
