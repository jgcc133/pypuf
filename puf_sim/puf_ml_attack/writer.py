"""Persistence helpers for machine-learning attack reports."""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports_ml_attacks"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _scenario_path(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._ -]+", "_", value.strip()).strip(" .")
    if not label:
        raise ValueError("scenario must contain at least one valid path character.")
    return label


def save_ml_attack_report(
    results: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    families: Sequence[str],
    scenario: str = "general",
    report_root: str | Path | None = None,
) -> Path:
    """Save a timestamped attack report under a scenario folder."""
    root = DEFAULT_REPORT_ROOT if report_root is None else Path(report_root)
    if os.name == "nt" and root.root and not root.drive:
        project_root = DEFAULT_REPORT_ROOT.parents[1]
        root = project_root / str(root).lstrip("/\\")
    root = root / _scenario_path(scenario)
    root.mkdir(parents=True, exist_ok=True)
    report_number = 1 + sum(path.is_dir() for path in root.iterdir())
    date_label = datetime.now().strftime("%Y %m %d")
    default_families = parameters.get("default_families", ())
    family_label = (
        "All PUFs" if set(families) == set(default_families) else "Selected PUFs"
    )
    report_directory = root / f"{date_label} Report {report_number} - {family_label}"
    report_directory.mkdir()

    metadata = dict(parameters)
    metadata.update({
        "families": list(families),
        "report_number": report_number,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_directory": str(report_directory),
    })
    with (report_directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(results), handle, indent=2)
    with (report_directory / "parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metadata), handle, indent=2)
    fields = ["family"] + sorted({key for report in results.values() for key in report})
    with (report_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, report in results.items():
            row = {"family": family}
            row.update({
                key: json.dumps(_json_value(value)) if isinstance(value, (dict, list)) else value
                for key, value in report.items()
            })
            writer.writerow(row)
    return report_directory


__all__ = ["DEFAULT_REPORT_ROOT", "save_ml_attack_report"]