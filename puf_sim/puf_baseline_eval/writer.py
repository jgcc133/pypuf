"""Persistence helpers for baseline evaluation reports."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports_baseline_eval"


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


def _summary_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if np.issubdtype(value.dtype, np.number):
            return float(np.mean(value))
        return str(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None:
        return ""
    return value


def save_baseline_report(
    results: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    families: Sequence[str],
    report_root: Path | str | None = None,
) -> Path:
    """Save one timestamped baseline report in the reports data directory."""
    root = DEFAULT_REPORT_ROOT if report_root is None else Path(report_root)
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

    summary_fields = ["family"] + sorted(
        {key for report in results.values() for key in report}
    )
    with (report_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for family, report in results.items():
            row = {"family": family}
            row.update({key: _summary_value(value) for key, value in report.items()})
            writer.writerow(row)

    return report_directory


__all__ = ["DEFAULT_REPORT_ROOT", "save_baseline_report"]
