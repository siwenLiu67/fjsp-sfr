import csv
import json

import pytest

from fjsp_sfr.experiments.analyze_comparison import paired_table, rank_tests
from fjsp_sfr.experiments.build_comparison import build_comparison


def _write_run(root, method, seed, values):
    directory = root / "center" / method / f"seed_{seed}"
    directory.mkdir(parents=True)
    rows = [{"scenario_id": "center", "instance_id": index, "F": value, "TT": value, "WSF": 0, "solve_seconds": 0.1} for index, value in enumerate(values)]
    with (directory / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    mean = sum(values) / len(values)
    (directory / "summary.json").write_text(json.dumps({"means": {"F": mean, "TT": mean, "WSF": 0}, "training_seconds": 1, "solve_seconds_per_instance": 0.1, "decision_seconds_mean": 0.01}), encoding="utf-8")
    (directory / "COMPLETED").write_text("ok\n", encoding="utf-8")


def test_comparison_skips_missing_and_incomplete_runs(tmp_path):
    results, output = tmp_path / "results", tmp_path / "comparison"
    _write_run(results, "method_a", 1, [1, 2])
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"main": [{"id": "a", "label": "A", "directory": "method_a"}, {"id": "b", "label": "B", "directory": "method_b"}]}), encoding="utf-8")
    summary, long_path = build_comparison(results, "center", "main", output, groups)
    assert len(list(csv.DictReader(summary.open(encoding="utf-8")))) == 1
    assert len(list(csv.DictReader(long_path.open(encoding="utf-8")))) == 2


def test_paired_comparison_reports_strict_wins_and_ties():
    rows = []
    for method, values in (("proposed", [1, 2, 3]), ("other", [2, 2, 4])):
        rows.extend({"method": method, "scenario_id": "center", "instance_id": str(index), "F": str(value)} for index, value in enumerate(values))
    result = paired_table(rows, "proposed")[0]
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["tie_rate"] == pytest.approx(1 / 3)


def test_rank_pipeline_returns_one_rank_per_method():
    rows = []
    for method, values in (("a", [1, 2, 3]), ("b", [2, 3, 4]), ("c", [3, 4, 5])):
        rows.extend({"method": method, "scenario_id": "center", "instance_id": str(index), "F": str(value)} for index, value in enumerate(values))
    ranks, _wilcoxon, friedman = rank_tests(rows)
    assert [row["method"] for row in ranks] == ["a", "b", "c"]
    assert friedman["n"] == 3
