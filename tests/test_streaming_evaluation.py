import argparse
import csv

import pytest
import torch

from fjsp_sfr.experiments import compare_runs, train_streaming


def _evaluation(values):
    return train_streaming.PolicyEvaluation(values=list(values), usage={})


def test_write_test_results_csv_exports_aligned_rows(tmp_path):
    path = tmp_path / "per_instance.csv"
    train_streaming.write_test_results_csv(
        path,
        seeds=(200, 201),
        fixed_results={"rule_a": [10.0, 20.0]},
        random_values=[11.0, 21.0],
        best_greedy=_evaluation([9.0, 19.0]),
        best_sampled=_evaluation([9.5, 19.5]),
        last_greedy=_evaluation([8.0, 18.0]),
        last_sampled=_evaluation([8.5, 18.5]),
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "seed",
        "rule_a",
        "random_mean_10",
        "best_greedy",
        "best_sampled_mean_5",
        "last_greedy",
        "last_sampled_mean_5",
    ]
    assert rows[1]["seed"] == "201"
    assert float(rows[1]["last_greedy"]) == 18.0


def test_write_test_results_csv_adds_tt_and_wsf_when_available(tmp_path):
    path = tmp_path / "breakdown.csv"
    best = train_streaming.PolicyEvaluation(
        values=[9.0], usage={}, tt_values=[4.0], wsf_values=[5.0]
    )
    other = _evaluation([10.0])
    train_streaming.write_test_results_csv(
        path,
        seeds=(200,),
        fixed_results={"rule_a": [10.0]},
        random_values=[11.0],
        best_greedy=best,
        best_sampled=other,
        last_greedy=other,
        last_sampled=other,
    )
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["best_greedy_TT"]) == 4.0
    assert float(row["best_greedy_WSF"]) == 5.0


def test_eval_only_validates_metadata_and_does_not_modify_checkpoints(
    tmp_path, monkeypatch
):
    library, _all_six = train_streaming.select_rule_library("top3")
    state = {
        "agent_state_dict": train_streaming.new_agent(
            library.action_size()
        ).state_dict(),
        "library_name": "top3",
        "seed": 42,
        "base_seed": 1000,
    }
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    torch.save(state, best_path)
    torch.save(state, last_path)
    before = (best_path.read_bytes(), last_path.read_bytes())
    captured = {}
    monkeypatch.setattr(train_streaming, "TEST_SEEDS", (200,))
    monkeypatch.setattr(train_streaming, "make_stream_instance", lambda seed: seed)
    monkeypatch.setattr(
        train_streaming,
        "print_test_report",
        lambda **kwargs: captured.update(kwargs),
    )
    args = argparse.Namespace(
        library="top3",
        library_file=None,
        seed=42,
        base_seed=1000,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        test_csv_path=str(tmp_path / "results.csv"),
    )

    train_streaming.run_evaluation_only(args)

    assert captured["test_instances"] == [200]
    assert captured["output_csv"] == tmp_path / "results.csv"
    assert (best_path.read_bytes(), last_path.read_bytes()) == before


def test_eval_only_rejects_library_mismatch():
    with pytest.raises(ValueError, match="library mismatch"):
        train_streaming.validate_checkpoint_metadata(
            {"library_name": "all6"},
            library_name="top3",
            library_source={"kind": "builtin", "name": "top3"},
            seed=42,
            base_seed=1000,
            checkpoint_label="best checkpoint",
        )


def test_compare_runs_aligns_seeds_and_computes_blocks(tmp_path):
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    for path, offset in ((left_path, 0.0), (right_path, 1.0)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("seed", "score"))
            for seed in range(200, 500):
                writer.writerow((seed, seed + offset))

    left = compare_runs.read_method_series("left", left_path, "score")
    right = compare_runs.read_method_series("right", right_path, "score")
    seeds, left_values, right_values = compare_runs.aligned_values(left, right)

    assert seeds == list(range(200, 500))
    assert all(left < right for left, right in zip(left_values, right_values))
    blocks = compare_runs.block_means(left_values)
    assert len(blocks) == 10
    assert blocks[1] - blocks[0] == pytest.approx(30.0)
