"""Benchmark harness tests with a fake runner, so no model is downloaded or loaded."""
import argparse
import csv
import json

import pytest

from trialx import evaluate
from trialx.evaluate import percentile, run, time_generations
from trialx.prompts import format_target


def make_example(nct_id, condition):
    labels = {
        "study_type": "INTERVENTIONAL",
        "intervention_types": ["DEVICE"],
        "interventions": ["Deep Brain Stimulation"],
        "conditions": [condition],
    }
    return {"nct_id": nct_id, "text": f"DBS for {condition}", "labels": labels}


EXAMPLES = [make_example("NCT1", "Parkinson Disease"), make_example("NCT2", "Essential Tremor")]


class EchoRunner:
    """Answers perfectly for every example and records which prompts it saw."""

    def __init__(self, examples):
        self.answers = {example["text"]: format_target(example["labels"]) for example in examples}
        self.calls = []

    def generate(self, messages):
        trial_text = messages[-1]["content"].split("\n", 1)[1]
        self.calls.append(trial_text)
        return self.answers[trial_text], 10


def make_args(tmp_path, run_name="fake", **overrides):
    data_path = tmp_path / "test.jsonl"
    data_path.write_text("".join(json.dumps(example) + "\n" for example in EXAMPLES), encoding="utf-8")
    values = dict(run_name=run_name, backend="gguf", model=str(tmp_path / "missing.gguf"), data=str(data_path),
                  limit=0, warmup=1, dtype="float16", device="cpu", threads=2,
                  results_dir=str(tmp_path / "results"))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_percentile_uses_nearest_rank():
    assert percentile([0.3, 0.1, 0.2], 0.5) == 0.2
    assert percentile(list(range(1, 101)), 0.95) == 95
    assert percentile([7.0], 0.95) == 7.0


def test_warmup_generations_are_not_timed(monkeypatch):
    runner = EchoRunner(EXAMPLES)
    ticks = iter([0.0, 1.0, 1.0, 3.0])   # two timed calls: 1 s and 2 s
    monkeypatch.setattr(evaluate.time, "perf_counter", lambda: next(ticks))

    outputs, timing = time_generations(runner, EXAMPLES, warmup=1)

    assert len(runner.calls) == 3                   # 1 warmup + 2 timed
    assert len(outputs) == 2
    assert timing["mean_seconds_per_example"] == 1.5
    assert timing["p95_seconds_per_example"] == 2.0
    assert timing["tokens_per_second"] == pytest.approx(20 / 3)


def test_run_scores_and_writes_results(tmp_path):
    metrics = run(make_args(tmp_path), runner=EchoRunner(EXAMPLES))

    assert metrics["overall"] == 1.0
    assert metrics["examples"] == 2
    details = json.loads((tmp_path / "results" / "fake.json").read_text(encoding="utf-8"))
    assert [output["nct_id"] for output in details["outputs"]] == ["NCT1", "NCT2"]
    assert len(details["run"]["data_sha256"]) == 64
    assert details["run"]["warmup"] == 1
    assert details["run"]["threads"] == 2
    with open(tmp_path / "results" / "summary.csv", newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert [row["run_name"] for row in rows] == ["fake"]
    assert float(rows[0]["overall"]) == 1.0


def test_cli_rejects_missing_gguf_before_loading(tmp_path, capsys):
    args = make_args(tmp_path)
    with pytest.raises(SystemExit):
        evaluate.main(["--run-name", "x", "--backend", "gguf", "--model", args.model, "--data", args.data])
    assert "is not a file" in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--limit", "-1"), ("--warmup", "-2"), ("--threads", "0")])
def test_cli_rejects_bad_numbers(flag, value, capsys):
    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["--run-name", "x", "--backend", "hf", "--model", "m", flag, value])
    assert flag in capsys.readouterr().err
