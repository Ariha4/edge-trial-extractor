import json

import pytest

from trialx.evaluate import DataError, load_examples


def make_example(nct_id="NCT00000001", **label_overrides):
    labels = {
        "study_type": "INTERVENTIONAL",
        "intervention_types": ["DEVICE"],
        "interventions": ["Deep Brain Stimulation"],
        "conditions": ["Parkinson Disease"],
    }
    labels.update(label_overrides)
    return {"nct_id": nct_id, "text": "DBS for Parkinson Disease", "labels": labels}


def write_jsonl(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def test_loads_valid_examples_and_skips_blank_lines(tmp_path):
    path = write_jsonl(tmp_path / "test.jsonl", [json.dumps(make_example("NCT1")), "",
                                                 json.dumps(make_example("NCT2"))])
    assert [example["nct_id"] for example in load_examples(path)] == ["NCT1", "NCT2"]


def test_limit_stops_early(tmp_path):
    path = write_jsonl(tmp_path / "test.jsonl", [json.dumps(make_example(f"NCT{i}")) for i in range(5)])
    assert len(load_examples(path, limit=2)) == 2


def test_negative_limit_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "test.jsonl", [json.dumps(make_example())])
    with pytest.raises(ValueError, match="limit"):
        load_examples(path, limit=-1)


def test_bad_json_reports_file_and_line(tmp_path):
    path = write_jsonl(tmp_path / "test.jsonl", [json.dumps(make_example()), "{not json"])
    with pytest.raises(DataError, match=r"test\.jsonl:2: not valid JSON"):
        load_examples(path)


@pytest.mark.parametrize("example, message", [
    (["a", "list"], "expected a JSON object"),
    ({**make_example(), "text": "  "}, "'text' must be a non-empty string"),
    ({k: v for k, v in make_example().items() if k != "nct_id"}, "'nct_id'"),
    ({**make_example(), "labels": None}, "'labels' must be an object"),
    (make_example(study_type="interventional"), "labels.study_type"),
    (make_example(conditions="Parkinson Disease"), "labels.conditions must be a list"),
    (make_example(interventions=["DBS", 3]), "labels.interventions must be a list of strings"),
])
def test_malformed_examples_are_rejected(tmp_path, example, message):
    path = write_jsonl(tmp_path / "test.jsonl", [json.dumps(example)])
    with pytest.raises(DataError, match=message):
        load_examples(path)


def test_empty_file_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "test.jsonl", [])
    with pytest.raises(DataError, match="no examples"):
        load_examples(path)


def test_committed_test_split_is_valid():
    assert len(load_examples("data/test.jsonl")) == 332
