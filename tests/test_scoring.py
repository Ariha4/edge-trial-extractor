import json

from trialx.scoring import f1_from_counts, parse_prediction, score


def make_reference():
    return {
        "study_type": "INTERVENTIONAL",
        "intervention_types": ["DEVICE"],
        "interventions": ["Deep Brain Stimulation"],
        "conditions": ["Parkinson Disease", "Tremor"],
    }


def test_perfect_prediction_scores_one():
    reference = make_reference()
    metrics = score([json.dumps(reference)], [reference])
    assert metrics["overall"] == 1.0
    assert metrics["valid_json_rate"] == 1.0


def test_normalization_ignores_case_and_punctuation():
    reference = make_reference()
    prediction = json.dumps({
        "study_type": "interventional",
        "intervention_types": ["device"],
        "interventions": ["deep-brain stimulation"],
        "conditions": ["parkinson disease", "TREMOR"],
    })
    assert score([prediction], [reference])["overall"] == 1.0


def test_unparseable_output_counts_as_empty_prediction():
    reference = make_reference()
    metrics = score(["sorry, I cannot help"], [reference])
    assert metrics["valid_json_rate"] == 0.0
    assert metrics["conditions_f1"] == 0.0
    assert metrics["examples"] == 1


def test_json_wrapped_in_extra_text_is_parsed():
    parsed, parsed_ok = parse_prediction('Here you go: {"study_type": "OBSERVATIONAL"} done')
    assert parsed_ok
    assert parsed["study_type"] == "OBSERVATIONAL"


def test_partial_overlap_f1():
    reference = make_reference()
    # one of two conditions right, plus one wrong condition: P = 1/2, R = 1/2, F1 = 0.5
    prediction = json.dumps({**reference, "conditions": ["Tremor", "Epilepsy"]})
    assert abs(score([prediction], [reference])["conditions_f1"] - 0.5) < 1e-9


def test_empty_reference_and_empty_prediction_is_perfect():
    assert f1_from_counts(0, 0, 0) == 1.0
    assert f1_from_counts(0, 1, 0) == 0.0
