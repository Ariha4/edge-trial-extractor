import json

from trialx.prompts import FIELD_NAMES, build_messages, format_target


def test_target_is_deterministic_and_sorted():
    labels = {
        "conditions": ["Tremor", "Parkinson Disease"],
        "study_type": "INTERVENTIONAL",
        "interventions": ["Deep Brain Stimulation"],
        "intervention_types": ["DEVICE"],
    }
    target = format_target(labels)
    assert list(json.loads(target).keys()) == FIELD_NAMES
    assert json.loads(target)["conditions"] == ["Parkinson Disease", "Tremor"]
    assert target == format_target(dict(reversed(list(labels.items()))))


def test_messages_have_system_and_user_roles():
    messages = build_messages("Some trial text")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Some trial text" in messages[1]["content"]
