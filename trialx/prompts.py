"""Prompt and target formatting shared by training and evaluation.

Training and evaluation must build prompts the exact same way, otherwise the
fine-tuned model is scored on inputs it never saw during training.
"""
import json

SYSTEM_PROMPT = (
    "You extract facts from clinical trial descriptions. "
    "Reply with only a JSON object with these keys: "
    '"study_type" (INTERVENTIONAL or OBSERVATIONAL), '
    '"intervention_types" (list, e.g. DEVICE, DRUG, BEHAVIORAL, PROCEDURE), '
    '"interventions" (list of intervention names), '
    '"conditions" (list of conditions studied). '
    "Use only information stated in the text."
)

FIELD_NAMES = ["study_type", "intervention_types", "interventions", "conditions"]


def build_user_message(trial_text):
    return "Trial description:\n" + trial_text


def build_messages(trial_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(trial_text)},
    ]


def format_target(labels):
    """Serialize labels deterministically so the model learns one stable output format."""
    ordered_labels = {}
    for field_name in FIELD_NAMES:
        value = labels[field_name]
        if isinstance(value, list):
            ordered_labels[field_name] = sorted(value)
        else:
            ordered_labels[field_name] = value
    return json.dumps(ordered_labels, ensure_ascii=False)
