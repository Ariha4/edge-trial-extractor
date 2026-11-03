"""Score model outputs against the answer key.

Metrics:
    valid_json_rate      share of outputs that parse as a JSON object
    study_type_accuracy  exact match on study_type
    <list field>_f1      micro F1 over normalized strings, for intervention_types,
                         interventions, and conditions
    overall              mean of study_type_accuracy and the three F1 scores

Outputs that fail to parse count as empty predictions (so they hurt recall)
instead of being dropped (which would hide failures).
"""
import json

from trialx.text_utils import normalize

LIST_FIELDS = ["intervention_types", "interventions", "conditions"]


def parse_prediction(raw_output):
    """Pull the first JSON object out of a model reply. Returns (dict, parsed_ok)."""
    start_index = raw_output.find("{")
    end_index = raw_output.rfind("}")
    if start_index == -1 or end_index <= start_index:
        return {}, False
    try:
        parsed = json.loads(raw_output[start_index:end_index + 1])
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(parsed, dict):
        return {}, False
    return parsed, True


def to_normalized_set(value):
    normalized_items = set()
    if not isinstance(value, list):
        return normalized_items
    for item in value:
        if isinstance(item, str):
            normalized_item = normalize(item)
            if normalized_item:
                normalized_items.add(normalized_item)
    return normalized_items


def f1_from_counts(true_positives, false_positives, false_negatives):
    if true_positives == 0:
        # Nothing expected and nothing predicted is a perfect score, not zero.
        if false_positives == 0 and false_negatives == 0:
            return 1.0
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * precision * recall / (precision + recall)


def score(predictions, references):
    """predictions: list of raw model output strings. references: list of label dicts."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")

    valid_json_count = 0
    study_type_correct = 0
    counts = {}
    for field_name in LIST_FIELDS:
        counts[field_name] = {"tp": 0, "fp": 0, "fn": 0}

    for raw_output, reference in zip(predictions, references):
        predicted, parsed_ok = parse_prediction(raw_output)
        if parsed_ok:
            valid_json_count += 1

        predicted_study_type = str(predicted.get("study_type", "")).strip().upper()
        if predicted_study_type == reference["study_type"]:
            study_type_correct += 1

        for field_name in LIST_FIELDS:
            predicted_set = to_normalized_set(predicted.get(field_name, []))
            reference_set = to_normalized_set(reference[field_name])
            for item in predicted_set:
                if item in reference_set:
                    counts[field_name]["tp"] += 1
                else:
                    counts[field_name]["fp"] += 1
            for item in reference_set:
                if item not in predicted_set:
                    counts[field_name]["fn"] += 1

    example_count = len(references)
    metrics = {
        "examples": example_count,
        "valid_json_rate": valid_json_count / example_count if example_count else 0.0,
        "study_type_accuracy": study_type_correct / example_count if example_count else 0.0,
    }
    overall_parts = [metrics["study_type_accuracy"]]
    for field_name in LIST_FIELDS:
        field_counts = counts[field_name]
        field_f1 = f1_from_counts(field_counts["tp"], field_counts["fp"], field_counts["fn"])
        metrics[f"{field_name}_f1"] = field_f1
        overall_parts.append(field_f1)
    metrics["overall"] = sum(overall_parts) / len(overall_parts)
    return metrics
