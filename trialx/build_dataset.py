"""Turn raw ClinicalTrials.gov records into train / val / test extraction examples.

Usage:
    python -m trialx.build_dataset --raw data/raw_studies.jsonl --out-dir data

Input text  = title + brief summary + start of the detailed description.
Labels      = the API's own structured fields (study type, intervention types,
              intervention names, conditions). These are the free answer key.

Grounding filter: a condition or intervention name is kept only if it appears
in the input text. Otherwise the model would be scored on facts it cannot see,
which measures memorization, not extraction. study_type and intervention_types
come from fixed vocabularies and are kept as is because they can be inferred.

Splits are assigned by hashing the NCT ID, so a trial always lands in the same
split no matter how many times the dataset is rebuilt.
"""
import argparse
import hashlib
import json
import os

from trialx.text_utils import appears_in

MAX_DESCRIPTION_CHARS = 1200
VALID_STUDY_TYPES = {"INTERVENTIONAL", "OBSERVATIONAL"}


def build_text(protocol):
    identification = protocol.get("identificationModule", {})
    description = protocol.get("descriptionModule", {})
    title = identification.get("briefTitle", "").strip()
    summary = description.get("briefSummary", "").strip()
    detailed = description.get("detailedDescription", "").strip()[:MAX_DESCRIPTION_CHARS]

    parts = []
    for part in [title, summary, detailed]:
        if part:
            parts.append(part)
    return "\n\n".join(parts)


def extract_labels(protocol, text):
    design = protocol.get("designModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    arms = protocol.get("armsInterventionsModule", {})

    study_type = design.get("studyType", "")

    intervention_types = set()
    grounded_interventions = set()
    for intervention in arms.get("interventions", []):
        intervention_type = intervention.get("type")
        if intervention_type:
            intervention_types.add(intervention_type)
        intervention_name = intervention.get("name", "").strip()
        if intervention_name and appears_in(intervention_name, text):
            grounded_interventions.add(intervention_name)

    grounded_conditions = set()
    for condition in conditions_module.get("conditions", []):
        condition = condition.strip()
        if condition and appears_in(condition, text):
            grounded_conditions.add(condition)

    return {
        "study_type": study_type,
        "intervention_types": sorted(intervention_types),
        "interventions": sorted(grounded_interventions),
        "conditions": sorted(grounded_conditions),
    }


def make_example(study):
    """Return an example dict, or None if the study is unusable."""
    protocol = study.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId")
    text = build_text(protocol)
    if not nct_id or not text:
        return None

    labels = extract_labels(protocol, text)
    if labels["study_type"] not in VALID_STUDY_TYPES:
        return None
    # Skip trials where nothing extractable survived grounding.
    if not labels["conditions"] and not labels["interventions"]:
        return None
    return {"nct_id": nct_id, "text": text, "labels": labels}


def assign_split(nct_id, val_percent=10, test_percent=10):
    bucket = int(hashlib.sha256(nct_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"


def build(raw_path, out_dir):
    examples_by_split = {"train": [], "val": [], "test": []}
    skipped_count = 0
    with open(raw_path, encoding="utf-8") as raw_file:
        for line in raw_file:
            example = make_example(json.loads(line))
            if example is None:
                skipped_count += 1
                continue
            examples_by_split[assign_split(example["nct_id"])].append(example)

    os.makedirs(out_dir, exist_ok=True)
    for split_name, examples in examples_by_split.items():
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_file:
            for example in examples:
                out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(examples)} examples -> {out_path}")
    print(f"skipped: {skipped_count}")
    return examples_by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw_studies.jsonl")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    build(args.raw, args.out_dir)


if __name__ == "__main__":
    main()
