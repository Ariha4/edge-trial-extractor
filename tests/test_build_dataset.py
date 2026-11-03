from trialx.build_dataset import assign_split, make_example
from trialx.text_utils import appears_in


def make_study(conditions, interventions, study_type="INTERVENTIONAL", nct_id="NCT00000001"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": "DBS for Parkinson Disease"},
            "descriptionModule": {
                "briefSummary": "This study tests deep brain stimulation in people with Parkinson disease.",
            },
            "designModule": {"studyType": study_type},
            "conditionsModule": {"conditions": conditions},
            "armsInterventionsModule": {"interventions": interventions},
        }
    }


def test_labels_not_in_text_are_dropped():
    study = make_study(
        conditions=["Parkinson Disease", "Depression"],
        interventions=[{"type": "DEVICE", "name": "Deep Brain Stimulation"}, {"type": "DRUG", "name": "Levodopa"}],
    )
    example = make_example(study)
    assert example["labels"]["conditions"] == ["Parkinson Disease"]
    assert example["labels"]["interventions"] == ["Deep Brain Stimulation"]
    # Types come from a fixed vocabulary and are kept even when the name is not grounded.
    assert example["labels"]["intervention_types"] == ["DEVICE", "DRUG"]


def test_study_with_nothing_grounded_is_skipped():
    study = make_study(conditions=["Epilepsy"], interventions=[{"type": "DRUG", "name": "Valproate"}])
    assert make_example(study) is None


def test_unknown_study_type_is_skipped():
    study = make_study(
        conditions=["Parkinson Disease"], interventions=[], study_type="EXPANDED_ACCESS"
    )
    assert make_example(study) is None


def test_appears_in_matches_whole_words_only():
    assert appears_in("EEG", "We record EEG during sleep")
    assert not appears_in("EEG", "We record EEGs during sleep")
    assert not appears_in("", "anything")


def test_split_assignment_is_stable():
    assert assign_split("NCT01234567") == assign_split("NCT01234567")
    assert assign_split("NCT01234567") in {"train", "val", "test"}
