import math
import tempfile
from pathlib import Path

from hr_classifier.config import CATEGORIES
from hr_classifier.inference import PolicyClassifier
from hr_classifier.model import DEFAULT_DATA_PATH, fine_tune_gpt2, load_gpt2_classifier
from hr_classifier.preprocess import load_dataset_jsonl


def _tiny_subset():
    records = load_dataset_jsonl(DEFAULT_DATA_PATH)
    picked = []
    for cat in CATEGORIES:
        cat_rows = [r for r in records if r["category"] == cat][:2]
        picked.extend(cat_rows)
    return picked


def test_fine_tune_and_inference_round_trip():
    records = _tiny_subset()
    out_dir = Path(tempfile.mkdtemp()) / "gpt2-test"
    meta = fine_tune_gpt2(records=records, output_dir=out_dir, epochs=1, batch_size=4)
    assert meta["num_records"] == len(records)
    assert (out_dir / "config.json").exists()

    clf = PolicyClassifier.from_pretrained(model_dir=out_dir, auto_train_data=None)
    sample_doc = (
        "Section 1: Sick Leave\n"
        "Sick leave of up to 12 days per year is provided for personal illness.\n\n"
        "Section 2: Laptop Encryption\n"
        "Company laptops must have full-disk encryption and VPN enabled."
    )
    results = clf.classify_document(sample_doc)
    assert len(results) == 2
    for res in results:
        assert res["category"] in CATEGORIES
        assert 0.0 <= res["confidence"] <= 1.0
        assert math.isclose(sum(res["scores"].values()), 1.0, abs_tol=1e-2)
        assert res["department"]


def test_empty_predict_returns_empty_list():
    records = _tiny_subset()
    out_dir = Path(tempfile.mkdtemp()) / "gpt2-empty"
    fine_tune_gpt2(records=records, output_dir=out_dir, epochs=1, batch_size=4)
    tok, mod = load_gpt2_classifier(out_dir)
    clf = PolicyClassifier(tokenizer=tok, model=mod)
    assert clf.predict_texts([]) == []
    assert clf.classify_document("") == []
