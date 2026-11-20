from app import results_to_csv
from hr_classifier.evaluate import evaluate_tfidf_baseline, stratified_split, to_markdown
from hr_classifier.model import DEFAULT_DATA_PATH
from hr_classifier.preprocess import load_dataset_jsonl


def test_stratified_split_preserves_all_categories():
    records = load_dataset_jsonl(DEFAULT_DATA_PATH)
    train, test = stratified_split(records, test_fraction=0.2, seed=42)
    assert len(train) + len(test) == len(records)
    assert {r["category"] for r in test} == {r["category"] for r in records}


def test_baseline_and_markdown_report():
    records = load_dataset_jsonl(DEFAULT_DATA_PATH)
    train, test = stratified_split(records, test_fraction=0.2, seed=42)
    base = evaluate_tfidf_baseline(train, test)
    assert 0.0 <= base["accuracy"] <= 1.0
    md = to_markdown({
        "total": len(records),
        "train": len(train),
        "test": len(test),
        "categories": 5,
        "baseline": base,
        "gpt2": {"accuracy": 0.90, "macro_f1": 0.90},
    })
    assert "Fine-tuned GPT-2" in md
    assert "TF-IDF" in md


def test_results_to_csv_formats_headers_and_rows():
    csv_text = results_to_csv([{
        "index": 1,
        "title": "Section 1: Leave",
        "category": "Leave & Attendance",
        "confidence": 0.9123,
        "department": "HR Operations & Timekeeping Desk",
        "body": "Employees accrue 20 days of leave.",
    }])
    assert "index,title,category,confidence,department,body" in csv_text
    assert "Leave & Attendance" in csv_text
