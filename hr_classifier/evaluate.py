"""Held-out evaluation comparing fine-tuned GPT-2 against a TF-IDF baseline.

Run: python -m hr_classifier.evaluate
"""
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .config import CATEGORIES
from .inference import PolicyClassifier
from .model import DEFAULT_DATA_PATH, fine_tune_gpt2, load_gpt2_classifier
from .preprocess import load_dataset_jsonl


def stratified_split(records: list[dict], test_fraction: float = 0.2, seed: int = 42):
    """Deterministic stratified train/test split across HR categories."""
    rng = random.Random(seed)
    by_cat = {cat: [] for cat in CATEGORIES}
    for r in records:
        by_cat[r["category"]].append(r)

    train, test = [], []
    for cat in CATEGORIES:
        items = list(by_cat[cat])
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - test_fraction)))
        train.extend(items[:cut])
        test.extend(items[cut:])
    return train, test


def evaluate_tfidf_baseline(train: list[dict], test: list[dict]) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform([r["text"] for r in train])
    y_train = [r["label"] for r in train]
    X_test = vec.transform([r["text"] for r in test])
    y_test = [r["label"] for r in test]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }


def run_evaluation(
    data_path: str | Path = DEFAULT_DATA_PATH,
    epochs: int = 8,
    seed: int = 42,
) -> dict:
    records = load_dataset_jsonl(data_path)
    train, test = stratified_split(records, test_fraction=0.2, seed=seed)

    baseline_metrics = evaluate_tfidf_baseline(train, test)

    tmp_dir = Path("/tmp/gpt2-hr-eval")
    fine_tune_gpt2(records=train, output_dir=tmp_dir, epochs=epochs, seed=seed)
    tokenizer, model = load_gpt2_classifier(tmp_dir)
    classifier = PolicyClassifier(tokenizer=tokenizer, model=model)

    preds_out = classifier.predict_texts([r["text"] for r in test])
    y_true = [r["category"] for r in test]
    y_pred = [p["category"] for p in preds_out]

    gpt2_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return {
        "total": len(records),
        "train": len(train),
        "test": len(test),
        "categories": len(CATEGORIES),
        "baseline": baseline_metrics,
        "gpt2": gpt2_metrics,
    }


def to_markdown(r: dict) -> str:
    lines = [
        f"Dataset: {r['total']} labeled HR policy clauses across {r['categories']} categories. "
        f"Held out {r['test']} clauses for testing; {r['train']} used for training.",
        "",
        "| Model | Accuracy (higher is better) | Macro F1 |",
        "|---|---|---|",
        f"| Majority-class guess | {1 / r['categories']:.1%} | {1 / (r['categories'] + 1):.1%} |",
        f"| TF-IDF + Logistic Regression | {r['baseline']['accuracy']:.1%} | {r['baseline']['macro_f1']:.1%} |",
        f"| **Fine-tuned GPT-2 (this tool)** | **{r['gpt2']['accuracy']:.1%}** | **{r['gpt2']['macro_f1']:.1%}** |",
    ]
    return "\n".join(lines)


def write_report(r: dict, root: Path | None = None) -> str:
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    block = to_markdown(r)
    (root / "RESULTS.md").write_text("# Results\n\n" + block + "\n", encoding="utf-8")
    readme = root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        updated = re.sub(
            r"<!--RESULTS-->.*?<!--/RESULTS-->",
            lambda _: "<!--RESULTS-->\n" + block + "\n<!--/RESULTS-->",
            text,
            flags=re.S,
        )
        readme.write_text(updated, encoding="utf-8")
    return block


def main():
    print(write_report(run_evaluation()))


if __name__ == "__main__":
    main()
