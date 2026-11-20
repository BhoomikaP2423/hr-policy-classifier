"""Inference pipeline for classifying HR policy sections with fine-tuned GPT-2."""
from pathlib import Path

import torch

from .config import CATEGORIES, DEPARTMENT_ROUTING, ID2LABEL
from .model import DEFAULT_DATA_PATH, DEFAULT_OUT_DIR, fine_tune_gpt2, load_gpt2_classifier
from .preprocess import load_dataset_jsonl, split_sections


class PolicyClassifier:
    """Wraps a fine-tuned GPT-2 sequence classifier for section-level inference."""

    def __init__(self, tokenizer, model, device: str | None = None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path = DEFAULT_OUT_DIR,
        auto_train_data: str | Path | None = DEFAULT_DATA_PATH,
    ) -> "PolicyClassifier":
        model_path = Path(model_dir)
        if not (model_path / "config.json").exists():
            if auto_train_data is None:
                raise FileNotFoundError(f"No trained model found at {model_path}")
            records = load_dataset_jsonl(auto_train_data)
            fine_tune_gpt2(records=records, output_dir=model_path, epochs=5)
        tokenizer, model = load_gpt2_classifier(model_path)
        return cls(tokenizer=tokenizer, model=model)

    def predict_texts(self, texts: list[str], max_length: int = 128) -> list[dict]:
        """Predict HR category, confidence, probability distribution, and routing."""
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for text, prob_row in zip(texts, probs):
            best_idx = max(range(len(prob_row)), key=lambda i: prob_row[i])
            category = ID2LABEL[best_idx]
            scores = {CATEGORIES[i]: round(prob_row[i], 4) for i in range(len(CATEGORIES))}
            results.append({
                "text": text,
                "category": category,
                "confidence": round(prob_row[best_idx], 4),
                "department": DEPARTMENT_ROUTING[category],
                "scores": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
            })
        return results

    def classify_document(self, raw_text: str) -> list[dict]:
        """Parse an uploaded HR policy document into sections and classify each one."""
        sections = split_sections(raw_text)
        if not sections:
            return []
        predictions = self.predict_texts([s["full_text"] for s in sections])
        classified = []
        for sec, pred in zip(sections, predictions):
            classified.append({
                "index": sec["index"],
                "title": sec["title"],
                "body": sec["body"],
                "category": pred["category"],
                "confidence": pred["confidence"],
                "department": pred["department"],
                "scores": pred["scores"],
            })
        return classified
