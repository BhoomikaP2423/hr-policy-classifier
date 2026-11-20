"""Fine-tuning logic for GPT-2 sequence classification on HR policy sections."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CATEGORIES, ID2LABEL, LABEL2ID
from .preprocess import load_dataset_jsonl

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hr_policies.jsonl"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "gpt2-hr"


class PolicyDataset(Dataset):
    """Tokenized dataset of HR policy text snippets and integer category IDs."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            [r["text"] for r in records],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_gpt2_classifier(model_name_or_path: str | Path = DEFAULT_MODEL_NAME):
    """Initialize tokenizer (with pad_token set) and GPT-2 sequence classifier."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_name_or_path),
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def fine_tune_gpt2(
    records: list[dict],
    base_model: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path = DEFAULT_OUT_DIR,
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_length: int = 128,
    seed: int = 42,
) -> dict:
    """Fine-tune GPT-2 on labeled HR policy records and save the checkpoint."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_gpt2_classifier(base_model)
    model.to(device)

    dataset = PolicyDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
        avg_loss = total_loss / max(len(dataset), 1)
        history.append({"epoch": epoch, "loss": round(avg_loss, 4)})
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    meta = {
        "base_model": base_model,
        "num_records": len(records),
        "epochs": epochs,
        "categories": CATEGORIES,
        "history": history,
    }
    (out_path / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on HR policy sections.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    records = load_dataset_jsonl(args.data)
    fine_tune_gpt2(
        records=records,
        base_model=args.base_model,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
