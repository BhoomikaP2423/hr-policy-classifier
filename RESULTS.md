# Results

Dataset: 50 labeled HR policy clauses across 5 categories. Held out 10 clauses for testing; 40 used for training.

| Model | Accuracy (higher is better) | Macro F1 |
|---|---|---|
| Majority-class guess | 20.0% | 16.7% |
| TF-IDF + Logistic Regression | 80.0% | 79.3% |
| **Fine-tuned GPT-2 (this tool)** | **90.0%** | **89.8%** |
