# HR Policy Document Classifier: fine-tuned GPT-2 + Streamlit

Upload an internal HR policy `.txt` file, split it into logical sections, and
classify each section into one of 5 HR categories using a fine-tuned GPT-2 sequence
classifier (`distilgpt2` / `gpt2`) for quick retrieval and departmental routing.

![tests](../../actions/workflows/ci.yml/badge.svg)

## Categories and departmental routing

| HR Category | Routed Department |
|---|---|
| `Leave & Attendance` | HR Operations & Timekeeping Desk |
| `Compensation & Benefits` | Payroll, Rewards & Benefits Team |
| `Code of Conduct & Ethics` | Legal, Compliance & Ethics Committee |
| `Remote Work & IT Security` | Workplace IT & InfoSec Governance |
| `Onboarding & Offboarding` | Talent Acquisition & People Experience |

## How it works

1. **Parse sections (`hr_classifier/preprocess.py`).** Uploaded `.txt` documents are
   split on blank lines and section headers (`Section 1: ...`) into individual clauses.
2. **Fine-tune GPT-2 (`hr_classifier/model.py`).** Loads `distilgpt2` (or `gpt2`) with
   `AutoModelForSequenceClassification`, assigns `eos_token` as `pad_token`, and
   fine-tunes across the 5 HR policy categories.
3. **Classify and route (`hr_classifier/inference.py`).** Runs softmax over logits
   to return predicted category, confidence score, full probability breakdown, and
   target HR team.
4. **Streamlit UI (`app.py`).** Lets HR administrators upload `.txt` policies, inspect
   per-section predictions and confidence scores, and export a CSV routing report.

## Install and run

```bash
pip install -r requirements.txt

# 1. Fine-tune GPT-2 on the HR policy corpus
python -m hr_classifier.model --epochs 6 --out artifacts/gpt2-hr

# 2. Launch the Streamlit app
streamlit run app.py
```

If `artifacts/gpt2-hr` does not exist when Streamlit starts, `PolicyClassifier.from_pretrained`
automatically fine-tunes a checkpoint from `data/hr_policies.jsonl`.

## Results

Held-out stratified evaluation (`python -m hr_classifier.evaluate`):

<!--RESULTS-->
Dataset: 50 labeled HR policy clauses across 5 categories. Held out 10 clauses for testing; 40 used for training.

| Model | Accuracy (higher is better) | Macro F1 |
|---|---|---|
| Majority-class guess | 20.0% | 16.7% |
| TF-IDF + Logistic Regression | 60.0% | 55.3% |
| **Fine-tuned GPT-2 (this tool)** | **60.0%** | **54.0%** |
<!--/RESULTS-->

## Limits and baselines

- **Small training corpus.** `data/hr_policies.jsonl` contains 50 curated policy clauses
  (10 per category). Adding real company handbook sections and re-running
  `python -m hr_classifier.evaluate` improves generalization.
- **Overlapping policy clauses.** Clauses that span two domains (e.g., a home-office
  stipend touching both *Compensation & Benefits* and *Remote Work & IT Security*)
  receive single-label predictions; inspect the full probability breakdown in the UI.
- **CPU latency.** Causal transformer sequence classification is heavier than TF-IDF
  linear models; `@st.cache_resource` keeps the model warm in memory across uploads.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit web application for `.txt` upload, classification, and CSV export |
| `hr_classifier/config.py` | HR categories, ID mappings, and departmental routing table |
| `hr_classifier/preprocess.py` | Document section splitter and JSONL dataset loader |
| `hr_classifier/model.py` | GPT-2 sequence classification fine-tuning CLI and training loop |
| `hr_classifier/inference.py` | Inference pipeline (`PolicyClassifier`) for single texts and full documents |
| `hr_classifier/evaluate.py` | Stratified held-out evaluation vs. TF-IDF baseline; updates `RESULTS.md` |
| `data/hr_policies.jsonl` | 50 labeled HR policy training clauses across 5 categories |
| `data/sample_policy.txt` | Multi-section sample HR policy `.txt` file for testing the Streamlit UI |
| `tests/` | pytest suite covering preprocessing, GPT-2 fine-tuning, inference, and evaluation |
| `.github/workflows/ci.yml` | Runs pytest on every push and pull request |

## Tests

```bash
python -m pytest -q
```
