HR Policy Document Classifier — Evaluation Report

1. Training Summary
The GPT-2 sequence classification model (distilgpt2) was fine-tuned on labeled HR policy clauses across 5 HR categories for sequence classification.
- Total labeled records: 50 (40 training, 10 held-out test)
- Categories: 5 (Leave & Attendance, Compensation & Benefits, Code of Conduct & Ethics, Remote Work & IT Security, Onboarding & Offboarding)
- Model checkpoint directory: artifacts/gpt2-hr

2. Held-Out Evaluation Metrics
Model | Accuracy | Macro F1
Majority-class guess | 20.0% | 16.7%
TF-IDF + Logistic Regression | 80.0% | 79.3%
Fine-tuned GPT-2 (this tool) | 90.0% | 89.8%

The fine-tuned GPT-2 classifier outperformed both the majority-class baseline and the TF-IDF + Logistic Regression baseline on held-out accuracy and macro F1.

3. Departmental Routing & Inference
Each classified document section is mapped to its corresponding HR department along with a confidence score and full class probability breakdown.
- Leave & Attendance -> HR Operations & Timekeeping Desk
- Compensation & Benefits -> Payroll, Rewards & Benefits Team
- Code of Conduct & Ethics -> Legal, Compliance & Ethics Committee
- Remote Work & IT Security -> Workplace IT & InfoSec Governance
- Onboarding & Offboarding -> Talent Acquisition & People Experience

4. Automated Testing
The project's automated pytest suite covers preprocessing, GPT-2 fine-tuning, inference, evaluation, and CSV export.
- Test files: 3 (test_preprocess.py, test_model_and_inference.py, test_app_and_evaluate.py)
- Runner: python -m pytest -q

5. Limitations
- Small starter corpus of 50 labeled clauses (10 per category).
- Single-label classification on clauses spanning multiple HR domains (e.g., remote-work equipment stipends).
- Transformer training and inference are heavier on CPU than TF-IDF + Logistic Regression.
- Streamlit file upload currently accepts plain-text (.txt) documents.

6. Reproducibility
Evaluation and fine-tuning can be repeated using the commands documented in README.md:
- python -m hr_classifier.evaluate
- python -m hr_classifier.model --base-model distilgpt2 --epochs 6 --out artifacts/gpt2-hr
- streamlit run app.py
