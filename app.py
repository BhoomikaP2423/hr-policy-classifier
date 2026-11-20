"""Streamlit web app for HR policy document section classification."""
import csv
import io
from pathlib import Path

import streamlit as st

from hr_classifier.config import CATEGORIES, DEPARTMENT_ROUTING
from hr_classifier.inference import PolicyClassifier
from hr_classifier.model import DEFAULT_DATA_PATH, DEFAULT_OUT_DIR

SAMPLE_FILE = Path(__file__).resolve().parent / "data" / "sample_policy.txt"

st.set_page_config(page_title="HR Policy Document Classifier", layout="wide")


@st.cache_resource
def load_classifier(model_dir: str) -> PolicyClassifier:
    return PolicyClassifier.from_pretrained(model_dir=model_dir, auto_train_data=DEFAULT_DATA_PATH)


def results_to_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["index", "title", "category", "confidence", "department", "body"],
    )
    writer.writeheader()
    for row in results:
        writer.writerow({k: row[k] for k in writer.fieldnames})
    return buf.getvalue()


def main():
    st.title("HR Policy Document Classifier")
    st.caption(
        "Upload an internal HR policy `.txt` file to split it into sections, classify each "
        "section with a fine-tuned GPT-2 model, and route it to the responsible HR team."
    )

    with st.sidebar:
        st.header("Model Settings")
        model_dir = st.text_input("Model Checkpoint Directory", value=str(DEFAULT_OUT_DIR))
        st.subheader("HR Categories & Routing")
        for cat in CATEGORIES:
            st.markdown(f"- **{cat}** -> {DEPARTMENT_ROUTING[cat]}")

    uploaded = st.file_uploader("Upload HR Policy Document (.txt)", type=["txt"])
    use_sample = st.checkbox("Use built-in sample HR policy document", value=uploaded is None)

    raw_text = ""
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="replace")
    elif use_sample and SAMPLE_FILE.exists():
        raw_text = SAMPLE_FILE.read_text(encoding="utf-8")

    if not raw_text.strip():
        st.info("Upload a `.txt` policy file or enable the sample policy checkbox above.")
        return

    with st.spinner("Loading fine-tuned GPT-2 model and classifying sections..."):
        classifier = load_classifier(model_dir)
        results = classifier.classify_document(raw_text)

    if not results:
        st.warning("No readable sections found in the uploaded file.")
        return

    st.subheader(f"Classified Sections ({len(results)})")
    summary_rows = [
        {
            "Section": f"#{r['index']} — {r['title']}",
            "Predicted Category": r["category"],
            "Confidence": f"{r['confidence']:.1%}",
            "Departmental Routing": r["department"],
        }
        for r in results
    ]
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Routing Report (CSV)",
        data=results_to_csv(results),
        file_name="hr_policy_routing.csv",
        mime="text/csv",
    )

    st.divider()
    for r in results:
        with st.expander(
            f"Section {r['index']}: {r['title']}  |  {r['category']} ({r['confidence']:.1%})",
            expanded=True,
        ):
            st.markdown(f"**Routed To:** `{r['department']}`")
            st.write(r["body"])
            st.caption("Category probabilities: " + ", ".join(
                f"{cat}: {score:.1%}" for cat, score in r["scores"].items()
            ))


if __name__ == "__main__":
    main()
