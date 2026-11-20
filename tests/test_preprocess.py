from hr_classifier.config import CATEGORIES, DEPARTMENT_ROUTING, ID2LABEL, LABEL2ID
from hr_classifier.model import DEFAULT_DATA_PATH
from hr_classifier.preprocess import load_dataset_jsonl, split_sections


def test_split_sections_extracts_titled_blocks():
    doc = (
        "Section 1: Annual Leave\n"
        "Employees receive 20 days of paid annual leave.\n\n"
        "Section 2: VPN Access\n"
        "Remote workers must connect through the corporate VPN."
    )
    sections = split_sections(doc)
    assert len(sections) == 2
    assert sections[0]["title"] == "Section 1: Annual Leave"
    assert "20 days" in sections[0]["body"]
    assert sections[1]["title"] == "Section 2: VPN Access"


def test_split_sections_handles_empty_input():
    assert split_sections("   \n\n  ") == []


def test_categories_and_routing_are_aligned():
    assert 3 <= len(CATEGORIES) <= 5
    for cat in CATEGORIES:
        assert cat in DEPARTMENT_ROUTING
        assert ID2LABEL[LABEL2ID[cat]] == cat


def test_corpus_loads_cleanly_and_covers_all_categories():
    records = load_dataset_jsonl(DEFAULT_DATA_PATH)
    assert len(records) >= 40
    present = {r["category"] for r in records}
    assert present == set(CATEGORIES)
