"""Schema validation for the eval gold set. No network, no dataset, no model --
just checks backend/eval/gold/enterprise_docs.json is well-formed."""
import json
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parent.parent / "eval" / "gold" / "enterprise_docs.json"

REQUIRED_KEYS = {"id", "question", "answer", "source_doc", "source_pages", "answerable"}


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def test_gold_set_non_empty_list():
    gold = load_gold()
    assert isinstance(gold, list)
    assert len(gold) > 0


def test_gold_items_have_required_keys_and_types():
    for item in load_gold():
        assert REQUIRED_KEYS <= item.keys(), f"missing keys in {item!r}"
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["question"], str) and item["question"]
        assert isinstance(item["source_pages"], list)
        assert isinstance(item["answerable"], bool)
        if item["answerable"]:
            assert isinstance(item["answer"], str) and item["answer"]
            assert isinstance(item["source_doc"], str) and item["source_doc"]
            assert all(isinstance(p, int) for p in item["source_pages"])
        else:
            assert item["source_pages"] == []


def test_gold_set_has_both_answerable_and_unanswerable():
    gold = load_gold()
    assert any(item["answerable"] for item in gold)
    assert any(not item["answerable"] for item in gold)


def test_gold_set_ids_are_unique():
    ids = [item["id"] for item in load_gold()]
    assert len(ids) == len(set(ids))
