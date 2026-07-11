# backend/tests/test_eval_report_endpoint.py
import json

from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


def test_eval_report_returns_json_when_present(tmp_path, monkeypatch):
    report = {"dataset": "DocVQA (lmms-lab)", "n_docs": 1, "n_answerable": 1, "n_ood": 0, "modes": {}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(main, "REPORT_PATH", report_path)

    r = client.get("/eval/report")
    assert r.status_code == 200
    assert r.json() == report


def test_eval_report_404_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORT_PATH", tmp_path / "does_not_exist.json")

    r = client.get("/eval/report")
    assert r.status_code == 404
    assert r.json() == {"detail": "no report"}
