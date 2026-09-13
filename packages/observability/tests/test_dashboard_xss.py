"""Regression tests for dashboard DOM XSS sinks (Aikido High SAST).

The replay UI previously interpolated audit-log fields into HTML strings.
Those sinks are replaced with createElement / textContent / setAttribute.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pythia_observability.server import ReplayServer

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pythia_observability"
    / "templates"
    / "dashboard.html"
)

XSS_PAYLOAD = "<img src=x onerror=alert(1)>"
WRITE_CALL = re.compile(r"document\s*\.\s*write(?:ln)?\s*\(", re.IGNORECASE)
INNERHTML_ASSIGN = re.compile(r"\.innerHTML\s*=")


def _xss_entry(**overrides: object) -> dict:
    entry = {
        "timestamp": "2026-01-10T09:00:00Z",
        "market_id": XSS_PAYLOAD,
        "estimates": [
            {
                "analyst_id": XSS_PAYLOAD,
                "probability": 0.7,
                "confidence": 0.7,
                "rationale": XSS_PAYLOAD,
                "evidence": [XSS_PAYLOAD],
            }
        ],
        "decision": {
            "market_id": XSS_PAYLOAD,
            "consensus_prob": 0.7,
            "agreement_score": 0.85,
            "gate": "skip",
            "contributor_ids": [XSS_PAYLOAD],
            "method": "logit-mean",
            "market_category": "politics",
        },
        "plan": {
            "market_id": XSS_PAYLOAD,
            "side": "YES",
            "size_usd": 50.0,
            "limit_price": 0.6,
            "decision": "REJECT",
            "bankroll_before": 1000.0,
        },
        "receipt": None,
        "skipped_reason": XSS_PAYLOAD,
        "signature": f"stub:sha256:{XSS_PAYLOAD}",
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def dashboard_html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture
def xss_log(tmp_path: Path) -> Path:
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps(_xss_entry()) + "\n", encoding="utf-8")
    return path


class TestDashboardHasNoHtmlSinks:
    def test_template_has_no_document_write(self, dashboard_html: str) -> None:
        assert WRITE_CALL.search(dashboard_html) is None

    def test_template_has_no_innerhtml_assignment(self, dashboard_html: str) -> None:
        assert INNERHTML_ASSIGN.search(dashboard_html) is None

    def test_template_uses_safe_dom_builder(self, dashboard_html: str) -> None:
        assert "function el(tag, attrs, ...children)" in dashboard_html
        assert "createElement" in dashboard_html
        assert "textContent" in dashboard_html
        assert "replaceChildren" in dashboard_html


class TestDashboardRendersWithoutSinks:
    def test_served_html_has_no_write_or_innerhtml(self, xss_log: Path) -> None:
        from fastapi.testclient import TestClient

        app = ReplayServer(xss_log).app()
        with TestClient(app) as client:
            resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert WRITE_CALL.search(html) is None
        assert INNERHTML_ASSIGN.search(html) is None
        # Server-rendered shell does not embed audit fields (those come from /api).
        assert XSS_PAYLOAD not in html

    def test_api_returns_payload_as_json_string(self, xss_log: Path) -> None:
        from fastapi.testclient import TestClient

        app = ReplayServer(xss_log).app()
        with TestClient(app) as client:
            stats = client.get("/api/stats")
            trades = client.get("/api/trades")
            detail = client.get(f"/api/trades/{XSS_PAYLOAD}")
        assert stats.status_code == 200
        assert XSS_PAYLOAD in stats.json()["skipped_reasons"]
        assert trades.status_code == 200
        row = trades.json()["trades"][0]
        assert row["market_id"] == XSS_PAYLOAD
        assert row["skipped_reason"] == XSS_PAYLOAD
        assert detail.status_code == 200
        estimate = detail.json()["entries"][0]["estimates"][0]
        assert estimate["analyst_id"] == XSS_PAYLOAD
        assert estimate["rationale"] == XSS_PAYLOAD
