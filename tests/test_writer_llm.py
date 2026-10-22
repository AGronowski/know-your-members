import json
from types import SimpleNamespace

import pytest

from kym import llm, writer
from kym.config import get_settings

REASONS = [
    {"id": "r1", "kind": "b_helps_a", "strength": 0.8, "shared_tags": ["evals"],
     "want": {"id": 1, "kind": "wants", "text": "Help evaluating the bot's answers", "url": None, "source": "answer", "tags": ["evals"]},
     "offer": {"id": 2, "kind": "offers", "text": "LLM evals and red teaming", "url": None, "source": "answer", "tags": ["evals"]},
     "proof": {"id": 3, "kind": "shipped", "text": "Built ragcheck: eval harness for RAG (★120, Python)",
               "url": "https://github.com/example/ragcheck", "source": "github", "tags": ["evals"]}},
]


def test_verify_claims_drops_what_it_cannot_back():
    allowed = {"r1": ["Help evaluating the bot's answers", "Built ragcheck (★120)"]}
    kept, dropped = writer.verify_claims([
        {"text": "Priya built ragcheck, which has 120 stars.", "reason_ids": ["r1"]},
        {"text": "Priya has 5,000 followers.", "reason_ids": ["r1"]},
        {"text": "See https://example.com", "reason_ids": ["r1"]},
        {"text": "You two would get on.", "reason_ids": []},
        {"text": "Something else.", "reason_ids": ["r9"]},
    ], allowed)
    assert [k["text"] for k in kept] == ["Priya built ragcheck, which has 120 stars."]
    assert {d["problem"].split(":")[0] for d in dropped} == {"numbers not in sources", "contains a link", "cites nothing", "cites unknown id ['r9']"}


def test_write_intro_keeps_verified_claims_and_falls_back_per_side(monkeypatch):
    def fake(session, role, system, user, schema, name):
        assert role == "write" and "r1" in user
        return {"to_a": [{"text": "Priya built an eval harness with 120 stars.", "reason_ids": ["r1"]},
                         {"text": "She has raised 2 million.", "reason_ids": ["r1"]}],
                "to_b": [{"text": "Trust me.", "reason_ids": []}], "opener": "How do you score groundedness?"}, "grok"

    monkeypatch.setattr(llm, "complete_json", fake)
    copy = writer.write_intro(None, "Tom", "Priya", REASONS)
    assert [c["text"] for c in copy["to_a"]] == ["Priya built an eval harness with 120 stars."]
    assert copy["to_b"] == writer.template_copy("Tom", "Priya", REASONS)["to_b"]
    assert copy["writer"] == "grok+template" and len(copy["dropped"]) == 2


def test_rendered_sources_link_the_other_persons_evidence():
    copy = writer.template_copy("Tom", "Priya", REASONS)
    lines = writer.render_intro(copy, REASONS, "a", "Priya")
    assert "https://github.com/example/ragcheck" in lines[0]


def test_claude_request_shape_and_refusal(monkeypatch):
    monkeypatch.setenv("KYM_CLAUDE", "1")
    get_settings.cache_clear()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        text = SimpleNamespace(type="text", text=json.dumps({"ok": True}))
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, content=[text])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_anthropic", lambda: fake)
    data, provider = llm.call("extract", "sys", "user", {"type": "object"})
    assert (data, provider) == ({"ok": True}, "claude")
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    assert "temperature" not in kw

    def refuse(**kwargs):
        return SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), content=[])

    fake.beta.messages.create = refuse
    with pytest.raises(llm.LLMFailed, match="refused"):
        llm.call("judge", "sys", "user", {"type": "object"})


def test_grok_request_shape(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    get_settings.cache_clear()
    seen = {}

    def post(url, json, timeout, headers):
        seen.update(url=url, body=json, headers=headers)
        return SimpleNamespace(status_code=200, headers={}, text="",
                               json=lambda: {"choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}]})

    monkeypatch.setattr(llm.httpx, "post", post)
    data, provider = llm.call("write", "sys", "user", {"type": "object"}, name="intro")
    assert (data, provider) == ({"a": 1}, "grok")
    assert seen["url"] == "https://api.x.ai/v1/chat/completions"
    assert seen["body"]["model"] == "grok-4.7"
    assert seen["body"]["response_format"]["json_schema"] == {"name": "intro", "schema": {"type": "object"}, "strict": True}
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_role_fallback_order(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "k")
    monkeypatch.setenv("KYM_CLAUDE", "1")
    get_settings.cache_clear()
    assert llm.providers_for("write") == ["grok", "claude"]
    assert llm.providers_for("judge") == ["claude", "grok"]
