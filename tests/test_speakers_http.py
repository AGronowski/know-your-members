from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

from kym.enrich import http
from kym.enrich.sources import Evidence, keys_from_links, london_signal, luma_person_keys, web_key
from kym.speakers import Relevance, _identity_basis, _merge, _score, is_person


def ev(source, kind, text, keys, location=None, **metrics):
    return Evidence(source=source, kind=kind, text=text, url=f"https://example.com/{source}", date=datetime.now(timezone.utc),
                    topic_text=text, person_name="Jane Doe", person_keys=set(keys), metrics=metrics, location=location,
                    location_source=f"{source} location" if location else None)


def test_people_merge_on_shared_handles_never_on_names():
    a = ev("github", "built", "Built voicebox: realtime TTS", {"github:jane", "twitter:janedoe"}, location="London", stars=300)
    b = ev("luma", "hosted", "Hosted Voice AI London", {"luma:usr-1", "twitter:janedoe"}, location="London", guests=120)
    c = ev("hn", "posted", "Show HN: voicebox", {"hn:someone-else"})
    c.person_name = "Jane Doe"  # same name, no shared key: must stay separate
    people = _merge([a, b, c], [0.9, 0.9, 0.9])
    assert sorted(len(p.evidence) for p in people) == [1, 2]
    merged = next(p for p in people if len(p.evidence) == 2)
    assert any("twitter 'janedoe' on GitHub and Luma" in line for line in _identity_basis(merged))
    s = _score(merged)
    assert s["build"] > 0 and s["speak"] > 0 and s["london"]["score"] == 1.0


def test_london_from_events_is_marked_as_inferred():
    person = _merge([ev("luma", "hosted", "Hosted Voice AI night", {"luma:usr-2"}, location="London", guests=50)], [0.9])[0]
    s = _score(person)
    assert s["london"]["score"] == 0.7 and "inferred" in s["london"]["basis"]
    assert "London base inferred from events, not confirmed" in s["unverified"]


def test_key_and_location_helpers():
    assert london_signal("London, UK") == 1.0 and london_signal("Manchester, UK") == 0.5 and london_signal("Berlin") == 0
    assert web_key("https://www.janedoe.dev/blog") == "janedoe.dev" and web_key("https://github.com/jane") is None
    assert keys_from_links("see github.com/Jane and https://x.com/jd") == {"github:jane", "twitter:jd"}
    assert luma_person_keys({"api_id": "usr-1", "twitter_handle": "@JD", "linkedin_handle": "/in/jane-doe"}) == {
        "luma:usr-1", "twitter:jd", "linkedin:jane-doe"}


def test_organisations_are_not_speakers():
    assert is_person({"name": "Jane Doe", "first_name": "Jane", "last_name": "Doe"})
    assert not is_person({"name": "Blott Labs"})
    assert not is_person({"name": "Imperial FinTech Society", "first_name": "Imperial", "last_name": "FinTech Society"})


def test_relevance_needs_more_than_vague_similarity():
    rel = Relevance("voice AI agents", {"keywords": ["voice agents", "speech-to-text"], "tags": ["voice"]},
                    np.ones(384, dtype=np.float32) / np.sqrt(384), {"voice", "agents"})
    assert rel.lexical("Hosting a night on voice agents in production") == 1.0
    assert rel.lexical("Halloween party: Gemini edition") == 0.0
    assert rel.lexical("Realtime speech pipelines for agents") >= 0.6


def _fetcher(handler):
    return http.Fetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_fetcher_backs_off_on_luma_rate_limit_then_caches(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(request.url)
        if len(calls) == 1:
            return httpx.Response(429, json={"message": "You’ve exceeded our rate limit."})
        return httpx.Response(200, json={"entries": [1, 2]})

    f = _fetcher(handler)
    url = "https://api.lu.ma/discover/get-paginated-events?x=1"
    assert f.get(url) == {"entries": [1, 2]} and len(calls) == 2
    assert f.get(url) == {"entries": [1, 2]} and len(calls) == 2  # served from cache


def test_fetcher_uses_stale_cache_when_source_is_down(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    up = {"ok": True}
    f = _fetcher(lambda r: httpx.Response(200, json={"v": 1}) if up["ok"] else httpx.Response(503))
    url = "https://hn.algolia.com/api/v1/search?q=x"
    assert f.get(url) == {"v": 1}
    up["ok"] = False
    assert f.get(url, max_age_hours=0) == {"v": 1} and f.stats["stale_hits"] == 1
    with pytest.raises(http.FetchError):
        f.get("https://hn.algolia.com/api/v1/search?q=never-cached")


def test_fetcher_404_is_none():
    f = _fetcher(lambda r: httpx.Response(404))
    assert f.get("https://api.github.com/users/nobody-here") is None
