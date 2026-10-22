"""Enrichment: opted-in members' public work, and Luma events (hosts + featured guests) for the speaker finder."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from .. import log
from ..facts import FactDraft, replace_facts
from ..models import Event, Member
from ..taxonomy import tags_for
from .http import FetchError, fetcher
from .sources import (
    _parse_dt,
    devpost_profile,
    github_repo_evidence,
    github_top_repos,
    github_user,
)

PERSON_FIELDS = ("api_id", "name", "first_name", "last_name", "username", "bio_short", "twitter_handle",
                 "linkedin_handle", "website", "instagram_handle")


def enrich_member(session: Session, member: Member) -> dict:
    """Pull an opted-in member's public work into sourced facts. Each source fails independently and is logged."""
    if not member.allows("enrich"):
        return {"status": "skipped", "reason": "no enrich consent"}
    handles = member.handles or {}
    drafts: list[FactDraft] = []
    notes: list[str] = []
    if handles.get("github"):
        try:
            user = github_user(handles["github"])
            if user is None:
                notes.append(f"GitHub user {handles['github']} not found")
            else:
                for repo in github_top_repos(user["login"], 5):
                    ev = github_repo_evidence(repo, user)
                    drafts.append(FactDraft(kind="shipped", text=ev.text, url=ev.url, source="github",
                                            source_ref=repo["full_name"], observed_at=ev.date, tags=sorted(tags_for(ev.topic_text)),
                                            meta={"stars": repo.get("stargazers_count", 0), "language": repo.get("language")}))
                if user.get("location"):
                    drafts.append(FactDraft(kind="location", text=user["location"], url=user["html_url"], source="github"))
        except FetchError as exc:
            notes.append(f"GitHub unavailable: {exc}")
    if handles.get("devpost"):
        try:
            prof = devpost_profile(handles["devpost"])
            if prof is None:
                notes.append(f"Devpost user {handles['devpost']} not found")
            else:
                for p in prof["projects"][:5]:
                    text = f"Built {p['name']} at a hackathon" + (f": {p['tagline']}" if p["tagline"] else "") + (" (winner)" if p["winner"] else "")
                    drafts.append(FactDraft(kind="shipped", text=text[:300], url=p["url"], source="devpost",
                                            tags=sorted(tags_for(p["name"] + " " + p["tagline"])), meta={"winner": p["winner"]}))
                if prof.get("location"):
                    drafts.append(FactDraft(kind="location", text=prof["location"], url=prof["url"], source="devpost"))
        except FetchError as exc:
            notes.append(f"Devpost unavailable: {exc}")
    if handles.get("hn"):
        try:
            url = (f"https://hn.algolia.com/api/v1/search?tags=show_hn,author_{handles['hn']}&hitsPerPage=20")
            hits = sorted((fetcher().get(url) or {}).get("hits", []), key=lambda h: h.get("points") or 0, reverse=True)
            for h in hits[:3]:
                drafts.append(FactDraft(kind="shipped", text=f"{h['title']} ({h.get('points', 0)} points on Hacker News)",
                                        url=f"https://news.ycombinator.com/item?id={h['objectID']}", source="hn",
                                        observed_at=_parse_dt(h.get("created_at")), tags=sorted(tags_for(h["title"])),
                                        meta={"points": h.get("points", 0)}))
        except FetchError as exc:
            notes.append(f"Hacker News unavailable: {exc}")
    replace_facts(session, member, drafts, {"github", "devpost", "hn"})
    status = "failed" if notes and not drafts else "ok"
    log.record(session, "enrich", status, f"{len(drafts)} facts" + (f"; {'; '.join(notes)}" if notes else ""),
               target=member.name, notes=notes)
    return {"status": status, "facts": len(drafts), "notes": notes}


def _person(p: dict) -> dict:
    return {k: p.get(k) for k in PERSON_FIELDS if p.get(k)}


def upsert_luma_entries(session: Session, entries: list[dict], source: str) -> int:
    n = 0
    for entry in entries:
        ev = entry.get("event") or {}
        api_id = ev.get("api_id") or entry.get("api_id")
        if not api_id:
            continue
        geo = ev.get("geo_address_info") or {}
        row = session.get(Event, api_id) or Event(id=api_id, source=source)
        row.name = ev.get("name") or row.name or ""
        row.start_at = _parse_dt(ev.get("start_at") or entry.get("start_at"))
        row.url = f"https://lu.ma/{ev['url']}" if ev.get("url") else row.url
        row.city = geo.get("city") or (entry.get("featured_city") or {}).get("name") or row.city
        cal = entry.get("calendar") or {}
        row.calendar_name, row.calendar_api_id = cal.get("name") or row.calendar_name, cal.get("api_id") or row.calendar_api_id
        row.guest_count = entry.get("guest_count")
        row.hosts = [_person(h) for h in entry.get("hosts") or []]
        row.featured = [_person(h) for h in entry.get("featured_guests") or []]
        row.fetched_at = datetime.now().astimezone()
        session.merge(row)
        n += 1
    return n


def crawl_luma(session: Session, target: str, pages: int = 3, period: str = "future") -> dict:
    """Store events (with hosts and featured guests) from a Luma discover place ('london'), a calendar slug/URL,
    or a single event ('evt-...' or event URL). period: future | past | all (past needs a login on most calendars)."""
    from .sources import luma_calendar, luma_discover, luma_event, luma_resolve

    slug = target.rstrip("/").split("/")[-1].split("?")[0]
    notes: list[str] = []
    entries: list[dict] = []
    source = f"luma:{slug}"
    try:
        if slug.startswith("evt-"):
            detail = luma_event(slug)
            entries = [detail] if detail else []
            source = "luma:event"
        else:
            resolved = luma_resolve(slug) or {}
            kind = resolved.get("kind")
            data = resolved.get("data") or {}
            if kind == "discover-place":
                entries = luma_discover(data["place"]["api_id"], pages)
                source = f"discover:{slug}"
            elif kind == "calendar":
                source = f"calendar:{slug}"
                for per in (("future", "past") if period == "all" else (period,)):
                    try:
                        entries += luma_calendar(data["calendar"]["api_id"], per, pages)
                    except FetchError as exc:
                        notes.append(f"{per} events unavailable: {exc}")
            elif kind == "event":
                detail = luma_event(data["event"]["api_id"])
                entries = [detail] if detail else []
                source = "luma:event"
            else:
                notes.append(f"unrecognised Luma target {target!r} (kind={kind})")
    except FetchError as exc:
        notes.append(str(exc))
    n = upsert_luma_entries(session, entries, source)
    log.record(session, "crawl_luma", "ok" if n else "failed", f"{n} events" + (f"; {'; '.join(notes)}" if notes else ""),
               target=target, notes=notes)
    return {"events": n, "notes": notes, "source": source}
