"""Public data sources, normalised into Evidence about people.

GitHub (REST), Devpost (search JSON + profile HTML), Hacker News (Algolia) and Luma (hosts and featured guests of
events). Person identity keys (github:x, twitter:y, web:domain, ...) let the speaker finder merge the same person
across sources, and every piece of evidence keeps its URL and date.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlparse

from .http import fetcher

LONDON = re.compile(r"\b(london|ldn)\b", re.I)
UK = re.compile(r"\b(uk|u\.k\.|united kingdom|england|britain|scotland|wales)\b", re.I)
GENERIC_DOMAINS = {"github.com", "linkedin.com", "twitter.com", "x.com", "medium.com", "youtube.com", "instagram.com",
                   "facebook.com", "lu.ma", "luma.com", "devpost.com", "news.ycombinator.com", "linktr.ee", "substack.com",
                   "notion.site", "google.com", "calendly.com", "t.co", "bit.ly"}


@dataclass
class Evidence:
    source: str  # github | devpost | hn | luma | member
    kind: str  # built | hosted | featured | posted | said
    text: str
    url: str | None
    date: datetime | None
    topic_text: str
    person_name: str | None
    person_keys: set[str] = field(default_factory=set)
    metrics: dict = field(default_factory=dict)
    location: str | None = None
    location_source: str | None = None
    profile_urls: dict[str, str] = field(default_factory=dict)


def london_signal(text: str | None) -> float:
    if not text:
        return 0.0
    if LONDON.search(text):
        return 1.0
    if UK.search(text):
        return 0.5
    return 0.0


def norm_handle(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lstrip("@").rstrip("/")
    value = re.sub(r"^https?://(www\.)?(twitter\.com|x\.com|linkedin\.com/in|github\.com)/", "", value, flags=re.I)
    value = value.split("?")[0].strip("/")
    return value.lower() or None


def web_key(url: str | None) -> str | None:
    if not url:
        return None
    if not re.match(r"https?://", url):
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host or "." not in host or host in GENERIC_DOMAINS:
        return None
    return host


def keys_from_links(text: str) -> set[str]:
    keys = set()
    for kind, pattern in (("github", r"github\.com/([A-Za-z0-9-]{1,39})"), ("twitter", r"(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})"),
                          ("linkedin", r"linkedin\.com/in/([A-Za-z0-9_%-]+)"), ("devpost", r"devpost\.com/(?!software\b)([A-Za-z0-9_-]+)")):
        for m in re.finditer(pattern, text or "", re.I):
            keys.add(f"{kind}:{m.group(1).lower()}")
    for m in re.finditer(r"https?://[^\s\"'<>)]+", text or ""):
        key = web_key(m.group(0))
        if key:
            keys.add(f"web:{key}")
    return keys


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------- GitHub

GH = "https://api.github.com"


def github_user(login: str) -> dict | None:
    return fetcher().get(f"{GH}/users/{quote(login)}")


def github_repos(login: str) -> list[dict]:
    return fetcher().get(f"{GH}/users/{quote(login)}/repos?per_page=100&sort=pushed") or []


def github_user_keys(user: dict) -> set[str]:
    keys = {f"github:{user['login'].lower()}"}
    if user.get("twitter_username"):
        keys.add(f"twitter:{user['twitter_username'].lower()}")
    w = web_key(user.get("blog"))
    if w:
        keys.add(f"web:{w}")
    keys |= keys_from_links(user.get("blog") or "")
    return keys


def github_repo_evidence(repo: dict, user: dict) -> Evidence:
    stars = repo.get("stargazers_count", 0)
    desc = (repo.get("description") or "").strip()
    topics = repo.get("topics") or []
    text = f"Built {repo['name']}" + (f": {desc}" if desc else "") + f" (★{stars}" + (f", {repo['language']}" if repo.get("language") else "") + ")"
    return Evidence(source="github", kind="built", text=text, url=repo["html_url"], date=_parse_dt(repo.get("pushed_at")),
                    topic_text=" ".join([repo["name"].replace("-", " "), desc, " ".join(topics)]),
                    person_name=user.get("name") or user["login"], person_keys=github_user_keys(user),
                    metrics={"stars": stars, "forks": repo.get("forks_count", 0)},
                    location=user.get("location"), location_source="GitHub profile location" if user.get("location") else None,
                    profile_urls={"GitHub": user["html_url"]})


def github_top_repos(login: str, n: int = 5) -> list[dict]:
    repos = [r for r in github_repos(login) if not r.get("fork") and not r.get("archived")]
    return sorted(repos, key=lambda r: (r.get("stargazers_count", 0), r.get("pushed_at") or ""), reverse=True)[:n]


def github_search(keywords: list[str], max_users: int = 20, since_days: int = 540) -> tuple[list[Evidence], list[str]]:
    """London builders on GitHub for these keywords: topical repos whose owner is in London, plus London users whose
    bio mentions the keyword."""
    notes: list[str] = []
    evidence: list[Evidence] = []
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).date().isoformat()
    owners: dict[str, list[dict]] = {}
    for kw in keywords[:3]:
        q = f'{kw} in:name,description,topics pushed:>{since} stars:>2'
        data = fetcher().get(f"{GH}/search/repositories?q={quote(q)}&sort=stars&order=desc&per_page=30")
        for repo in (data or {}).get("items", []):
            if repo["owner"]["type"] == "User":
                owners.setdefault(repo["owner"]["login"], []).append(repo)
        data = fetcher().get(f"{GH}/search/users?q={quote(f'{kw} in:bio location:London')}&per_page=10")
        for u in (data or {}).get("items", []):
            owners.setdefault(u["login"], [])
    looked_up = 0
    for login, repos in owners.items():
        if looked_up >= max_users:
            notes.append(f"GitHub: stopped after {max_users} profile lookups ({len(owners)} candidates)")
            break
        user = github_user(login)
        looked_up += 1
        if not user or london_signal(user.get("location")) == 0:
            continue
        if not repos:
            repos = github_top_repos(login, 3)
        for repo in repos[:3]:
            evidence.append(github_repo_evidence(repo, user))
    return evidence, notes


# ---------------------------------------------------------------- Devpost

def devpost_search(query: str) -> list[dict]:
    data = fetcher().get(f"https://devpost.com/software/search?query={quote(query)}", headers={"Accept": "application/json"})
    return (data or {}).get("software", [])


def devpost_profile(username: str) -> dict | None:
    page = fetcher().get(f"https://devpost.com/{quote(username)}", as_json=False)
    if not page:
        return None
    title = re.search(r"<title>\s*(.*?)\s*\(([^)]+)\) software portfolio", page, re.S)
    name = html.unescape(title.group(1)).removesuffix("'s").strip() if title else username
    loc = re.search(r'ss-location"></span>\s*([^<]+?)\s*</li>', page)
    links_block = re.search(r'id="portfolio-user-links">(.*?)</ul>', page, re.S)
    links = re.findall(r'href="(https?://[^"]+)"', links_block.group(1)) if links_block else []
    projects = []
    for m in re.finditer(r'href="(https://devpost\.com/software/[^"]+)">(.*?)</footer>', page, re.S):
        chunk = m.group(2)
        title_m = re.search(r"<h5>\s*(.*?)\s*</h5>", chunk, re.S)
        tagline = re.search(r'class="small tagline">\s*(.*?)\s*</p>', chunk, re.S)
        projects.append({"url": m.group(1), "name": html.unescape(title_m.group(1)) if title_m else m.group(1),
                         "tagline": html.unescape(tagline.group(1)) if tagline else "",
                         "winner": bool(re.search(r"winner", chunk, re.I))})
    return {"username": username, "name": name, "location": html.unescape(loc.group(1)).strip() if loc else None,
            "links": links, "projects": projects, "url": f"https://devpost.com/{username}"}


def devpost_keys(profile: dict) -> set[str]:
    keys = {f"devpost:{profile['username'].lower()}"}
    for link in profile.get("links", []):
        keys |= keys_from_links(link)
    return keys


def devpost_find(keywords: list[str], max_profiles: int = 12) -> tuple[list[Evidence], list[str]]:
    notes: list[str] = []
    projects: dict[str, dict] = {}
    for kw in keywords[:3]:
        for p in devpost_search(kw):
            projects.setdefault(p["url"], p)
    ranked = sorted(projects.values(), key=lambda p: (bool(p.get("winner")), p.get("like_count", 0)), reverse=True)
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for p in ranked:
        for username in p.get("members", []):
            if username in seen:
                continue
            if len(seen) >= max_profiles:
                notes.append(f"Devpost: checked {max_profiles} profiles for location (of {len(projects)} projects)")
                return evidence, notes
            seen.add(username)
            prof = devpost_profile(username)
            if not prof or london_signal(prof.get("location")) == 0:
                continue
            text = f"Built {p['name']} at a hackathon: {p.get('tagline', '').strip()}" + (" (winner)" if p.get("winner") else "")
            evidence.append(Evidence(
                source="devpost", kind="built", text=text, url=p["url"], date=None,
                topic_text=" ".join([p["name"], p.get("tagline", ""), " ".join(p.get("tags", []))]),
                person_name=prof["name"], person_keys=devpost_keys(prof),
                metrics={"winner": bool(p.get("winner")), "likes": p.get("like_count", 0)},
                location=prof.get("location"), location_source="Devpost profile location",
                profile_urls={"Devpost": prof["url"]}))
    return evidence, notes


# ---------------------------------------------------------------- Hacker News

def hn_user(username: str) -> dict | None:
    return fetcher().get(f"https://hn.algolia.com/api/v1/users/{quote(username)}")


def hn_find(keywords: list[str], since_days: int = 730, min_points: int = 15) -> tuple[list[Evidence], list[str]]:
    since = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp())
    hits: dict[str, dict] = {}
    for kw in keywords[:3]:
        url = (f"https://hn.algolia.com/api/v1/search?query={quote(kw)}&tags=show_hn"
               f"&numericFilters=created_at_i>{since},points>{min_points}&hitsPerPage=20")
        for h in (fetcher().get(url) or {}).get("hits", []):
            hits.setdefault(h["objectID"], h)
    evidence: list[Evidence] = []
    users: dict[str, dict | None] = {}
    for h in sorted(hits.values(), key=lambda h: h.get("points") or 0, reverse=True)[:25]:
        author = h.get("author")
        if author not in users:
            users[author] = hn_user(author)
        about = html.unescape(re.sub(r"<[^>]+>", " ", (users[author] or {}).get("about") or ""))
        keys = {f"hn:{author.lower()}"} | keys_from_links(about)
        evidence.append(Evidence(
            source="hn", kind="posted", text=f"{h['title']} ({h.get('points', 0)} points on Hacker News)",
            url=f"https://news.ycombinator.com/item?id={h['objectID']}", date=_parse_dt(h.get("created_at")),
            topic_text=f"{h['title']} {h.get('url') or ''}", person_name=author, person_keys=keys,
            metrics={"points": h.get("points", 0), "comments": h.get("num_comments", 0)},
            location=about if london_signal(about) else None,
            location_source="Hacker News profile" if london_signal(about) else None,
            profile_urls={"Hacker News": f"https://news.ycombinator.com/user?id={author}"}))
    return evidence, []


# ---------------------------------------------------------------- Luma

LUMA = "https://api.lu.ma"


def luma_person_keys(p: dict) -> set[str]:
    keys = set()
    if p.get("api_id"):
        keys.add(f"luma:{p['api_id']}")
    if p.get("twitter_handle"):
        keys.add(f"twitter:{norm_handle(p['twitter_handle'])}")
    if p.get("linkedin_handle"):
        slug = norm_handle(re.sub(r"^/?in/", "", p["linkedin_handle"]))
        if slug:
            keys.add(f"linkedin:{slug}")
    w = web_key(p.get("website"))
    if w:
        keys.add(f"web:{w}")
    return keys


def luma_profile_urls(p: dict) -> dict[str, str]:
    urls = {}
    if p.get("username"):
        urls["Luma"] = f"https://lu.ma/user/{p['username']}"
    if p.get("linkedin_handle"):
        urls["LinkedIn"] = "https://www.linkedin.com/in/" + re.sub(r"^/?in/", "", p["linkedin_handle"]).strip("/")
    if p.get("twitter_handle"):
        urls["X"] = f"https://x.com/{norm_handle(p['twitter_handle'])}"
    if p.get("website"):
        urls["Website"] = p["website"] if p["website"].startswith("http") else "https://" + p["website"]
    return urls


def luma_resolve(slug: str) -> dict | None:
    return fetcher().get(f"{LUMA}/url?url={quote(slug)}")


def luma_discover(place_api_id: str, pages: int = 3) -> list[dict]:
    entries, cursor = [], None
    for _ in range(pages):
        url = f"{LUMA}/discover/get-paginated-events?discover_place_api_id={place_api_id}&pagination_limit=50"
        if cursor:
            url += f"&pagination_cursor={quote(cursor)}"
        data = fetcher().get(url) or {}
        entries += data.get("entries", [])
        cursor = data.get("next_cursor")
        if not data.get("has_more") or not cursor:
            break
    return entries


def luma_calendar(calendar_api_id: str, period: str = "past", pages: int = 3) -> list[dict]:
    entries, cursor = [], None
    for _ in range(pages):
        url = f"{LUMA}/calendar/get-items?calendar_api_id={calendar_api_id}&period={period}&pagination_limit=50"
        if cursor:
            url += f"&pagination_cursor={quote(cursor)}"
        data = fetcher().get(url) or {}
        entries += data.get("entries", [])
        cursor = data.get("next_cursor")
        if not data.get("has_more") or not cursor:
            break
    return entries


def luma_event(event_api_id: str) -> dict | None:
    return fetcher().get(f"{LUMA}/event/get?event_api_id={event_api_id}")


def luma_description(detail: dict) -> str:
    """Plain text from Luma's ProseMirror-style description_mirror."""
    out: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "text":
                out.append(node.get("text", ""))
            for child in node.get("content", []) or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(detail.get("description_mirror") or {})
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def recency(date: datetime | None) -> float:
    if not date:
        return 0.6
    age_days = (datetime.now(timezone.utc) - date).days
    return 1.0 if age_days < 180 else 0.8 if age_days < 365 else 0.6 if age_days < 730 else 0.4


def log_scale(value: float, top: float) -> float:
    return min(1.0, math.log10(1 + max(0.0, value)) / math.log10(1 + top))
