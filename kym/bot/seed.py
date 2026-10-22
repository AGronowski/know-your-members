"""Turn an empty Discord server into the demo community, and reset it after a rehearsal.

Discord has no fake users, so the synthetic chat history is posted through a webhook under each member's name and
avatar (Discord marks those messages as coming from an app). Messages can't be backdated either: they show the time
they were posted, while the database keeps the dataset's timestamps, which is what the reviver reads. Each mirrored
message replaces its synthetic row under its real Discord id, so the bot's history sync recognises it and revival
replies land in the real thread. Running the seeder again only posts what isn't mirrored yet.

Plain REST over httpx: seeding runs from the CLI, with or without the bot connected.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from urllib.parse import quote

import httpx
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .. import log
from ..config import FIXTURES_DIR
from ..db import get_setting, set_setting
from ..importer import DEMO_THREAD_BASE, MIRRORED_THREAD_KEY, SEED_WEBHOOKS_SETTING
from ..models import Attempt, Intro, Member, Message, Revival, Setting

API = "https://discord.com/api/v10"
SEED_WEBHOOK = "KYM demo seed"
WELCOME_KEY = "discord-seed:welcome"
TEXT, CATEGORY = 0, 4
VIEW_CHANNEL, SEND_MESSAGES, EMBED_LINKS, READ_MESSAGE_HISTORY = 1 << 10, 1 << 11, 1 << 14, 1 << 16
CHAT_CHANNELS = ("general", "help", "show-and-tell", "events")
TOPICS = {
    "welcome": "Start here: what this server is, and the button to join the weekly intros",
    "general": "Anything goes",
    "help": "Stuck? Ask here",
    "show-and-tell": "Share what you shipped",
    "events": "Meetups, talks and hackathons",
    "intros": "Private intro threads: only the two people introduced (and moderators) can see each one",
    "kym-lead": "Leads only: proposed intros with veto buttons, nudge drafts and the weekly digest",
}
WELCOME = (
    "**Welcome to the Know Your Members demo.**\n\n"
    "The {members} members talking in {channels} are synthetic. Discord has no fake accounts, so a webhook replays "
    "their messages under their names (that's why they're marked as an app). Discord can't backdate messages either, "
    "so they show today's time; the bot uses the dataset's original timestamps.\n\n"
    "Press **Join intros** to answer five quick questions. Each week you'll get one introduction with the reasons "
    "behind it, and both of you have to say yes. `/me` shows what the bot holds about you; `/forget` deletes it."
)


class DiscordError(RuntimeError):
    pass


def avatar_url(name: str) -> str:
    return f"https://api.dicebear.com/9.x/notionists/png?seed={quote(name)}&size=128"


def _explain(method: str, shown: str, r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        body = {}
    message = body.get("message") or r.text[:200]
    hint = ""
    if r.status_code == 401:
        hint = " DISCORD_TOKEN is wrong or was reset."
    elif r.status_code == 403:
        hint = (" The bot is missing a permission: seeding needs Manage Channels and Manage Webhooks. Re-invite it with "
                "`uv run kym invite-url <application id> --demo` (re-inviting just updates its role).")
    elif r.status_code == 404 and shown.startswith("/guilds/"):
        hint = " The bot isn't in that server, or DISCORD_GUILD_ID is wrong."
    return f"{method} {shown}: HTTP {r.status_code} {message} (code {body.get('code')}).{hint}"


class DiscordREST:
    """The handful of Discord REST calls seeding and resetting need. Waits out rate limits; errors say what to fix."""

    def __init__(self, token: str, client: httpx.Client | None = None, pause: float = 0.5) -> None:
        self.token = token
        self.client = client or httpx.Client(base_url=API, timeout=30)
        self.pause = pause  # between webhook posts, to stay under the per-channel webhook limit

    def request(self, method: str, path: str, *, auth: bool = True, **kwargs):
        shown = re.sub(r"(/webhooks/\d+/)[^/?]+", r"\1<token>", path)  # never log a webhook token
        headers = {"Authorization": f"Bot {self.token}"} if auth else {}
        for _ in range(6):
            r = self.client.request(method, path, headers=headers, **kwargs)
            if r.status_code == 429:
                try:
                    wait = float(r.json().get("retry_after", 1.0))
                except ValueError:
                    wait = float(r.headers.get("retry-after", 1.0))
                time.sleep(min(wait, 30.0))
                continue
            if r.status_code >= 400:
                raise DiscordError(_explain(method, shown, r))
            return r.json() if r.content else None
        raise DiscordError(f"{method} {shown}: still rate limited after 6 tries")


# ------------------------------------------------------------------ seeding

def _seeded_threads(session: Session) -> dict[int, list[Message]]:
    """Synthetic threads still under their made-up ids, oldest message first."""
    rows = session.scalars(select(Message).where(Message.thread_id.between(DEMO_THREAD_BASE + 1, DEMO_THREAD_BASE + 999_999))
                           .order_by(Message.created_at, Message.id)).all()
    threads: dict[int, list[Message]] = {}
    for m in rows:
        threads.setdefault(m.thread_id, []).append(m)
    return dict(sorted(threads.items(), key=lambda kv: kv[1][0].created_at))


def _post(rest: DiscordREST, hook: dict, m: Message, avatars: bool, thread_id: str | None = None) -> dict:
    params = {"wait": "true", **({"thread_id": thread_id} if thread_id else {})}
    body = {"content": m.content[:2000], "username": m.author_name[:80], "allowed_mentions": {"parse": []}}
    if avatars:
        body["avatar_url"] = avatar_url(m.author_name)
    msg = rest.request("POST", f"/webhooks/{hook['id']}/{hook['token']}", auth=False, params=params, json=body)
    time.sleep(rest.pause)
    return msg


def _mirror(session: Session, guild_id: int, channel_id: int, old_thread_id: int, thread_id: int,
            sent: list[tuple[Message, dict, int]]) -> None:
    """Swap the synthetic rows for rows under the real Discord ids (same authors, text and dataset timestamps)."""
    ids = {old.id: int(new["id"]) for old, new, _ in sent}
    for old, new, container in sent:
        session.add(Message(id=int(new["id"]), guild_id=guild_id, channel_id=channel_id, channel_name=old.channel_name,
                            thread_id=thread_id, thread_name=old.thread_name, author_id=old.author_id,
                            author_name=old.author_name, author_is_bot=False, content=old.content, created_at=old.created_at,
                            reply_to_id=ids.get(old.reply_to_id, old.reply_to_id), mention_ids=old.mention_ids or [],
                            jump_url=f"https://discord.com/channels/{guild_id}/{container}/{new['id']}"))
        session.delete(old)
    # Revivals drafted against the made-up thread would post nowhere; the live /kym revive finds it again.
    session.execute(delete(Revival).where(Revival.thread_id == old_thread_id))
    set_setting(session, MIRRORED_THREAD_KEY.format(old_thread_id), str(thread_id))


def seed_server(session: Session, rest: DiscordREST, guild_id: int, *, avatars: bool = True) -> dict:
    """Create the channels, post the synthetic chat into real threads, point the bot at #intros and #kym-lead, and
    post the welcome note with the Join button. Idempotent: existing channels, webhooks and threads are reused."""
    me = rest.request("GET", "/users/@me")
    guild = rest.request("GET", f"/guilds/{guild_id}")
    existing = rest.request("GET", f"/guilds/{guild_id}/channels")
    created: list[str] = []

    def ensure(name: str, kind: int, parent: dict | None = None, overwrites: list[dict] | None = None) -> dict:
        for c in existing:
            if c["name"] == name and c["type"] == kind:
                return c
        body: dict = {"name": name, "type": kind}
        if parent:
            body["parent_id"] = parent["id"]
        if name in TOPICS:
            body["topic"] = TOPICS[name]
        if overwrites:
            body["permission_overwrites"] = overwrites
        channel = rest.request("POST", f"/guilds/{guild_id}/channels", json=body)
        existing.append(channel)
        created.append(f"#{name}" if kind == TEXT else name)
        return channel

    threads = _seeded_threads(session)
    community = ensure("Community", CATEGORY)
    welcome = ensure("welcome", TEXT, community)
    names = list(dict.fromkeys([*CHAT_CHANNELS, *(msgs[0].channel_name for msgs in threads.values())]))
    chat = {name: ensure(name, TEXT, community) for name in names}
    kym = ensure("Know Your Members", CATEGORY)
    intros = ensure("intros", TEXT, kym)
    lead = ensure("kym-lead", TEXT, kym, overwrites=[
        {"id": str(guild_id), "type": 0, "deny": str(VIEW_CHANNEL)},  # @everyone can't see it; owners and admins can
        {"id": str(me["id"]), "type": 1, "allow": str(VIEW_CHANNEL | SEND_MESSAGES | EMBED_LINKS | READ_MESSAGE_HISTORY)},
    ])
    set_setting(session, "discord:intros_channel", str(intros["id"]))
    set_setting(session, "discord:lead_channel", str(lead["id"]))
    session.commit()

    hooks: dict[str, dict] = {}

    def hook_for(channel: dict) -> dict:
        if channel["id"] not in hooks:
            found = [w for w in rest.request("GET", f"/channels/{channel['id']}/webhooks")
                     if w.get("name") == SEED_WEBHOOK and w.get("token")]
            hook = found[0] if found else rest.request("POST", f"/channels/{channel['id']}/webhooks", json={"name": SEED_WEBHOOK})
            known = set(json.loads(get_setting(session, SEED_WEBHOOKS_SETTING) or "[]")) | {int(hook["id"])}
            set_setting(session, SEED_WEBHOOKS_SETTING, json.dumps(sorted(known)))
            session.commit()  # before posting, so a running bot already ignores this webhook's messages
            hooks[channel["id"]] = hook
        return hooks[channel["id"]]

    posted = 0
    for old_thread_id, msgs in threads.items():
        channel = chat[msgs[0].channel_name]
        hook = hook_for(channel)
        first = _post(rest, hook, msgs[0], avatars)
        thread = rest.request("POST", f"/channels/{channel['id']}/messages/{first['id']}/threads",
                              json={"name": (msgs[0].thread_name or msgs[0].content)[:100], "auto_archive_duration": 10080})
        sent = [(msgs[0], first, int(channel["id"]))]
        sent += [(m, _post(rest, hook, m, avatars, thread_id=thread["id"]), int(thread["id"])) for m in msgs[1:]]
        _mirror(session, guild_id, int(channel["id"]), old_thread_id, int(thread["id"]), sent)
        session.commit()  # thread by thread, so a failure halfway can simply be re-run
        posted += len(sent)

    welcomed = False
    if not get_setting(session, WELCOME_KEY):
        members = session.scalar(select(func.count()).select_from(Member).where(Member.source.in_(("synthetic", "claimed"))))
        msg = rest.request("POST", f"/channels/{welcome['id']}/messages", json={
            "content": WELCOME.format(members=members, channels=", ".join(f"<#{c['id']}>" for c in chat.values())),
            "components": [{"type": 1, "components": [{"type": 2, "style": 1, "label": "Join intros", "custom_id": "kym:join"}]}],
            "allowed_mentions": {"parse": []}})
        set_setting(session, WELCOME_KEY, str(msg["id"]))
        welcomed = True
    result = {"server": guild.get("name"), "channels_created": created, "threads_posted": len(threads),
              "messages_posted": posted, "welcome_posted": welcomed,
              "intros_channel": f"#{intros['name']}", "lead_channel": f"#{lead['name']}"}
    log.record(session, "discord_seed", "ok", f"{len(threads)} threads, {posted} messages, created {', '.join(created) or 'nothing'}",
               target=str(guild.get("name")))
    return result


# ------------------------------------------------------------------ reset after a rehearsal

def is_demo_database(session: Session) -> bool:
    return bool(get_setting(session, WELCOME_KEY)) or bool(
        session.scalar(select(func.count()).select_from(Member).where(Member.source.in_(("synthetic", "claimed")))))


def _try(counts: Counter, key: str, failures: list[str], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
        counts[key] += 1
    except DiscordError as exc:
        if "HTTP 404" not in str(exc):  # already gone is fine
            failures.append(str(exc)[:200])


def reset_demo(session: Session, rest: DiscordREST | None, *, keep_log: bool = False) -> dict:
    """Undo a rehearsal so the live run starts clean: delete intros (or Aisha and Fatima could never be paired again),
    revivals, cycle markers and, unless keep_log, the bot log. With Discord access it also deletes what the bot posted:
    offer DMs, intro threads, nudges and its messages in the lead channel. Members, consent, personas and the seeded
    chat stay; anyone who joined during the rehearsal can /forget themselves."""
    counts: Counter = Counter()
    failures: list[str] = []
    if rest is not None:
        bot_id = rest.request("GET", "/users/@me")["id"]
        dm_channels: dict[int, str] = {}
        for intro in session.scalars(select(Intro)):
            if intro.thread_id:
                _try(counts, "intro threads", failures, rest.request, "DELETE", f"/channels/{intro.thread_id}")
            for side, member in (("a", intro.member_a), ("b", intro.member_b)):
                d = ((intro.copy or {}).get("delivery") or {}).get(side) or {}
                if d.get("via") == "private_thread":
                    _try(counts, "private notes", failures, rest.request, "DELETE", f"/channels/{d['thread_id']}")
                elif d.get("via") == "dm" and member.discord_id:
                    if member.discord_id not in dm_channels:
                        dm_channels[member.discord_id] = rest.request("POST", "/users/@me/channels",
                                                                      json={"recipient_id": str(member.discord_id)})["id"]
                    _try(counts, "offer DMs", failures, rest.request, "DELETE",
                         f"/channels/{dm_channels[member.discord_id]}/messages/{d['message_id']}")
        for rev in session.scalars(select(Revival).where(Revival.nudge_message_id.is_not(None))):
            _try(counts, "nudges", failures, rest.request, "DELETE", f"/channels/{rev.thread_id or rev.channel_id}/messages/{rev.nudge_message_id}")
        lead = get_setting(session, "discord:lead_channel")
        if lead:
            for m in rest.request("GET", f"/channels/{lead}/messages", params={"limit": 100}):
                if m["author"]["id"] == bot_id:
                    _try(counts, "lead channel posts", failures, rest.request, "DELETE", f"/channels/{lead}/messages/{m['id']}")
    counts["intros deleted"] = session.execute(delete(Intro)).rowcount
    counts["revivals deleted"] = session.execute(delete(Revival)).rowcount
    for prefix in ("cycle:", "revive:"):
        counts["markers cleared"] += session.execute(delete(Setting).where(Setting.key.startswith(prefix))).rowcount
    if not keep_log:
        counts["log entries deleted"] = session.execute(delete(Attempt)).rowcount
    # A rehearsal can pause a persona or lower their responsiveness; put the dataset's values back.
    spec = {row["name"]: row for row in json.loads((FIXTURES_DIR / "demo_members.json").read_text())["members"]}
    for m in session.scalars(select(Member).where(Member.source.in_(("synthetic", "claimed")))):
        if m.name in spec:
            m.intro_frequency = spec[m.name].get("frequency", "weekly")
            m.responsiveness = spec[m.name].get("responsiveness", 0.7)
    log.record(session, "demo_reset", "failed" if failures else "ok",
               "; ".join(f"{k}: {v}" for k, v in counts.items() if v) + (f"; problems: {' | '.join(failures[:3])}" if failures else ""))
    return {**{k: v for k, v in counts.items() if v}, **({"problems": failures} if failures else {})}
