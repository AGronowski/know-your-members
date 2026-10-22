"""The demo-server tooling against a fake Discord API: seeding, personas, demo-safe sending, nudges and reset."""

import asyncio
import json
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import discord
import httpx
from sqlalchemy import func, select

from kym import clock, cycle, db, reviver
from kym.bot import ops
from kym.bot.app import KYMBot, is_chat
from kym.bot.seed import API, DiscordREST, reset_demo, seed_server
from kym.db import get_setting, set_setting
from kym.importer import SEED_WEBHOOKS_SETTING, seed_demo
from kym.models import CONSENT_SCOPES, Attempt, Fact, Intro, Member, Message, Revival

from .conftest import MONDAY

GUILD, BOT_ID = 900, "901"


class FakeDiscord:
    """Just enough of Discord's REST API, starting from a new server's default #general."""

    def __init__(self) -> None:
        self.next_id = 1_500_000_000_000_000_000
        self.channels = [{"id": "11", "name": "Text Channels", "type": 4}, {"id": "12", "name": "general", "type": 0, "parent_id": "11"}]
        self.webhooks: dict[str, list[dict]] = {}
        self.messages: dict[str, list[dict]] = {}
        self.calls: list[tuple] = []

    def _id(self) -> str:
        self.next_id += 1
        return str(self.next_id)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path.removeprefix("/api/v10"), request.method
        body = json.loads(request.content) if request.content else None
        self.calls.append((method, path, body, dict(request.url.params), request.headers.get("authorization")))
        if path == "/users/@me":
            return httpx.Response(200, json={"id": BOT_ID, "username": "kym"})
        if path == "/users/@me/channels":
            return httpx.Response(200, json={"id": f"77{body['recipient_id'][-6:]}"})
        if path == f"/guilds/{GUILD}":
            return httpx.Response(200, json={"id": str(GUILD), "name": "KYM demo"})
        if path == f"/guilds/{GUILD}/channels":
            if method == "POST":
                self.channels.append({"id": self._id(), **body})
                return httpx.Response(201, json=self.channels[-1])
            return httpx.Response(200, json=self.channels)
        if m := re.fullmatch(r"/channels/(\d+)/webhooks", path):
            hooks = self.webhooks.setdefault(m[1], [])
            if method == "POST":
                hooks.append({"id": self._id(), "token": "secret", "name": body["name"], "channel_id": m[1]})
                return httpx.Response(200, json=hooks[-1])
            return httpx.Response(200, json=hooks)
        if m := re.fullmatch(r"/webhooks/(\d+)/([^/]+)", path):
            channel = request.url.params.get("thread_id") or next(h["channel_id"] for hs in self.webhooks.values() for h in hs if h["id"] == m[1])
            msg = {"id": self._id(), "channel_id": channel, "content": body["content"], "author": {"id": m[1], "username": body["username"]}}
            self.messages.setdefault(channel, []).append(msg)
            return httpx.Response(200, json=msg)
        if m := re.fullmatch(r"/channels/(\d+)/messages/(\d+)/threads", path):
            return httpx.Response(201, json={"id": m[2], "name": body["name"], "type": 11, "parent_id": m[1]})
        if m := re.fullmatch(r"/channels/(\d+)/messages", path):
            if method == "POST":
                msg = {"id": self._id(), "channel_id": m[1], "author": {"id": BOT_ID}, **body}
                self.messages.setdefault(m[1], []).append(msg)
                return httpx.Response(200, json=msg)
            return httpx.Response(200, json=self.messages.get(m[1], []))
        if method == "DELETE" and re.fullmatch(r"/channels/\d+(/messages/\d+)?", path):
            return httpx.Response(204)
        return httpx.Response(404, json={"message": "Unknown", "code": 10003})

    def executes(self) -> list[tuple]:
        return [c for c in self.calls if c[1].startswith("/webhooks/")]


def _rest(fake: FakeDiscord) -> DiscordREST:
    return DiscordREST("token", client=httpx.Client(base_url=API, transport=httpx.MockTransport(fake)), pause=0)


def test_seed_mirrors_the_synthetic_chat_into_real_threads(demo):
    before = sorted((m.author_id, m.content, m.created_at) for m in demo.scalars(select(Message)))
    fake = FakeDiscord()
    r = seed_server(demo, _rest(fake), GUILD, avatars=False)

    after = demo.scalars(select(Message)).all()
    assert sorted((m.author_id, m.content, m.created_at) for m in after) == before  # same authors, words and dataset times
    assert all(m.guild_id == GUILD and m.id > 10**18 and m.jump_url.startswith(f"https://discord.com/channels/{GUILD}/") for m in after)
    assert len({m.thread_id for m in after}) == r["threads_posted"] == 8 and r["messages_posted"] == len(before)
    assert all(m.reply_to_id in {x.id for x in after} for m in after if m.reply_to_id)

    names = [c["name"] for c in fake.channels]
    assert names.count("general") == 1 and {"welcome", "help", "intros", "kym-lead"} <= set(names)
    lead = next(c for c in fake.channels if c["name"] == "kym-lead")
    assert {"id": str(GUILD), "type": 0, "deny": str(1 << 10)} in lead["permission_overwrites"]
    assert ops.channel_setting("lead") == int(lead["id"]) and ops.private_channels() == {int(lead["id"]), ops.channel_setting("intros")}

    execs = fake.executes()
    assert all(c[2]["allowed_mentions"] == {"parse": []} and c[4] is None for c in execs)  # no pings, no bot token sent
    assert {"Aisha Rahman", "Hannah Lee"} <= {c[2]["username"] for c in execs}
    join = next(c[2] for c in fake.calls if c[2] and "components" in c[2])
    assert join["components"][0]["components"][0]["custom_id"] == "kym:join"

    # A second run posts nothing, and reseeding the database doesn't bring the made-up rows back.
    again = seed_server(demo, _rest(fake), GUILD, avatars=False)
    assert again["messages_posted"] == 0 and not again["channels_created"] and not again["welcome_posted"]
    seed_demo(demo, now=MONDAY)
    assert demo.scalar(select(func.count()).select_from(Message)) == len(before)

    # The reviver works on the mirrored threads, so its replies land in real ones.
    revivals = reviver.scan(demo, MONDAY)
    assert revivals and all(rev.thread_id in {m.thread_id for m in after} for rev in revivals)


def test_history_sync_skips_seed_webhooks_and_system_lines(demo):
    set_setting(demo, SEED_WEBHOOKS_SETTING, json.dumps([42]))
    demo.commit()
    row = {"guild_id": 9, "channel_id": 10, "channel_name": "help", "thread_id": None, "thread_name": None, "author_id": 5,
           "author_name": "Priya Shah", "author_is_bot": True, "content": "hi", "created_at": MONDAY, "reply_to_id": None,
           "mention_ids": [], "jump_url": None}
    assert ops.store_messages([{**row, "id": 1, "webhook_id": 42}, {**row, "id": 2, "webhook_id": None}], {}) == 1
    with db.session_scope() as s:
        assert s.get(Message, 1) is None and s.get(Message, 2) is not None
    assert is_chat(SimpleNamespace(type=discord.MessageType.default))
    assert is_chat(SimpleNamespace(type=discord.MessageType.reply))
    assert not is_chat(SimpleNamespace(type=discord.MessageType.thread_created))
    assert not is_chat(SimpleNamespace(type=discord.MessageType.thread_starter_message))


def test_claim_routes_a_persona_to_a_real_account(demo):
    demo.commit()
    synthetic_id = demo.scalar(select(Member.discord_id).where(Member.name == "Aisha Rahman"))
    r = ops.claim_persona("aisha rahman", 777, "Adam", MONDAY)
    assert r["ok"] and r["name"] == "Aisha Rahman" and r["facts"]
    with db.session_scope() as s:
        aisha = s.scalar(select(Member).where(Member.name == "Aisha Rahman"))
        assert (aisha.discord_id, aisha.source) == (777, "claimed")
        assert s.scalar(select(func.count()).select_from(Message).where(Message.author_id == 777)) == 1  # her LiveKit question
    assert ops.claimed_persona(777) == "Aisha Rahman"
    assert ops.persona_names("fat") == ["Fatima Bello"] and ops.persona_names("", claimed=True) == ["Aisha Rahman"]
    assert "already plays" in ops.claim_persona("Fatima Bello", 777, "Adam", MONDAY)["error"]
    assert "already played" in ops.claim_persona("Aisha Rahman", 888, "Sam", MONDAY)["error"]
    ops.do_onboard(999, "Real Person", {"building": "a CRM", "offers": "sales", "wants": "investors", "speak": "", "links": ""},
                   {"store", "intro"})
    assert "own profile" in ops.claim_persona("Fatima Bello", 999, "Real", MONDAY)["error"]

    # /forget from the playing account drops the link, never the persona's profile.
    assert ops.forget_me(777) == "released Aisha Rahman"
    with db.session_scope() as s:
        aisha = s.scalar(select(Member).where(Member.name == "Aisha Rahman"))
        assert (aisha.discord_id, aisha.source) == (synthetic_id, "synthetic") and aisha.facts and aisha.allows("intro")
        assert s.scalar(select(func.count()).select_from(Message).where(Message.author_id == synthetic_id)) == 1


class Inbox:
    def __init__(self) -> None:
        self.sent: list[tuple[int, discord.Embed]] = []

    def user(self, uid: int):
        inbox = self

        class User:
            id = uid

            async def send(self, *args, **kwargs):
                inbox.sent.append((uid, kwargs.get("embed")))
                return SimpleNamespace(id=len(inbox.sent))

        return User()


class Channel:
    def __init__(self) -> None:
        self.posts: list[tuple[str, discord.ui.View | None]] = []

    async def send(self, content=None, view=None, **kwargs):
        self.posts.append((content, view))


def test_demo_safe_sending(demo):
    week = clock.week_id(MONDAY)
    r = cycle.propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    top, second = r.intros[0], r.intros[1]
    both = (top.id, top.member_a.name, top.member_b.name)
    one = (second.id, second.member_a.name, second.member_b.name)
    demo.commit()
    for i, name in enumerate((both[1], both[2], one[1])):
        assert ops.claim_persona(name, 700 + i, name, MONDAY)["ok"]

    notes = {i["id"]: i["note"] for i in ops.proposed_intros(week)}
    assert notes[both[0]] == "" and notes[one[0]].startswith(f"demo: only {one[1]} gets this")
    assert all(n.startswith("demo preview") for k, n in notes.items() if k not in (both[0], one[0]))

    inbox, lead = Inbox(), Channel()
    bot = KYMBot()
    bot.get_user = inbox.user

    async def lead_channel(_name):
        return lead

    bot._channel = lead_channel

    async def run():
        await bot.post_proposals(week)
        return {i["id"]: await bot.deliver_offer(i["id"]) for i in ops.proposed_intros(week)}

    results = asyncio.run(run())

    # The lead sees every intro once, with at most five veto buttons per message and nothing cut off.
    shown = [int(x) for content, _ in lead.posts for x in re.findall(r"\*\*#(\d+) ", content or "")]
    assert sorted(shown) == sorted(notes) and all(len(c or "") <= 2000 for c, _ in lead.posts)
    assert all(len(v.children) <= 5 for _, v in lead.posts if v)

    assert results[both[0]] == "offered" and results[one[0]] == "offered"
    assert all(v == "preview" for k, v in results.items() if k not in (both[0], one[0]))
    by_user = {uid: embed for uid, embed in inbox.sent}
    assert set(by_user) == {700, 701, 702}  # nobody synthetic was messaged
    assert "synthetic member" in by_user[702].description and "synthetic" not in by_user[700].description
    with db.session_scope() as s:
        one_sided = s.get(Intro, one[0])
        assert {d["via"] for d in one_sided.copy["delivery"].values()} == {"dm", "none"}
        failed = s.scalars(select(Attempt).where(Attempt.kind == "intro_send", Attempt.status == "failed")).all()
        assert not failed  # previews are skips, not failures

    # Re-proposing (say, after a judge joins) redoes the previews but leaves what was sent alone.
    with db.session_scope() as s:
        again = cycle.propose(s, MONDAY, force=True, use_llm=False, include_synthetic=True)
        assert again.intros and not s.scalars(select(Intro).where(Intro.status == "preview")).all()
        assert {s.get(Intro, both[0]).status, s.get(Intro, one[0]).status} == {"offered"}


def test_members_without_discord_are_not_matched(demo):
    m = Member(name="Imported Person", source="import", status="active", handles={}, meta={},
               consent={scope: True for scope in CONSENT_SCOPES})
    demo.add(m)
    demo.flush()
    demo.add(Fact(member_id=m.id, kind="wants", text="help scaling Postgres", source="import", meta={"tags": ["databases"]}))
    demo.flush()
    eligible, skipped = cycle._eligible(demo, clock.week_id(MONDAY), include_synthetic=True)
    assert m.id not in eligible and skipped[m.id] == "not on Discord yet"


def test_nudges_name_synthetic_helpers_and_ping_real_ones(demo):
    emily = demo.scalar(select(Member).where(Member.name == "Emily Zhang"))
    rev = Revival(conversation_key="thread:1", channel_id=1, thread_id=1, rule="unanswered_question", title="t", excerpt="e",
                  draft="Bringing this back.", judge={"helpers": [{"member_id": emily.id, "name": emily.name, "discord_id": emily.discord_id,
                                                                    "fact": "Postgres performance tuning"}]})
    text, mentions = reviver.render(rev, demo)
    assert "**Emily Zhang** might know" in text and "<@" not in text and mentions == []
    demo.commit()
    ops.claim_persona("Emily Zhang", 555, "E", MONDAY)
    demo.expire_all()
    text, mentions = reviver.render(rev, demo)
    assert "<@555> might know" in text and mentions == [555]
    emily = demo.scalar(select(Member).where(Member.name == "Emily Zhang"))
    emily.consent = {**emily.consent, "tag": False}
    text, mentions = reviver.render(rev, demo)
    assert "Emily" not in text and "555" not in text and mentions == []  # consent withdrawn after the scan


def test_reset_undoes_a_rehearsal(demo):
    fake = FakeDiscord()
    seed_server(demo, _rest(fake), GUILD, avatars=False)
    r = cycle.propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    intro = r.intros[0]
    pair = frozenset((intro.member_a_id, intro.member_b_id))
    cycle.mark_offered(demo, intro, MONDAY, {"a": {"via": "dm", "message_id": 5}, "b": {"via": "private_thread", "thread_id": 6}})
    cycle.mark_connected(demo, intro, 7, MONDAY)
    a = demo.get(Member, intro.member_a_id)
    a.intro_frequency, a.responsiveness = "paused", 0.1  # someone tapped "Pause my intros" during the rehearsal
    demo.commit()

    out = reset_demo(demo, _rest(fake))
    assert out["intros deleted"] == len(r.intros) and "problems" not in out
    deleted = {path for method, path, *_ in fake.calls if method == "DELETE"}
    assert {"/channels/7", "/channels/6"} <= deleted and any(p.endswith("/messages/5") for p in deleted)
    assert not get_setting(demo, f"cycle:proposed:{r.week}")
    assert (a.intro_frequency, a.responsiveness) != ("paused", 0.1)
    assert demo.scalar(select(func.count()).select_from(Message)) > 0  # the seeded chat stays

    again = cycle.propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    assert pair in {frozenset((i.member_a_id, i.member_b_id)) for i in again.intros}  # the pair can be made again


def test_invite_url_for_demo_adds_seeding_permissions(capsys):
    from kym.cli import main

    main(["invite-url", "123"])
    main(["invite-url", "123", "--demo"])
    plain, demo_url = capsys.readouterr().out.split()
    perms = [int(parse_qs(urlparse(u).query)["permissions"][0]) for u in (plain, demo_url)]
    manage_channels, manage_webhooks = 1 << 4, 1 << 29
    assert not perms[0] & (manage_channels | manage_webhooks)
    assert perms[1] & manage_channels and perms[1] & manage_webhooks
