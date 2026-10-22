"""Command line: `uv run kym <command>`. Run `uv run kym --help` for the list."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import clock, db
from .config import get_settings

console = Console()


def _status(_args) -> None:
    from sqlalchemy import func, select

    from . import llm, notion
    from .models import Event, Intro, Member, Message

    s = get_settings()
    db.init_db()
    with db.session_scope() as session:
        counts = {name: session.scalar(select(func.count()).select_from(model)) for name, model in
                  (("members", Member), ("intros", Intro), ("messages", Message), ("luma events", Event))}
        personas = [f"{m.name} ({(m.meta or {}).get('played_by', '?')})" for m in session.scalars(select(Member).where(Member.source == "claimed"))]
    t = Table(title="Know Your Members: status", show_header=False)
    for label, ok, note in (
        ("Database", True, s.database_url.split("@")[-1]),
        ("Discord bot token", bool(s.discord_token), "DISCORD_TOKEN"),
        ("Grok (writer)", llm.available("grok"), f"XAI_API_KEY, model {s.xai_model}"),
        ("Claude (extract + judge)", llm.available("claude"), f"ANTHROPIC_API_KEY, model {s.claude_model}"),
        ("Notion lead view", notion.configured(), "NOTION_TOKEN + NOTION_PARENT_PAGE_ID"),
        ("GitHub token", bool(s.github_token), "GITHUB_TOKEN (60 requests/hour without)"),
        ("Autopilot (weekly schedule)", s.autopilot, "KYM_AUTOPILOT; off = steps run only via /kym run"),
        ("Synthetic members matched", s.include_synthetic, "KYM_INCLUDE_SYNTHETIC (demo servers only)"),
    ):
        t.add_row(label, "[green]yes[/]" if ok else "[yellow]no[/]", note)
    t.add_row("Demo personas played", str(len(personas)), ", ".join(personas))
    for name, n in counts.items():
        t.add_row(name.capitalize(), str(n), "")
    console.print(t)


def _seed(args) -> None:
    from .importer import seed_demo

    db.init_db()
    with db.session_scope() as session:
        console.print(seed_demo(session, use_llm=args.llm))


def _import_members(args) -> None:
    from .importer import import_members

    db.init_db()
    with db.session_scope() as session:
        console.print(import_members(session, Path(args.file), assume_consent=args.assume_consent,
                                     use_llm=not args.no_llm, limit=args.limit))


def _import_prospects(args) -> None:
    from .importer import import_prospects

    db.init_db()
    with db.session_scope() as session:
        console.print(import_prospects(session, Path(args.file)))


def _import_discord(args) -> None:
    from .importer import import_discord_export

    db.init_db()
    with db.session_scope() as session:
        for f in args.files:
            console.print(f"{f}: {import_discord_export(session, Path(f))} messages")


def _enrich(args) -> None:
    from sqlalchemy import select

    from .enrich import enrich_member
    from .models import Member

    db.init_db()
    with db.session_scope() as session:
        members = session.scalars(select(Member).where(Member.status == "active")).all()
        if args.member:
            members = [m for m in members if m.name.lower() == args.member.lower()]
        for m in members:
            if m.allows("enrich") and m.handles:
                console.print(m.name, enrich_member(session, m))


def _crawl_luma(args) -> None:
    from .enrich import crawl_luma

    db.init_db()
    with db.session_scope() as session:
        for target in args.targets:
            console.print(target, crawl_luma(session, target, pages=args.pages, period=args.period))


def _print_intros(session, intros) -> None:
    from .writer import render_intro

    for i in intros:
        console.rule(f"#{i.id} {i.member_a.name} × {i.member_b.name}  ·  {i.confidence} ({i.score:.2f})  ·  writer: {i.copy.get('writer')}")
        console.print(f"[dim]{json.dumps({k: v for k, v in i.breakdown.items() if k != 'llm'})}[/]")
        if i.breakdown.get("llm"):
            console.print(f"[dim]LLM judge: {i.breakdown['llm']['rating']}/5, {i.breakdown['llm']['why']}[/]")
        for side, name in (("a", i.member_a.name), ("b", i.member_b.name)):
            console.print(f"[bold]To {name}:[/]")
            for line in render_intro(i.copy, i.reasons, side, i.member_b.name if side == "a" else i.member_a.name):
                console.print("  " + line)
        console.print(f"[bold]Opener:[/] {i.copy.get('opener')}")
        for d in i.copy.get("dropped", []):
            console.print(f"[red]  dropped claim ({d['problem']}): {d['text']}[/]")


def _match(args) -> None:
    from sqlalchemy import select

    from .cycle import propose
    from .models import Member

    db.init_db()
    now = db.utcnow()
    with db.session_scope() as session:
        r = propose(session, now, force=True, use_llm=None if not args.no_llm else False,
                    include_synthetic=args.include_synthetic)
        console.print(f"[bold]{r.week}[/]: {len(r.intros)} intros for {len(r.eligible)} eligible members; rerank: {r.rerank}")
        _print_intros(session, r.intros)
        names = {m.id: m.name for m in session.scalars(select(Member))}
        for mid, why in r.unmatched.items():
            console.print(f"[yellow]unmatched[/] {names[mid]}: {why}")
        if args.dry_run:
            session.rollback()
            console.print("[dim](dry run: nothing saved)[/]")


def _simulate(args) -> None:
    from .simulate import simulate

    logging.getLogger("kym").setLevel(logging.ERROR)  # the report lists failures itself
    r = simulate(weeks=args.weeks, seed=args.seed, use_llm=args.llm)
    t = Table(title=f"{args.weeks}-week simulation (same code as the bot, simulated members)")
    for col in ("week", "eligible", "intros", "unmatched", "connected", "declined", "no_reply", "useful", "meh", "silent_or_no_show", "avg_score", "joined"):
        t.add_column(col.replace("_", " "))
    for w in r["weeks"]:
        t.add_row(*(str(w[c] if w[c] is not None else "") for c in ("week", "eligible", "intros", "unmatched", "connected", "declined",
                                                                     "no_reply", "useful", "meh", "silent_or_no_show", "avg_score", "joined")))
    console.print(t)
    inv = r["invariants"]
    ok = not any(inv.values())
    console.print(("[green]✓[/]" if ok else "[red]✗[/]") + f" invariants: {inv}")
    console.print(f"coverage: {r['coverage']}; late joiner: {r['late_joiner']}")
    for label in ("graph_before", "graph_after"):
        g = r[label]
        console.print(f"{label.replace('_', ' ')}: {g['connections']} connections, {g['isolated_members']} isolated members, "
                      f"{g['share_connected']:.0%} connected, density {g['density']}")
    console.print(f"failures: {dict(r['failures'])}")
    for f in r["failure_examples"]:
        console.print(f"  [red]•[/] {f}")


def _speakers(args) -> None:
    from . import notion
    from .speakers import ALL_SOURCES, find_speakers

    db.init_db()
    with db.session_scope() as session:
        r = find_speakers(session, args.topic, limit=args.limit, use_llm=not args.no_llm,
                          sources=tuple(args.sources.split(",")) if args.sources else ALL_SOURCES)
        if args.notion and notion.configured() and r["shortlist"]:
            r["notion_url"] = notion.publish_speakers(session, r)
    if args.json:
        print(json.dumps(r, indent=1, default=str))
        return
    console.print(f"[bold]Speakers for \"{r['topic']}\"[/]  (keywords: {', '.join(r['expansion']['keywords'])}; "
                  f"{r.get('people_considered', 0)} people considered; {r.get('seconds')}s; writer: {r.get('writer')})")
    console.print(f"[dim]evidence per source: {r['counts']}[/]")
    for note in r["notes"]:
        console.print(f"[yellow]note:[/] {note}")
    for i, c in enumerate(r["shortlist"], 1):
        console.rule(f"{i}. {c['name']}{' (member)' if c['member'] else ''}  score {c['score']}")
        console.print(f"[bold]{c.get('why', '')}[/]")
        for e in c["evidence"]:
            console.print(f"  • [{e['source']}] {e['text']}" + (f"  {e['url']}" if e.get("url") else "") + (f"  ({e['date']})" if e.get("date") else ""))
        console.print(f"  London: {c['london']['basis']}")
        for line in c["identity"]:
            console.print(f"  Same person: {line}")
        for u in c["unverified"]:
            console.print(f"  [yellow]Unverified:[/] {u}")
    if r.get("notion_url"):
        console.print(f"Notion: {r['notion_url']}")


def _revive(args) -> None:
    from . import reviver

    db.init_db()
    now = db.utcnow()
    with db.session_scope() as session:
        rows = reviver.scan(session, now)
        if not rows:
            console.print("No dead threads worth looking at (or already handled in the last 30 days).")
        for r in rows:
            colour = "green" if r.status == "drafted" else "yellow"
            console.rule(f"[{colour}]{r.status}[/] {r.rule}: {r.title[:70]}")
            console.print(f"[dim]why: {r.failure_reason or r.judge.get('reason')} (judge: {r.judge.get('by')}, score {r.judge.get('score')})[/]")
            if r.draft:
                text, mentions = reviver.render(r, session)
                console.print(text)
                named = len(r.judge.get("helpers", [])) - len(mentions)
                console.print(f"[dim]writer: {r.writer}; would ping {len(mentions)} member(s) who allowed tagging"
                              + (f"; names {named} synthetic member(s) without pinging" if named else "") + "[/]")
        if args.dry_run:
            session.rollback()
            console.print("[dim](dry run: nothing saved)[/]")


def _notion(args) -> None:
    from . import notion

    db.init_db()
    with db.session_scope() as session:
        console.print(notion.setup(session) if args.cmd == "notion-setup" else notion.sync(session))


def _digest(args) -> None:
    from . import digest

    db.init_db()
    with db.session_scope() as session:
        for line in digest.render(digest.build(session, args.week or clock.week_id(db.utcnow()))):
            console.print(line)


def _bot(_args) -> None:
    from .bot.app import run

    db.init_db()
    run()


def _invite(args) -> None:
    import discord

    perms = discord.Permissions(view_channel=True, send_messages=True, send_messages_in_threads=True, embed_links=True,
                                read_message_history=True, create_private_threads=True, create_public_threads=True,
                                manage_threads=True)
    if args.demo:  # `kym discord-seed` creates channels and posts the synthetic chat through webhooks
        perms.update(manage_channels=True, manage_webhooks=True)
    print(discord.utils.oauth_url(args.client_id, permissions=perms, scopes=("bot", "applications.commands")))


def _discord_rest():
    from .bot.seed import DiscordREST

    s = get_settings()
    if not s.discord_token or not s.discord_guild_id:
        raise SystemExit("Set DISCORD_TOKEN and DISCORD_GUILD_ID in .env first (README: Demo server).")
    return DiscordREST(s.discord_token), s.discord_guild_id


def _discord_seed(args) -> None:
    from sqlalchemy import func, select

    from .bot.seed import DiscordError, seed_server
    from .importer import seed_demo
    from .models import Member

    rest, guild_id = _discord_rest()
    db.init_db()
    with db.session_scope() as session:
        if not session.scalar(select(func.count()).select_from(Member).where(Member.source.in_(("synthetic", "claimed")))):
            console.print("Loading the synthetic community first:", seed_demo(session))
            session.commit()
        try:
            console.print(seed_server(session, rest, guild_id, avatars=not args.no_avatars))
        except DiscordError as exc:
            raise SystemExit(f"Discord refused: {exc}") from None
    s = get_settings()
    for ok, warning in ((s.include_synthetic, "Set KYM_INCLUDE_SYNTHETIC=1, or the weekly round ignores the synthetic members."),
                        (not s.autopilot, "Set KYM_AUTOPILOT=0, or the bot proposes and sends intros by itself as soon as it starts.")):
        if not ok:
            console.print(f"[yellow]{warning}[/]")


def _demo_reset(args) -> None:
    from .bot.seed import DiscordError, is_demo_database, reset_demo

    rest = None if args.db_only else _discord_rest()[0]
    db.init_db()
    with db.session_scope() as session:
        if not is_demo_database(session):
            raise SystemExit("This database has no synthetic members or seeded server: refusing to wipe a real community's intros.")
        if not args.yes and input("Delete all intros, revivals and the bot log (members and chat stay)? [y/N] ").strip().lower() != "y":
            raise SystemExit("Nothing changed.")
        try:
            console.print(reset_demo(session, rest, keep_log=args.keep_log))
        except DiscordError as exc:
            raise SystemExit(f"Discord refused: {exc}") from None


def _claim(args) -> None:
    from .bot import ops

    db.init_db()
    r = (ops.claim_persona(args.persona, args.discord_user_id, args.played_by or str(args.discord_user_id), db.utcnow())
         if args.cmd == "claim" else ops.release_persona(args.persona, db.utcnow()))
    console.print(r if r["ok"] else f"[red]{r['error']}[/]")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO if "bot" in (argv or sys.argv) else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="kym", description="Know Your Members: member graph, weekly intros, speakers, thread reviver")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show configuration and counts").set_defaults(fn=_status)
    sp = sub.add_parser("seed-demo", help="load the synthetic demo community and chat history")
    sp.add_argument("--llm", action="store_true", help="extract facts with the LLM instead of rules")
    sp.set_defaults(fn=_seed)
    sp = sub.add_parser("import-members", help="import a profile file (CSV/JSON/JSONL, any columns)")
    sp.add_argument("file")
    sp.add_argument("--assume-consent", action="store_true", help="only for synthetic data or members who opted in elsewhere")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--limit", type=int)
    sp.set_defaults(fn=_import_members)
    sp = sub.add_parser("import-prospects", help="import a Luma subscriber export as prospects (no profiling)")
    sp.add_argument("file")
    sp.set_defaults(fn=_import_prospects)
    sp = sub.add_parser("import-discord-export", help="import DiscordChatExporter JSON files into the message mirror")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(fn=_import_discord)
    sp = sub.add_parser("enrich", help="pull opted-in members' public GitHub/Devpost/HN work")
    sp.add_argument("--member")
    sp.set_defaults(fn=_enrich)
    sp = sub.add_parser("crawl-luma", help="cache Luma events with hosts and featured guests")
    sp.add_argument("targets", nargs="+", help="'london', a calendar slug/URL, or event URLs / evt- ids")
    sp.add_argument("--pages", type=int, default=3)
    sp.add_argument("--period", default="future", choices=("future", "past", "all"))
    sp.set_defaults(fn=_crawl_luma)
    sp = sub.add_parser("match", help="run this week's matching now and print the intros")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--include-synthetic", action="store_true", help="match synthetic demo members too")
    sp.set_defaults(fn=_match)
    sp = sub.add_parser("simulate", help="run N weeks of the real cycle on the synthetic community")
    sp.add_argument("--weeks", type=int, default=4)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--llm", action="store_true")
    sp.set_defaults(fn=_simulate)
    sp = sub.add_parser("speakers", help="speaker shortlist for a topic, with evidence")
    sp.add_argument("topic")
    sp.add_argument("--limit", type=int, default=8)
    sp.add_argument("--sources", help="comma list of members,luma,github,devpost,hn")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--notion", action="store_true", help="publish the shortlist as a Notion page")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_speakers)
    sp = sub.add_parser("revive", help="scan mirrored chat history for dead threads and draft nudges")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=_revive)
    sub.add_parser("notion-setup", help="create the Notion databases under NOTION_PARENT_PAGE_ID").set_defaults(fn=_notion)
    sub.add_parser("notion-sync", help="push everything to Notion now").set_defaults(fn=_notion)
    sp = sub.add_parser("digest", help="print the weekly digest")
    sp.add_argument("--week")
    sp.set_defaults(fn=_digest)
    sub.add_parser("bot", help="run the Discord bot").set_defaults(fn=_bot)
    sp = sub.add_parser("invite-url", help="print the bot invite URL with the right permissions")
    sp.add_argument("client_id", help="the application id (Developer Portal > General Information)")
    sp.add_argument("--demo", action="store_true", help="add Manage Channels + Manage Webhooks, which discord-seed needs")
    sp.set_defaults(fn=_invite)
    sp = sub.add_parser("discord-seed", help="demo: build channels in DISCORD_GUILD_ID and replay the synthetic chat into them")
    sp.add_argument("--no-avatars", action="store_true", help="skip the generated avatars (they load from api.dicebear.com)")
    sp.set_defaults(fn=_discord_seed)
    sp = sub.add_parser("demo-reset", help="demo: undo a rehearsal (intros, nudges, bot log, the bot's Discord posts)")
    sp.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    sp.add_argument("--keep-log", action="store_true", help="keep the bot log")
    sp.add_argument("--db-only", action="store_true", help="leave Discord alone")
    sp.set_defaults(fn=_demo_reset)
    sp = sub.add_parser("claim", help="demo: let a real Discord account play a synthetic member")
    sp.add_argument("persona", help='e.g. "Aisha Rahman"')
    sp.add_argument("discord_user_id", type=int, help="Developer Mode on, right-click the user, Copy User ID")
    sp.add_argument("--played-by", help="a name for the log")
    sp.set_defaults(fn=_claim)
    sp = sub.add_parser("unclaim", help="demo: make a claimed member synthetic again")
    sp.add_argument("persona")
    sp.set_defaults(fn=_claim)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
