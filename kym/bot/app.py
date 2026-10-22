"""Discord bot: questionnaire, weekly intros (double opt-in, private threads), follow-ups, speaker finder, thread
reviver and the lead's commands.

All state lives in the database. The tick loop re-derives what is due from that state, so a restarted bot catches
up without double-sending, and buttons are dynamic items, so they keep working across restarts.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import tasks

from .. import clock
from ..config import get_settings
from ..db import get_setting, session_scope, set_setting, utcnow
from ..members import QUESTIONS
from ..models import CONSENT_SCOPES
from . import ops

log = logging.getLogger("kym.bot")

BLURPLE = 0x5865F2
CONSENT_TEXT = (
    "**Know Your Members**: five quick questions, then one well-chosen introduction a week to someone here, with the "
    "reasons and links.\n\nChoose what the bot may do (the first two are needed to take part; you can change your "
    "mind any time with `/join`, `/pause` or `/forget`):"
)


def is_lead(interaction: discord.Interaction) -> bool:
    perms = interaction.permissions
    return bool(perms and (perms.manage_guild or perms.administrator))


# ------------------------------------------------------------------ join flow

class JoinButton(discord.ui.DynamicItem[discord.ui.Button], template=r"kym:join"):
    def __init__(self) -> None:
        super().__init__(discord.ui.Button(label="Join intros", style=discord.ButtonStyle.primary, custom_id="kym:join"))

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls()

    async def callback(self, interaction: discord.Interaction) -> None:
        await start_join(interaction)


async def start_join(interaction: discord.Interaction) -> None:
    # Every handler acknowledges Discord before touching the database: Discord allows 3 seconds, and a fresh
    # connection to a hosted Postgres (Supabase) can take most of that. A modal can't follow an acknowledgement,
    # so the questionnaire's previous answers are fetched here, ahead of the "Continue" click.
    await interaction.response.defer(ephemeral=True, thinking=True)
    persona, previous = await asyncio.to_thread(ops.join_context, interaction.user.id)
    if persona:  # answering would overwrite the synthetic profile this account is playing
        await interaction.followup.send(f"This account plays **{persona}** in the demo, so it already has a profile "
                                        "(see `/me`). A lead can run `/kym unclaim` if you want to join as yourself.", ephemeral=True)
        return
    await interaction.followup.send(CONSENT_TEXT, view=ConsentView(previous), ephemeral=True)


class ConsentView(discord.ui.View):
    def __init__(self, previous: dict[str, str]) -> None:
        super().__init__(timeout=900)
        self.previous = previous
        self.scopes = {"store", "intro"}
        options = [discord.SelectOption(label=label[:100], value=scope, default=scope in self.scopes)
                   for scope, label in CONSENT_SCOPES.items()]
        self.choice = discord.ui.Select(placeholder="What may the bot do?", min_values=1, max_values=len(options), options=options)
        self.choice.callback = self._chosen
        self.add_item(self.choice)

    async def _chosen(self, interaction: discord.Interaction) -> None:
        self.scopes = set(self.choice.values)
        await interaction.response.defer()

    @discord.ui.button(label="Continue to the questions", style=discord.ButtonStyle.success)
    async def go(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        if not {"store", "intro"} <= self.scopes:
            await interaction.response.send_message(
                "Storing your answers and weekly intros are both needed to take part. Pick both (you can `/forget` any time).",
                ephemeral=True)
            return
        await interaction.response.send_modal(QuestionnaireModal(self.scopes, self.previous))


class QuestionnaireModal(discord.ui.Modal, title="Know Your Members"):
    def __init__(self, scopes: set[str], previous: dict[str, str]) -> None:
        super().__init__(timeout=1800)
        self.scopes = scopes
        self.inputs: dict[str, discord.ui.TextInput] = {}
        for q in QUESTIONS:
            field = discord.ui.TextInput(
                label=q.label, placeholder=q.placeholder, required=q.required, max_length=q.max_length,
                style=discord.TextStyle.paragraph if q.paragraph else discord.TextStyle.short, default=previous.get(q.key))
            self.inputs[q.key] = field
            self.add_item(field)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        answers = {key: field.value for key, field in self.inputs.items()}
        try:
            result = await asyncio.to_thread(ops.do_onboard, interaction.user.id, interaction.user.display_name, answers, self.scopes)
        except Exception:
            log.exception("onboarding failed for %s", interaction.user.id)
            await interaction.followup.send("Sorry, saving your answers failed. It's in the bot log; please try `/join` again.", ephemeral=True)
            return
        labels = {"building": "Building", "offers": "Can help with", "wants": "Looking for", "speak_topic": "Would speak on"}
        lines = [f"• **{labels.get(kind, kind)}:** {text}" for kind, text in result["facts"]]
        handles = ", ".join(f"{k}: {v}" for k, v in result["handles"].items())
        msg = ("You're in. Here's what I understood (I'll only use this, and say where it came from):\n" + "\n".join(lines)
               + (f"\n• **Links:** {handles}" if handles else "")
               + "\n\nIf anything's wrong, run `/join` again. Your first intro arrives with the next weekly round.")
        await interaction.followup.send(msg[:1990], ephemeral=True)
        if result["enrich"] and result["handles"]:
            asyncio.create_task(asyncio.to_thread(ops.do_enrich, result["member_id"]))


# ------------------------------------------------------------------ intro buttons

class IntroButton(discord.ui.DynamicItem[discord.ui.Button], template=r"kym:intro:(?P<id>\d+):(?P<action>accept|decline|pause)"):
    LABELS = {"accept": ("Yes, introduce us", discord.ButtonStyle.success),
              "decline": ("Not this week", discord.ButtonStyle.secondary),
              "pause": ("Pause my intros", discord.ButtonStyle.danger)}

    def __init__(self, intro_id: int, action: str) -> None:
        label, style = self.LABELS[action]
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"kym:intro:{intro_id}:{action}"))
        self.intro_id, self.action = intro_id, action

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(int(match["id"]), match["action"])

    async def callback(self, interaction: discord.Interaction) -> None:
        bot: KYMBot = interaction.client  # type: ignore[assignment]
        await interaction.response.defer()
        result = await asyncio.to_thread(ops.respond, self.intro_id, interaction.user.id, self.action, utcnow())
        outcome = result["outcome"]
        if outcome == "not_yours":
            await interaction.followup.send("This intro isn't addressed to you.", ephemeral=True)
            return
        if outcome == "closed":
            await interaction.edit_original_response(content="This intro has already closed.", view=None)
            return
        other = result.get("other_name", "them")
        text = {
            "waiting": f"Great. I'll connect you as soon as {other} says yes too.",
            "connect": f"You both said yes. I'm opening a private thread for you and {other} now.",
            "declined": ("Paused. You won't get intros until you run `/resume`." if self.action == "pause"
                         else "No problem, skipped for this week."),
        }[outcome]
        await interaction.edit_original_response(content=text, view=None)
        if outcome == "connect":
            await bot.open_intro_thread(self.intro_id)
        elif outcome == "declined" and result.get("other_accepted"):
            await bot.dm(result["other_discord_id"], "This week's intro didn't come together. Your next one arrives with the next weekly round.")


class FeedbackButton(discord.ui.DynamicItem[discord.ui.Button], template=r"kym:fb:(?P<id>\d+):(?P<value>useful|meh|didnt_happen)"):
    LABELS = {"useful": ("Yes, it was useful", discord.ButtonStyle.success),
              "meh": ("We talked, it was OK", discord.ButtonStyle.secondary),
              "didnt_happen": ("Didn't happen", discord.ButtonStyle.secondary)}

    def __init__(self, intro_id: int, value: str) -> None:
        label, style = self.LABELS[value]
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"kym:fb:{intro_id}:{value}"))
        self.intro_id, self.value = intro_id, value

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(int(match["id"]), match["value"])

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        ok = await asyncio.to_thread(ops.feedback, self.intro_id, interaction.user.id, self.value, utcnow())
        await interaction.followup.send("Thanks, noted. It helps the next intros." if ok
                                        else "Only the two people in this intro can answer.", ephemeral=True)


class VetoButton(discord.ui.DynamicItem[discord.ui.Button], template=r"kym:veto:(?P<id>\d+)"):
    def __init__(self, intro_id: int) -> None:
        super().__init__(discord.ui.Button(label=f"Veto #{intro_id}", style=discord.ButtonStyle.danger, custom_id=f"kym:veto:{intro_id}"))
        self.intro_id = intro_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(int(match["id"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if not is_lead(interaction):
            await interaction.response.send_message("Only community leads can veto intros.", ephemeral=True)
            return
        await interaction.response.defer()
        ok = await asyncio.to_thread(ops.veto_intro, self.intro_id, f"lead {interaction.user.display_name} (Discord)", utcnow())
        await interaction.followup.send(f"Intro #{self.intro_id} vetoed." if ok else "Too late: it was already sent or vetoed.", ephemeral=True)


class ReviveButton(discord.ui.DynamicItem[discord.ui.Button], template=r"kym:rev:(?P<id>\d+):(?P<action>post|skip)"):
    def __init__(self, revival_id: int, action: str) -> None:
        label, style = ("Post it", discord.ButtonStyle.success) if action == "post" else ("Skip", discord.ButtonStyle.secondary)
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"kym:rev:{revival_id}:{action}"))
        self.revival_id, self.action = revival_id, action

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(int(match["id"]), match["action"])

    async def callback(self, interaction: discord.Interaction) -> None:
        if not is_lead(interaction):
            await interaction.response.send_message("Only community leads can approve nudges.", ephemeral=True)
            return
        bot: KYMBot = interaction.client  # type: ignore[assignment]
        await interaction.response.defer()
        payload = await asyncio.to_thread(ops.get_revival, self.revival_id)
        if not payload or payload["status"] != "drafted":
            await interaction.edit_original_response(view=None)
            await interaction.followup.send("Already handled.", ephemeral=True)
            return
        if self.action == "skip":
            await asyncio.to_thread(ops.revival_skip, self.revival_id, interaction.user.display_name)
            await interaction.edit_original_response(content=f"Skipped: {payload['title']}", view=None, embed=None)
            return
        ok, detail = await bot.post_revival(payload)
        if ok:  # the draft's buttons go, so nobody posts it twice
            await interaction.edit_original_response(content=f"Posted by {interaction.user.display_name}: {detail}", view=None)
        await interaction.followup.send(("Posted: " if ok else "Couldn't post: ") + detail, ephemeral=True)


def _view(*items: discord.ui.Item) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for item in items:
        view.add_item(item)
    return view


CHAT_TYPES = (discord.MessageType.default, discord.MessageType.reply)


def is_chat(m: discord.Message) -> bool:
    """People talking, not Discord's system lines (joins, pins, "X started a thread", thread starters)."""
    return m.type in CHAT_TYPES


def message_row(m: discord.Message) -> dict:
    channel = m.channel
    thread = channel if isinstance(channel, discord.Thread) else None
    parent = channel.parent if thread else channel
    return {
        "id": m.id, "guild_id": m.guild.id if m.guild else None, "channel_id": parent.id if parent else channel.id,
        "channel_name": getattr(parent, "name", None), "thread_id": thread.id if thread else None,
        "thread_name": thread.name if thread else None, "author_id": m.author.id, "author_name": m.author.display_name,
        "author_is_bot": m.author.bot, "content": m.content or "", "created_at": m.created_at,
        "reply_to_id": m.reference.message_id if m.reference and m.reference.message_id else None,
        "mention_ids": [u.id for u in m.mentions], "jump_url": m.jump_url, "webhook_id": m.webhook_id,
    }


# ------------------------------------------------------------------ the bot

class KYMBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.tree = app_commands.CommandTree(self)
        self.tick_lock = asyncio.Lock()
        self.pending: list[discord.Message] = []

    async def setup_hook(self) -> None:
        self.add_dynamic_items(JoinButton, IntroButton, FeedbackButton, VetoButton, ReviveButton)
        register_commands(self)
        guild_id = get_settings().discord_guild_id
        if guild_id:
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        self.tick.start()

    async def on_ready(self) -> None:
        log.info("logged in as %s in %d server(s)", self.user, len(self.guilds))

    async def on_member_join(self, member: discord.Member) -> None:
        if member.bot:
            return
        await self.dm(member.id, "Welcome! Answer five quick questions and I'll introduce you to one person here each week, "
                                 "with the reasons why.", view=_view(JoinButton()))

    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or not is_chat(message):
            return
        private = await asyncio.to_thread(ops.private_channels)
        parent = message.channel.parent if isinstance(message.channel, discord.Thread) else message.channel
        if parent is not None and parent.id in private:
            return  # private intro threads and the lead channel are never mined
        self.pending.append(message)

    async def dm(self, discord_id: int | None, text: str, **kwargs) -> bool:
        if not discord_id:
            return False
        try:
            user = self.get_user(discord_id) or await self.fetch_user(discord_id)
            await user.send(text, **kwargs)
            return True
        except discord.HTTPException as exc:
            log.warning("DM to %s failed: %s", discord_id, exc)
            return False

    async def _channel(self, name: str) -> discord.abc.Messageable | None:
        channel_id = await asyncio.to_thread(ops.channel_setting, name)
        if not channel_id:
            return None
        try:
            return self.get_channel(channel_id) or await self.fetch_channel(channel_id)
        except discord.HTTPException:
            return None

    # -------------------------------------------------------------- intros

    async def deliver_offer(self, intro_id: int) -> str:
        """DM both people (or open a private note in #intros if their DMs are closed). On a demo server, synthetic
        members can't receive anything: an intro between two of them stays a preview for the lead, and one between
        a real and a synthetic member goes to the real person only, saying so."""
        p = await asyncio.to_thread(ops.intro_payload, intro_id)
        if not p or p["status"] != "proposed":
            return "skipped"
        sides = p["sides"]
        if all(info["synthetic"] for info in sides.values()):
            await asyncio.to_thread(ops.mark_preview, intro_id, utcnow(), "both are synthetic demo members: shown to the lead, not sent")
            return "preview"
        delivered, problems = {}, []
        for side in ("a", "b"):
            info, other = sides[side], sides["b" if side == "a" else "a"]
            if info["synthetic"]:
                delivered[side] = {"via": "none", "why": "synthetic demo member"}
                continue
            if not info["discord_id"]:
                problems.append(f"{info['name']} isn't on Discord")
                continue
            note = (f"\n\n_Demo: {info['other_name']} is a synthetic member, so they won't see this or answer._"
                    if other["synthetic"] else "")
            embed = discord.Embed(title=f"Your intro this week: {info['other_name']}", colour=BLURPLE,
                                  description=("\n".join(info["lines"]) + (f"\n\n**Suggested opener:** {p['opener']}" if p["opener"] else "")
                                               + note)[:4000])
            embed.set_footer(text=f"Confidence: {p['confidence']} · you both have to say yes before I connect you · reply within 3 days")
            view = _view(*(IntroButton(intro_id, action) for action in ("accept", "decline", "pause")))
            try:
                user = self.get_user(info["discord_id"]) or await self.fetch_user(info["discord_id"])
                msg = await user.send(embed=embed, view=view)
                delivered[side] = {"via": "dm", "message_id": msg.id}
            except discord.Forbidden:
                thread_id = await self._private_note(info["discord_id"], f"Intro for {info['name']}", embed, view)
                if thread_id:
                    delivered[side] = {"via": "private_thread", "thread_id": thread_id}
                else:
                    problems.append(f"{info['name']} has DMs closed and no intros channel is set")
            except discord.HTTPException as exc:
                problems.append(f"{info['name']}: {exc}")
        if len(delivered) == 2:
            await asyncio.to_thread(ops.mark_offered, intro_id, utcnow(), delivered)
            return "offered"
        await asyncio.to_thread(ops.mark_failed, intro_id, utcnow(), "; ".join(problems) or "not delivered")
        return "failed"

    async def _private_note(self, discord_id: int, name: str, embed: discord.Embed, view: discord.ui.View) -> int | None:
        channel = await self._channel("intros")
        if not isinstance(channel, discord.TextChannel):
            return None
        try:
            thread = await channel.create_thread(name=name[:100], type=discord.ChannelType.private_thread, invitable=False)
            await thread.add_user(discord.Object(id=discord_id))
            await thread.send(f"<@{discord_id}>", embed=embed, view=view,
                              allowed_mentions=discord.AllowedMentions(users=[discord.Object(id=discord_id)]))
            return thread.id
        except discord.HTTPException as exc:
            log.warning("private fallback for %s failed: %s", discord_id, exc)
            return None

    async def open_intro_thread(self, intro_id: int) -> None:
        p = await asyncio.to_thread(ops.intro_payload, intro_id)
        a, b = p["sides"]["a"], p["sides"]["b"]
        channel = await self._channel("intros")
        thread_id = None
        if isinstance(channel, discord.TextChannel):
            try:
                thread = await channel.create_thread(name=f"{a['name']} × {b['name']}"[:100], invitable=False,
                                                     type=discord.ChannelType.private_thread, auto_archive_duration=10080)
                for side in (a, b):
                    await thread.add_user(discord.Object(id=side["discord_id"]))
                body = (f"<@{a['discord_id']}>, meet <@{b['discord_id']}>. You both said yes.\n\n"
                        f"**Why {b['name']} might be useful to {a['name']}**\n" + "\n".join(a["lines"]) +
                        f"\n\n**Why {a['name']} might be useful to {b['name']}**\n" + "\n".join(b["lines"]) +
                        (f"\n\n**Suggested opener:** {p['opener']}" if p["opener"] else "") +
                        "\n\n_Only you two and server moderators can see this thread. I'll check in after a few days._")
                await thread.send(body[:1990], allowed_mentions=discord.AllowedMentions(
                    users=[discord.Object(id=a["discord_id"]), discord.Object(id=b["discord_id"])]))
                thread_id = thread.id
            except discord.HTTPException as exc:
                log.warning("could not open intro thread %s: %s", intro_id, exc)
        if thread_id is None:
            for me, other in ((a, b), (b, a)):
                await self.dm(me["discord_id"], f"You and <@{other['discord_id']}> both said yes! I couldn't open a private "
                                                f"thread, so message them directly.")
        await asyncio.to_thread(ops.mark_connected, intro_id, thread_id, utcnow())

    async def post_followup(self, item: dict) -> None:
        view = _view(*(FeedbackButton(item["id"], v) for v in ("useful", "meh", "didnt_happen")))
        text = "Did you two get a chance to talk? One tap helps me make better intros."
        sent = False
        if item["thread_id"]:
            try:
                thread = self.get_channel(item["thread_id"]) or await self.fetch_channel(item["thread_id"])
                await thread.send(text, view=view)
                sent = True
            except discord.HTTPException as exc:
                log.warning("follow-up in thread %s failed: %s", item["thread_id"], exc)
        if not sent:
            for uid in (item["a"], item["b"]):
                await self.dm(uid, text, view=view)
        await asyncio.to_thread(ops.mark_followed_up, item["id"], utcnow())

    async def thread_counts(self, item: dict) -> tuple[int | None, int | None]:
        if not item["thread_id"]:
            return None, None
        try:
            thread = self.get_channel(item["thread_id"]) or await self.fetch_channel(item["thread_id"])
            counts = Counter()
            async for m in thread.history(limit=500):
                counts[m.author.id] += 1
            return counts[item["a"]], counts[item["b"]]
        except discord.HTTPException:
            return None, None

    async def post_proposals(self, week: str) -> None:
        channel = await self._channel("lead")
        intros = await asyncio.to_thread(ops.proposed_intros, week)
        if channel is None or not intros:
            return
        s = get_settings()
        send_at = clock.at(week, s.send_weekday, s.send_hour)
        from .. import notion

        when = (f"They go out <t:{int(send_at.timestamp())}:R>" if s.autopilot else "They go out when you run `/kym run step:send`")
        where = "here or in Notion" if notion.configured() else "here"
        await channel.send(f"**{len(intros)} intros proposed for {week}.** {when} unless vetoed {where}.")
        blocks = [(f"**#{i['id']} {i['a']} × {i['b']}** ({i['confidence']}, {i['score']:.2f}, writer: {i['writer']})"
                   + (f" · _{i['note']}_" if i["note"] else "") + "\n" + "\n".join(f"  • {w}" for w in i["why"]))[:1900]
                  for i in intros]
        chunk: list[int] = []
        for n in range(len(intros)):  # up to 5 per message (one row of veto buttons), never cut mid-intro
            chunk.append(n)
            last = n == len(intros) - 1
            if last or len(chunk) == 5 or sum(len(blocks[k]) + 1 for k in [*chunk, n + 1]) > 1990:
                await channel.send("\n".join(blocks[k] for k in chunk),
                                   view=_view(*(VetoButton(intros[k]["id"]) for k in chunk)))
                chunk = []

    # -------------------------------------------------------------- reviver

    async def flush_messages(self) -> int:
        pending, self.pending = self.pending, []
        return await asyncio.to_thread(ops.store_messages, [message_row(m) for m in pending], {}) if pending else 0

    async def sync_history(self, days: int | None = None) -> int:
        since = utcnow() - timedelta(days=days or get_settings().history_days)
        private = await asyncio.to_thread(ops.private_channels)
        rows, cursors = [], {}
        for guild in self.guilds:
            me = guild.me
            containers: list = []
            for ch in [*guild.text_channels, *guild.forums]:
                if ch.id in private or not ch.permissions_for(me).read_message_history:
                    continue
                if isinstance(ch, discord.TextChannel):
                    containers.append(ch)
                containers += list(ch.threads)
                try:
                    containers += [t async for t in ch.archived_threads(limit=50)]
                except discord.HTTPException:
                    pass
            for container in containers:
                last = await asyncio.to_thread(ops.last_synced, container.id)
                newest = last
                try:
                    async for m in container.history(limit=1000, after=discord.Object(id=last) if last else since, oldest_first=True):
                        if is_chat(m):
                            rows.append(message_row(m))
                        newest = max(newest or 0, m.id)
                except discord.HTTPException as exc:
                    log.info("history of %s skipped: %s", getattr(container, "name", container.id), exc)
                if newest:
                    cursors[container.id] = newest
        stored = 0
        for start in range(0, max(len(rows), 1), 400):
            stored += await asyncio.to_thread(ops.store_messages, rows[start:start + 400], cursors if start == 0 else {})
        return stored

    async def post_revival(self, payload: dict) -> tuple[bool, str]:
        target_id = payload["thread_id"] or payload["channel_id"]
        try:
            target = self.get_channel(target_id) or await self.fetch_channel(target_id)
            mentions = discord.AllowedMentions(users=[discord.Object(id=u) for u in payload["mentions"]], everyone=False, roles=False)
            if payload["thread_id"]:
                msg = await target.send(payload["text"][:1990], allowed_mentions=mentions)
            else:
                anchor = target.get_partial_message(payload["anchor_message_id"])
                msg = await anchor.reply(payload["text"][:1990], allowed_mentions=mentions, mention_author=False)
        except discord.Forbidden:
            await asyncio.to_thread(ops.revival_failed, payload["id"], "bot lacks permission to post there", utcnow())
            return False, "missing permission"
        except discord.HTTPException as exc:
            await asyncio.to_thread(ops.revival_failed, payload["id"], f"Discord error: {exc}", utcnow())
            return False, str(exc)
        await asyncio.to_thread(ops.revival_posted, payload["id"], msg.id, utcnow())
        return True, msg.jump_url

    async def revive(self, now, autopost: bool | None = None) -> dict:
        await self.flush_messages()
        synced = await self.sync_history()
        drafts = await asyncio.to_thread(ops.revive_scan, now)
        waiting = await asyncio.to_thread(ops.pending_revivals, {d["id"] for d in drafts})
        posted = []
        if get_settings().revive_autopost if autopost is None else autopost:
            for payload in await asyncio.to_thread(ops.revivals_to_post, now):
                ok, detail = await self.post_revival(payload)
                posted.append((payload["title"], ok, detail))
        else:
            lead = await self._channel("lead")
            for d in [*drafts, *waiting]:
                if lead is None:
                    break
                embed = discord.Embed(title=f"Dead thread in #{d['channel_name']}: {d['title'][:200]}", colour=0xF0B232,
                                      description=(f"**Status:** {d['status']} ({d['rule']}, silent {d['silent_days']} days)\n"
                                                   f"**Why:** {d['reason'][:500]}\n\n" + (f"**Draft:**\n{d['text']}" if d["text"] else ""))[:4000])
                view = _view(ReviveButton(d["id"], "post"), ReviveButton(d["id"], "skip")) if d["status"] == "drafted" else None
                await lead.send(embed=embed, **({"view": view} if view else {}))
        return {"synced": synced, "candidates": len(drafts), "drafted": sum(d["status"] == "drafted" for d in drafts),
                "waiting": len(waiting), "posted": posted}

    # -------------------------------------------------------------- the weekly cycle

    @tasks.loop(minutes=5)
    async def tick(self) -> None:
        if self.tick_lock.locked():
            return
        async with self.tick_lock:
            try:
                if get_settings().autopilot:
                    await self.run_due(utcnow())
                else:
                    await self.flush_messages()  # manual mode: steps run only via /kym run
            except Exception:
                log.exception("tick failed")

    @tick.before_loop
    async def _before_tick(self) -> None:
        await self.wait_until_ready()

    async def run_due(self, now) -> list[str]:
        done = []
        if await self.flush_messages():
            done.append("stored new messages")
        if await asyncio.to_thread(ops.propose_due, now):
            r = await asyncio.to_thread(ops.run_propose, now)
            done.append(f"proposed {r['intros']} intros")
            await self.post_proposals(r["week"])
        vetoed = await asyncio.to_thread(ops.apply_notion_vetoes, now)
        if vetoed:
            done.append(f"{vetoed} vetoed in Notion")
        for intro_id in await asyncio.to_thread(ops.due_offers, now):
            done.append(f"intro {intro_id}: {await self.deliver_offer(intro_id)}")
        for item in await asyncio.to_thread(ops.expire, now):
            for uid in item["accepted"]:
                await self.dm(uid, "This week's intro didn't come together: no reply from the other side. Your next one arrives with the next weekly round.")
            done.append(f"intro {item['id']} expired")
        for item in await asyncio.to_thread(ops.due_followups, now):
            await self.post_followup(item)
            done.append(f"followed up intro {item['id']}")
        for item in await asyncio.to_thread(ops.due_outcomes, now):
            a, b = await self.thread_counts(item)
            done.append(f"intro {item['id']}: {await asyncio.to_thread(ops.finalize, item['id'], now, a, b)}")
        week = await asyncio.to_thread(ops.digest_due, now)
        if week:
            d = await asyncio.to_thread(ops.make_digest, week)
            lead = await self._channel("lead")
            if lead:
                await lead.send(("\n".join(d["lines"]) + (f"\nNotion: {d['notion_url']}" if d["notion_url"] else ""))[:1990])
            done.append(f"digest {week}")
        done += await self._daily_jobs(now)
        return done

    async def _daily_jobs(self, now) -> list[str]:
        done = []
        local = now.astimezone(clock._tz(None))
        with session_scope() as session:
            last_scan = get_setting(session, "revive:last_scan")
            last_eval = get_setting(session, "revive:last_eval")
            last_notion = get_setting(session, "notion:last_sync")
        today = local.date().isoformat()
        if local.hour >= 10 and last_scan != today:
            r = await self.revive(now)
            done.append(f"reviver: {r['candidates']} candidates, {r['drafted']} drafted, {len(r['posted'])} posted")
            with session_scope() as session:
                set_setting(session, "revive:last_scan", today)
        if not last_eval or now.isoformat()[:13] != last_eval:
            await self.sync_history(days=10)
            n = await asyncio.to_thread(ops.revive_evaluate, now)
            done.append(f"evaluated {n} nudges")
            with session_scope() as session:
                set_setting(session, "revive:last_eval", now.isoformat()[:13])
        from .. import notion

        if notion.configured() and (not last_notion or now.isoformat()[:15] > (last_notion or "")[:15]):
            try:
                counts = await asyncio.to_thread(ops.notion_sync)
                done.append(f"notion sync {counts}")
            except Exception as exc:
                log.warning("notion sync failed: %s", exc)
            with session_scope() as session:
                set_setting(session, "notion:last_sync", now.isoformat())
        return done


# ------------------------------------------------------------------ slash commands

def speaker_embeds(result: dict) -> list[discord.Embed]:
    embeds = []
    for i, c in enumerate(result["shortlist"], 1):
        ev = "\n".join(f"• [{e['source']}]({e['url']}): {e['text'][:140]}" if e.get("url") else f"• {e['source']}: {e['text'][:140]}"
                       for e in c["evidence"][:3])
        extra = [f"**London:** {c['london']['basis']}"]
        extra += [f"**Same person:** {line}" for line in c.get("identity", [])[:2]]
        extra += [f"_Unverified: {u}_" for u in c.get("unverified", [])[:2]]
        profiles = " · ".join(f"[{k}]({v})" for k, v in list(c.get("profiles", {}).items())[:3])
        embed = discord.Embed(title=f"{i}. {c['name']}" + (" · member" if c["member"] else ""), colour=0x57F287 if c["member"] else BLURPLE,
                              description=(f"{c.get('why', '')}\n\n{ev}\n" + "\n".join(extra) + (f"\n{profiles}" if profiles else ""))[:900])
        embed.set_footer(text=f"score {c['score']} · relevance {c['relevance']} · built {c['build']} · spoke/hosted {c['speak']}")
        embeds.append(embed)
    return embeds


def register_commands(bot: KYMBot) -> None:
    tree = bot.tree

    @tree.command(name="join", description="Answer 5 quick questions and get one good intro a week")
    async def join(interaction: discord.Interaction) -> None:
        await start_join(interaction)

    @tree.command(name="me", description="See what the bot knows about you, and where it came from")
    async def me(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        p = await asyncio.to_thread(ops.my_profile, interaction.user.id)
        if not p:
            await interaction.followup.send("I don't have anything about you. Run `/join` to take part.", ephemeral=True)
            return
        facts = "\n".join(f"• {f['kind']}: {f['text']} ({f['source']})" for f in p["facts"][:20])
        consent = ", ".join(k for k, v in p["consent"].items() if v) or "none"
        intros = "\n".join(f"• {i['week']}: {i['with']} ({i['status']}{', ' + i['outcome'] if i['outcome'] else ''})" for i in p["intros"])
        await interaction.followup.send(
            (f"**Intros:** {p['frequency']} · **Consent:** {consent}\n{facts}\n" + (f"**Recent intros**\n{intros}" if intros else ""))[:1990],
            ephemeral=True)

    @tree.command(name="pause", description="Pause your weekly intros")
    async def pause(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        ok = await asyncio.to_thread(ops.change_frequency, interaction.user.id, "paused")
        await interaction.followup.send("Paused. `/resume` to restart." if ok else "You haven't joined yet.", ephemeral=True)

    @tree.command(name="resume", description="Resume weekly (or fortnightly) intros")
    @app_commands.choices(frequency=[app_commands.Choice(name="weekly", value="weekly"), app_commands.Choice(name="fortnightly", value="fortnightly")])
    async def resume(interaction: discord.Interaction, frequency: app_commands.Choice[str] | None = None) -> None:
        value = frequency.value if frequency else "weekly"
        await interaction.response.defer(ephemeral=True, thinking=True)
        ok = await asyncio.to_thread(ops.change_frequency, interaction.user.id, value)
        await interaction.followup.send(f"Intros are back on, {value}." if ok else "You haven't joined yet. Run `/join`.", ephemeral=True)

    @tree.command(name="forget", description="Delete everything the bot holds about you")
    async def forget(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        r = await asyncio.to_thread(ops.forget_me, interaction.user.id)
        if r and r.startswith("released "):
            text = f"Done. This account no longer plays {r.removeprefix('released ')} in the demo; their synthetic profile stays."
        else:
            text = "Done. Your answers, facts and links are deleted and you won't get intros." if r else "I don't hold anything about you."
        await interaction.followup.send(text, ephemeral=True)

    kym = app_commands.Group(name="kym", description="Community lead tools", default_permissions=discord.Permissions(manage_guild=True))

    @kym.command(name="setup", description="Choose the intros channel (private threads) and the lead channel")
    async def setup(interaction: discord.Interaction, intros_channel: discord.TextChannel, lead_channel: discord.TextChannel) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await asyncio.to_thread(ops.save_channels, intros_channel.id, lead_channel.id)
        perms = intros_channel.permissions_for(interaction.guild.me)
        missing = [name for name, ok in (("View Channel", perms.view_channel), ("Create Private Threads", perms.create_private_threads),
                                         ("Send Messages in Threads", perms.send_messages_in_threads)) if not ok]
        await interaction.followup.send(
            f"Saved. Intro threads go in {intros_channel.mention}; proposals, drafts and digests in {lead_channel.mention}."
            + (f"\n⚠️ Missing permissions in {intros_channel.mention}: {', '.join(missing)}" if missing else ""), ephemeral=True)

    @kym.command(name="post-join", description="Post a 'Join intros' button in this channel")
    async def post_join(interaction: discord.Interaction) -> None:
        await interaction.channel.send("**Get one useful intro a week.** Five quick questions, opt-in, delete any time.", view=_view(JoinButton()))
        await interaction.response.send_message("Posted.", ephemeral=True)

    @kym.command(name="run", description="Run a step of the weekly cycle now (demo / catch-up)")
    @app_commands.choices(step=[app_commands.Choice(name=n, value=n) for n in ("propose", "send", "followup", "outcomes", "digest", "due")])
    async def run(interaction: discord.Interaction, step: app_commands.Choice[str], force: bool = False) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        now = utcnow()
        if step.value == "propose":
            r = await asyncio.to_thread(ops.run_propose, now, force)
            await bot.post_proposals(r["week"])
            msg = (f"Already proposed for {r['week']} (use force to redo)" if r["already"] else
                   f"{r['intros']} intros for {r['eligible']} eligible members ({r['unmatched']} unmatched). Rerank: {r['rerank']}")
        elif step.value == "send":
            ids = [i["id"] for i in await asyncio.to_thread(ops.proposed_intros)]
            results = Counter([await bot.deliver_offer(i) for i in ids])
            msg = (f"Sent {results['offered']} intros; {results['failed']} failed (see `/kym log`)"
                   + (f"; {results['preview']} between synthetic members kept as previews" if results["preview"] else "") + ".")
        elif step.value == "followup":
            items = await asyncio.to_thread(ops.due_followups, now + timedelta(days=30))
            for item in items:
                await bot.post_followup(item)
            msg = f"Follow-ups posted for {len(items)} intros."
        elif step.value == "outcomes":
            items = await asyncio.to_thread(ops.due_outcomes, now + timedelta(days=30))
            outcomes = [await asyncio.to_thread(ops.finalize, item["id"], now, *(await bot.thread_counts(item))) for item in items]
            msg = f"Outcomes: {Counter(outcomes)}"
        elif step.value == "digest":
            d = await asyncio.to_thread(ops.make_digest, clock.week_id(now))
            msg = "\n".join(d["lines"]) + (f"\nNotion: {d['notion_url']}" if d["notion_url"] else "")
        else:
            msg = "Ran: " + ("; ".join(await bot.run_due(now)) or "nothing was due")
        await interaction.followup.send(msg[:1990], ephemeral=True)

    @kym.command(name="speakers", description="Find speakers for a topic: members and London builders, with evidence")
    async def speakers(interaction: discord.Interaction, topic: str, private: bool = False) -> None:
        await interaction.response.defer(thinking=True, ephemeral=private)
        result = await asyncio.to_thread(ops.speakers, topic)
        notes = "; ".join(result.get("notes", []))
        header = (f"**Speakers for \"{topic}\"**: {len(result['shortlist'])} picks from {result.get('people_considered', 0)} people "
                  f"(keywords: {', '.join(result['expansion']['keywords'])})" + (f"\n_Notes: {notes}_" if notes else "")
                  + (f"\nFull shortlist in Notion: {result['notion_url']}" if result.get("notion_url") else ""))
        embeds = speaker_embeds(result)
        await interaction.followup.send(header[:1990], embeds=embeds[:4], ephemeral=private)
        for start in range(4, len(embeds), 4):
            await interaction.followup.send(embeds=embeds[start:start + 4], ephemeral=private)

    @kym.command(name="revive", description="Find dead threads worth restarting and draft nudges")
    async def revive(interaction: discord.Interaction, autopost: bool = False) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        r = await bot.revive(utcnow(), autopost=autopost)
        posted = "\n".join(f"• {t[:60]}: {'posted ' + d if ok else 'failed: ' + d}" for t, ok, d in r["posted"])
        waiting = f" {r['waiting']} earlier draft(s) still need a decision." if r.get("waiting") else ""
        await interaction.followup.send(f"Synced {r['synced']} messages; {r['candidates']} dead threads found, {r['drafted']} worth reviving.{waiting}"
                                        + ("\n" + posted if posted else "\nDrafts are in the lead channel with Post/Skip buttons."), ephemeral=True)

    @kym.command(name="log", description="What the bot did recently, including failures")
    async def log_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        rows = await asyncio.to_thread(ops.recent_log, 15)
        icon = {"ok": "✅", "failed": "❌", "skipped": "⏭️"}
        text = "\n".join(f"{icon.get(r['status'], '•')} `{r['at']:%d %b %H:%M}` **{r['kind']}** {r['target'][:40]}: {r['detail'][:120]}" for r in rows)
        await interaction.followup.send(text[:1990] or "Nothing yet.", ephemeral=True)

    @kym.command(name="claim", description="Demo: a real account plays a synthetic member, so their intros and pings arrive")
    @app_commands.describe(persona="The synthetic member to play", user="The Discord account that will play them")
    async def claim(interaction: discord.Interaction, persona: str, user: discord.Member) -> None:
        if user.bot:
            await interaction.response.send_message("Bots can't play members: Discord doesn't let bots DM each other.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        r = await asyncio.to_thread(ops.claim_persona, persona, user.id, user.display_name, utcnow())
        if not r["ok"]:
            await interaction.followup.send(r["error"], ephemeral=True)
            return
        facts = "\n".join(f"• {kind}: {text}" for kind, text in r["facts"])
        await interaction.followup.send(
            f"{user.mention} now plays **{r['name']}**: their intros, DMs and pings go to that account. "
            f"Re-run `/kym run step:propose force:true` so this week's round uses it.\n{facts}"[:1990], ephemeral=True)

    @claim.autocomplete("persona")
    async def claim_persona_choices(_interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        return [app_commands.Choice(name=n, value=n) for n in await asyncio.to_thread(ops.persona_names, current)]

    @kym.command(name="unclaim", description="Demo: stop a real account playing a synthetic member")
    async def unclaim(interaction: discord.Interaction, persona: str) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        r = await asyncio.to_thread(ops.release_persona, persona, utcnow())
        await interaction.followup.send(f"{r['name']} is synthetic again." if r["ok"] else r["error"], ephemeral=True)

    @unclaim.autocomplete("persona")
    async def unclaim_persona_choices(_interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        return [app_commands.Choice(name=n, value=n) for n in await asyncio.to_thread(ops.persona_names, current, True)]

    @kym.command(name="notion-sync", description="Push members, intros, revivals and the bot log to Notion now")
    async def notion_sync(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            counts = await asyncio.to_thread(ops.notion_sync)
            await interaction.followup.send(f"Synced: {counts}", ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f"Notion sync failed: {exc}", ephemeral=True)

    tree.add_command(kym)


def run() -> None:
    token = get_settings().discord_token
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set (see README: Discord setup)")
    KYMBot().run(token, log_handler=None)
