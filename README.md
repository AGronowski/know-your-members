# Know Your Members

A Discord bot plus a CLI for community leads. It:

1. **Asks members five quick questions** with explicit, revocable consent: what they're building, what they can help with, what they want and whether they'd speak. Answers become short, tagged facts, each with its source.
2. **Makes one good intro a week per member.** Both people get the reasons, with links to the evidence. Both must say yes before a private thread opens. The bot follows up and records what happened. It keeps running without the organiser.
3. **Finds speakers for a topic.** Candidates come from consenting members plus London builders found on Luma, GitHub, Devpost and Hacker News. Every name comes with linked, dated evidence, how identities were matched across sources, and what couldn't be verified.
4. **Revives dead threads.** It finds unanswered questions and discussions that fizzled, decides whether they're worth restarting, drafts a nudge that adds something, tags only members who agreed to be tagged, and logs whether anyone replied.

Everything the bot does is written to a bot log, failures included. The lead sees it in Notion, or in Discord with `/kym log`, plus a weekly digest.

## How it works

```
Discord ──► questionnaire ─► facts (text, tags, source URL, consent) ─┐
Luma · GitHub · Devpost · HN ─► cached, rate-limited fetcher ─────────┤
                                                                      ▼
                  SQLite / Supabase Postgres: members, consent log, facts, intros, revivals, bot log
                                                                      │
      weekly state machine (idempotent; catches up after restarts)    │
      propose (Mon) ─► lead veto window ─► offer, double opt-in (Tue) ─► private thread ─► follow-up ─► outcome
                                                                      │
                                  Notion: Members · Intros (Veto) · Revivals · Bot log · digests · speaker pages
```

- **Matching:**
  - It scores how well A's *wants* meet B's *offers* in both directions. The score uses topic tags, weighted by how rare they are in the community, plus embedding similarity, and pairs that help both people score higher.
  - Pairs who already talk in chat get a penalty. Pairs that bridge communities, members unmatched last week, and responsive members get a bonus.
  - An LLM judge (Claude) rates each member's top candidates.
  - "One intro per member per week" is solved globally as a maximum-weight matching (networkx). The weights are squared scores, so the solver prefers one strong intro over two long shots.
- **Writing:**
  - Grok writes the intro, the nudge and the speaker blurbs, but it only sees the facts behind the match and must cite them by id.
  - Code drops any claim that cites nothing, contains a link, or uses a number not found in its sources.
  - Links are added by code from the facts, never by the model. If a claim fails these checks, a template quoting the members' own words is used instead, and it is labelled `writer: template`.
- **Honest about failures:**
  - The bot log records no-replies, declined intros, nudges nobody answered, sources that were down and claims that were dropped.
  - Speaker cards say "London inferred from events, not confirmed" when that's the case.
  - People are merged across sources only on shared handles or websites, never on name alone.

## Quick start (no keys needed)

```bash
uv sync
uv run kym seed-demo                               # 34 synthetic members + chat history
uv run kym match --dry-run --include-synthetic     # this week's intros with reasons and scores
uv run kym simulate                                # four weeks of the real cycle with simulated members
uv run kym revive --dry-run                        # dead threads worth restarting, with drafted nudges
uv run kym crawl-luma london                       # cache upcoming London events (hosts, featured guests)
uv run kym speakers "voice AI agents"              # shortlist with evidence
uv run kym status                                  # which integrations are configured
uv run pytest                                      # 46 tests, offline
```

Without LLM keys everything still runs: extraction uses keyword rules, intros use templates and the reviver judges by rules. Add keys to `.env` (see `.env.example`) for the real thing.

## Discord setup (about 10 minutes)

1. Go to <https://discord.com/developers/applications> → **New Application** → **Bot** → **Reset Token**, and put the token in `DISCORD_TOKEN`.
2. On the same page, turn on the privileged intents **Server Members** and **Message Content**.
3. Run `uv run kym invite-url <application id>`, open the link and add the bot to your server. Put the server id in `DISCORD_GUILD_ID`.
4. Create two channels:
   - **#intros**: members can view it, and the bot needs *Create Private Threads* and *Send Messages in Threads*. Private intro threads live here.
   - **#kym-lead**: leads only. Proposals with veto buttons, nudge drafts and digests go here.
5. Run `uv run kym bot`. Then, in Discord:
   - Run `/kym setup intros_channel:#intros lead_channel:#kym-lead`.
   - Run `/kym post-join` in your welcome channel.

Member commands: `/join`, `/me` (what the bot knows about you, with sources), `/pause`, `/resume`, `/forget` (deletes your data).

Lead commands (Manage Server permission): `/kym run step:<propose|send|followup|outcomes|digest|due>`, `/kym speakers`, `/kym revive`, `/kym log`, `/kym notion-sync`.

## Notion and Supabase

- **Notion:**
  1. Create an internal integration at <https://www.notion.so/my-integrations>.
  2. Share a page with it, and put the token and page id in `.env`.
  3. Run `uv run kym notion-setup`. This creates four databases: Members, Intros, Thread revivals and Bot log.

  The bot then syncs every 10 minutes and reads the **Veto** checkbox before sending intros. Speaker shortlists and weekly digests become pages.
- **Supabase:** set `DATABASE_URL` to the *Session pooler* connection string. Tables are created on start. Without it, SQLite at `data/kym.db` is used.

## Demo server

Discord has no fake users: bots can't create accounts, and scripting user accounts breaks Discord's terms. So the demo mixes a synthetic community with a few real accounts:

- **The synthetic members live in the bot's database.** `kym discord-seed` replays their chat into real channels through a webhook, under each member's name and a generated avatar. Discord marks those messages as coming from an app. Messages can't be backdated, so they show today's time; the bot keeps the dataset's timestamps, which is what the reviver reads.
- **Real accounts play synthetic members** (`/kym claim`), so the matcher's own picks reach real phones. The matcher is untouched: claiming only decides whose phone buzzes.
  - **You need two accounts**, one for each side of an intro, because both have to say yes before the private thread opens. The lead's own account can be one of them.
  - **A third account** lets a revived thread ping a real phone.
- **Judges can `/join` as themselves.** If they're matched with a synthetic member, only they get the DM, and it says so. Intros between two synthetic members are shown to the lead as previews and never sent.

### Setup (about 15 minutes)

1. **Bot account:**
   - In the [Developer Portal](https://discord.com/developers/applications), choose **New Application** → **Bot** → **Reset Token**, and put the token in `.env` as `DISCORD_TOKEN`.
   - Turn on **Server Members** and **Message Content**.
2. **Server:**
   - Create a server in the Discord app.
   - Turn on Developer Mode (User Settings → Advanced), right-click the server and choose **Copy Server ID**. Put it in `DISCORD_GUILD_ID`.
3. **Invite:** run `uv run kym invite-url <application id> --demo` and open the link. `--demo` adds Manage Channels and Manage Webhooks, which seeding needs.
4. **`.env` for the demo:**
   - `KYM_INCLUDE_SYNTHETIC=1`, so synthetic members are matched.
   - `KYM_AUTOPILOT=0`. Otherwise a bot started mid-week proposes and sends at once; with it off, nothing happens until you run `/kym run`.
   - Optionally `DATABASE_URL=sqlite:///data/demo-server.db`, to keep the demo apart from anything real.
5. **Seed:** run `uv run kym discord-seed`. It creates:
   - #welcome, with a note saying the members are synthetic and the **Join intros** button;
   - #general, #help, #show-and-tell and #events, with the chat in threads;
   - #intros;
   - a leads-only #kym-lead.

   It also points the bot at those channels. Run it again any time: it only posts what's missing.
6. **Luma:** run `uv run kym crawl-luma london` now. Luma rate-limits, so never do this on stage.
7. **Bot:** run `uv run kym bot`. Then, in Discord:
   - `/kym claim persona:Aisha Rahman user:@you`;
   - `/kym claim persona:Fatima Bello user:@second-account`;
   - optionally `/kym claim persona:Emily Zhang user:@third-account`. The reviver picks Emily for Hannah's unanswered Postgres question.
8. **Rehearse the run sheet once.** This also caches Grok's intro text, so the live run is quick. Then:
   - stop the bot;
   - run `uv run kym demo-reset`. It deletes intros, nudges, the bot log and the bot's Discord posts (DMs, intro threads, #kym-lead). Without a reset, Aisha and Fatima could never be paired again, because pairs never repeat;
   - start the bot again.

### Run sheet (about 6 minutes)

1. **Server tour:** show the channels, and say the members are synthetic and why their messages are marked as an app. Meanwhile, a judge scans a QR code of the invite link, presses **Join intros**, consents and answers five questions. `/me` then shows the facts the bot extracted, with sources.
2. **Propose:** run `/kym run step:propose force:true`. #kym-lead lists every intro with reasons quoted from members' own words, confidence, the writer used, a note on demo-only ones, and a **Veto** button. Veto one.
3. **Send:** run `/kym run step:send`. Both phones buzz, both tap **Yes** and a private thread opens in #intros with a suggested opener. The judge gets their own intro.
4. **Follow up:** run `/kym run step:followup` ("Did you two get a chance to talk?"). Tap an answer, then run `/kym run step:outcomes`. These steps look 30 days ahead, so you don't wait the usual 4–7 days.
5. **Speakers:** run `/kym speakers topic:<a topic the judges pick>`. Each name comes with evidence links, the basis for London, and what's unverified.
6. **Reviver:**
   - Run `/kym revive` and click **Post it** on Hannah's Postgres question.
   - The nudge lands in the real #help thread and pings whoever plays Emily; if nobody does, it names her.
   - `/kym log` then shows what failed.
7. **A month later:** run `uv run kym simulate`. It shows coverage, zero repeat pairs, a member who joined mid-month entering the cycle, the network before and after, and the failures.

## Known limits (say these out loud)

- **Luma past events:** `calendar/get-items?period=past` returns HTTP 429 without a login. The crawler uses upcoming events (London discover, and calendars) plus any event URLs you pass: `kym crawl-luma https://lu.ma/<event> ...`.
- **Devpost:** search results carry no location. Profiles are scraped only for top candidates, so few Devpost people qualify as London-based.
- **Hacker News:** location is free text in the profile, so it's a weak signal.
- **Embeddings alone match poorly:** bge-small rates any AI-flavoured text as similar. That's why matching uses tags, keyword relevance and the LLM judge.
- **Small communities run out of good pairs.** The simulation shows unmatched members rising by week four. Re-introducing pairs after N months isn't implemented.
- **Synthetic demo members are never matched** unless `KYM_INCLUDE_SYNTHETIC=1`, which is for demo servers only. Even then nothing is ever sent to them.
- **Seeded chat shows today's date in Discord.** Discord can't backdate messages; the bot uses the dataset's timestamps.
- **Lead-provided data:** imported exports such as Luma subscriber lists become *prospects* and are never matched or enriched until those people opt in. `data/` is gitignored; don't commit lead data.
