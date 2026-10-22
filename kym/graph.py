"""The member graph: who already talks to whom, communities, and connection-density metrics for the lead."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import Interaction, Intro, Member, Message


def pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def interaction_weights(session: Session) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for row in session.scalars(select(Interaction)):
        weights[pair(row.member_a_id, row.member_b_id)] += row.weight
    return dict(weights)


def build_graph(session: Session, include_intros: bool = True) -> nx.Graph:
    g = nx.Graph()
    for m in session.scalars(select(Member).where(Member.status == "active")):
        g.add_node(m.id, name=m.name)
    for (a, b), w in interaction_weights(session).items():
        if a in g and b in g:
            g.add_edge(a, b, weight=w, kind="chat")
    if include_intros:
        for intro in session.scalars(select(Intro).where(Intro.status.in_(("connected", "completed")))):
            a, b = intro.member_a_id, intro.member_b_id
            if a in g and b in g:
                prev = g.get_edge_data(a, b, {}).get("weight", 0.0)
                g.add_edge(a, b, weight=prev + 1.0, kind="intro")
    return g


def communities(g: nx.Graph, seed: int = 7) -> dict[int, int]:
    """Louvain communities over the chat graph; empty when there are no edges to learn from."""
    if g.number_of_edges() == 0:
        return {}
    labels = {}
    for i, group in enumerate(nx.community.louvain_communities(g, weight="weight", seed=seed)):
        for node in group:
            labels[node] = i
    return labels


def metrics(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    if n == 0:
        return {"members": 0}
    isolates = list(nx.isolates(g))
    components = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    degree = nx.degree_centrality(g) if n > 1 else {}
    top = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:5]
    intro_edges = sum(1 for *_, d in g.edges(data=True) if d.get("kind") == "intro")
    return {
        "members": n,
        "connections": g.number_of_edges(),
        "intro_connections": intro_edges,
        "density": round(nx.density(g), 4),
        "isolated_members": len(isolates),
        "share_connected": round(1 - len(isolates) / n, 3),
        "components": len(components),
        "largest_component_share": round(components[0] / n, 3) if components else 0,
        "top_connectors": [{"member_id": node, "name": g.nodes[node].get("name"), "degree": round(c, 3)} for node, c in top],
    }


def rebuild_interactions(session: Session) -> int:
    """Recompute chat interactions from mirrored messages: replies, mentions and small-thread co-participation."""
    by_discord = {m.discord_id: m.id for m in session.scalars(select(Member).where(Member.discord_id.is_not(None)))}
    weights: dict[tuple[int, int], float] = defaultdict(float)
    last_at: dict[tuple[int, int], object] = {}
    messages = session.scalars(select(Message).where(Message.author_is_bot.is_(False))).all()
    authors = {m.id: m.author_id for m in messages}
    threads: dict[int, set[int]] = defaultdict(set)

    def bump(u: int, v: int, w: float, when) -> None:
        a, b = by_discord.get(u), by_discord.get(v)
        if a and b and a != b:
            key = pair(a, b)
            weights[key] += w
            if key not in last_at or when > last_at[key]:
                last_at[key] = when

    for msg in messages:
        if msg.reply_to_id and msg.reply_to_id in authors:
            bump(msg.author_id, authors[msg.reply_to_id], 1.0, msg.created_at)
        for uid in msg.mention_ids or []:
            bump(msg.author_id, int(uid), 0.5, msg.created_at)
        if msg.thread_id:
            threads[msg.thread_id].add(msg.author_id)
    for participants in threads.values():
        if 2 <= len(participants) <= 15:
            for u, v in combinations(sorted(participants), 2):
                a, b = by_discord.get(u), by_discord.get(v)
                if a and b and a != b:
                    weights[pair(a, b)] += 0.3

    session.execute(delete(Interaction).where(Interaction.source == "chat"))
    for (a, b), w in weights.items():
        session.add(Interaction(member_a_id=a, member_b_id=b, weight=round(w, 2), last_at=last_at.get((a, b)), source="chat"))
    return len(weights)
