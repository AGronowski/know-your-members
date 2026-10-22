"""Canonical topic tags.

Embeddings alone match short "wants"/"offers" phrases poorly (bge-small rates "a frontend cofounder" closer to
"fundraising advice" than to "React and design systems"). Mapping phrases onto a shared tag vocabulary gives the
matcher a precise, explainable signal; embeddings and the LLM judge add nuance on top.
"""

from __future__ import annotations

import re

TAGS: dict[str, list[str]] = {
    # engineering
    "frontend": ["frontend", "front-end", "front end", "react", "next.js", "nextjs", "vue", "svelte", "css", "web ui", "web app"],
    "backend": ["backend", "back-end", "api design", "apis", "django", "fastapi", "rails", "node.js", "nodejs", "microservices"],
    "mobile": ["mobile", "ios", "android", "swift", "kotlin", "react native", "flutter"],
    "infra": ["infra", "infrastructure", "devops", "kubernetes", "k8s", "terraform", "aws", "gcp", "azure", "cloud", "sre", "deploy"],
    "data-eng": ["data engineering", "data pipelines", "etl", "dbt", "airflow", "spark", "warehouse", "snowflake", "bigquery", "postgres", "databases", "sql"],
    "security": ["security", "appsec", "pentest", "auth", "authentication", "privacy", "cryptography"],
    "open-source": ["open source", "open-source", "oss", "maintainer"],
    "hardware": ["hardware", "embedded", "firmware", "pcb", "iot", "electronics", "raspberry pi", "arduino"],
    "robotics": ["robotics", "robot", "ros", "drones", "autonomy"],
    # AI
    "llm-apps": ["llm", "llms", "genai", "generative ai", "chatbot", "gpt", "claude", "grok", "prompt", "copilot", "ai app"],
    "agents": ["agent", "agents", "agentic", "tool use", "mcp", "autonomous agents", "multi-agent"],
    "rag": ["rag", "retrieval", "retrieval-augmented", "vector database", "vector db", "embeddings", "semantic search", "knowledge graph"],
    "evals": ["eval", "evals", "evaluation", "evaluating", "benchmark", "llm testing", "red teaming", "guardrails", "hallucination"],
    "ml-research": ["ml research", "machine learning", "deep learning", "model training", "fine-tuning", "fine tuning", "pytorch", "transformers", "research papers"],
    "inference": ["inference", "gpu", "gpus", "cuda", "serving", "latency", "quantization", "vllm", "model serving"],
    "voice": ["voice", "speech", "tts", "text-to-speech", "stt", "speech-to-text", "audio", "whisper", "realtime audio"],
    "computer-vision": ["computer vision", "image recognition", "image models", "video understanding", "ocr", "object detection"],
    # product & company building
    "product": ["product", "product management", "pm", "roadmap", "user research", "discovery"],
    "design": ["design", "ux", "ui/ux", "figma", "design systems", "product design", "branding", "user experience"],
    "growth": ["growth", "marketing", "seo", "content marketing", "go-to-market", "gtm", "launch", "distribution", "audience"],
    "sales": ["sales", "b2b sales", "enterprise sales", "selling", "partnerships", "business development", "bizdev"],
    "fundraising": ["fundraising", "fundraise", "raise", "raising", "investors", "investor", "angel", "angels", "vc", "venture", "pre-seed", "seed round", "pitch deck"],
    "hiring": ["hiring", "recruiting", "recruit", "first hire", "first engineer", "team building"],
    "jobs": ["job", "jobs", "new role", "job hunting", "looking for work", "career", "career change"],
    "cofounder": ["cofounder", "co-founder", "co founder", "founding partner", "founding team"],
    "mentorship": ["mentor", "mentoring", "mentorship", "advice", "advisor", "coaching", "office hours"],
    "feedback": ["feedback", "user feedback", "beta testers", "beta users", "usability", "review my", "roast"],
    "customers": ["customers", "first customers", "users", "enterprise pilots", "paid pilots", "design partners", "early adopters"],
    "legal": ["legal", "contracts", "intellectual property", "incorporation", "gdpr", "compliance", "regulation"],
    "finance": ["finance", "accounting", "bookkeeping", "r&d tax", "grants", "pricing", "unit economics"],
    "community": ["community", "meetup", "meetups", "organising", "organizing", "event organising", "hackathons", "devrel",
                  "developer relations"],
    "writing": ["writing", "blogging", "newsletter", "technical writing", "docs", "documentation"],
    "speaking": ["speaking", "talks", "public speaking", "conference talk", "demo day"],
    # domains
    "fintech": ["fintech", "payments", "banking", "open banking", "trading", "lending", "insurtech", "insurance"],
    "health": ["health", "healthtech", "medical", "clinical", "nhs", "biotech", "mental health", "wellbeing"],
    "climate": ["climate", "climatetech", "energy", "carbon", "sustainability", "net zero", "solar", "battery"],
    "edtech": ["edtech", "education", "students", "teaching", "tutoring"],
    "web3": ["web3", "crypto", "blockchain", "ethereum", "solana", "defi", "smart contracts"],
    "gaming": ["gaming", "games", "game dev", "unity", "unreal", "game"],
    "creative-tools": ["creative tools", "music", "generative art", "video editing", "creators"],
    "devtools": ["devtools", "developer tools", "developer experience", "dx", "cli", "sdk", "ide"],
    "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace", "shopify", "d2c"],
    "legaltech": ["legaltech", "legal tech", "law firms", "lawyers"],
    "proptech": ["proptech", "real estate", "housing"],
    "govtech": ["govtech", "public sector", "civic tech", "government"],
    "nocode": ["no-code", "nocode", "low-code", "lowcode", "bubble", "zapier", "automation", "workflow automation"],
}

# Phrases that contain a tag's keyword but mean something else ("design partners" are customers, not design).
EXCLUDE: dict[str, list[str]] = {
    "design": ["design partner", "curriculum design", "system design", "api design", "chip design"],
    "product": ["product feedback from engineering", "product analytics"],
}

_PATTERNS: dict[str, re.Pattern] = {
    tag: re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(s) for s in sorted(syns + [tag.replace("-", " ")], key=len, reverse=True)) + r")(?![a-z0-9])")
    for tag, syns in TAGS.items()
}


def tags_for(text: str) -> set[str]:
    """Keyword tagging. The LLM extractor tags more cleverly; this is the deterministic floor."""
    lowered = text.lower()
    found = set()
    for tag, pattern in _PATTERNS.items():
        candidate = lowered
        for phrase in EXCLUDE.get(tag, []):
            candidate = candidate.replace(phrase, " ")
        if pattern.search(candidate):
            found.add(tag)
    return found


def clean_tags(tags) -> list[str]:
    return sorted({t for t in (tags or []) if t in TAGS})
