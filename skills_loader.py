"""Skill loader for batch analysis.

Reads markdown files from ./skills/ and picks the most relevant skill
for a cluster, based on the cluster's `situacao_processo` and tags.

Skill file format (markdown with simple key:value frontmatter):

    ---
    situacoes: Pronto para Sentença
    tags: execução fiscal, dívida ativa, CDA
    priority: 10
    ---

    # Skill body in markdown...

The selected skill body is injected into the LLM system prompt by the
batch analysis endpoint, so each process in the cluster receives the
specialized guidance for its type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"

# Minimum score for a skill to be considered a real match.
# A bare situacao match scores 10, so we require at least that.
MIN_MATCH_SCORE = 10


@dataclass
class Skill:
    name: str
    body: str
    situacoes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    priority: int = 0
    path: Optional[Path] = None


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse simple key:value frontmatter delimited by --- markers.

    Returns (metadata_dict, body_text). If no frontmatter is found, the
    metadata is empty and the body is the full text.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header = text[3:end].strip()
    body = text[end + 4:].lstrip()

    meta: dict = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in {"situacoes", "tags"}:
            meta[key] = [v.strip() for v in value.split(",") if v.strip()]
        elif key == "priority":
            try:
                meta[key] = int(value)
            except ValueError:
                meta[key] = 0
        elif key:
            meta[key] = value
    return meta, body


def load_skills(directory: Path = SKILLS_DIR) -> list[Skill]:
    """Load every *.md file under the skills directory."""
    if not directory.exists():
        logger.debug("Skills directory not found: %s", directory)
        return []
    skills: list[Skill] = []
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            skills.append(Skill(
                name=path.stem,
                body=body,
                situacoes=meta.get("situacoes", []),
                tags=[t.lower() for t in meta.get("tags", [])],
                priority=int(meta.get("priority", 0)),
                path=path,
            ))
        except Exception as exc:
            logger.warning("Failed to load skill %s: %s", path, exc)
    return skills


def select_skill(
    situacao: Optional[str],
    tags: Optional[list[str]] = None,
    skills: Optional[list[Skill]] = None,
) -> Optional[Skill]:
    """Pick the most relevant skill for a cluster.

    Scoring:
        - +10 if the cluster's situacao_processo matches a skill's situacoes
        - +3 per matched tag (fuzzy substring, either direction)
        - + skill.priority (tie-breaker)

    Returns None when no skill scores at least MIN_MATCH_SCORE
    (i.e. no situacao match). This keeps the existing behavior untouched
    for unmatched clusters.
    """
    if skills is None:
        skills = load_skills()
    if not skills:
        return None

    tags_lower = [t.lower() for t in (tags or []) if t]

    best: Optional[Skill] = None
    best_score = -1
    for skill in skills:
        # Count matched tags first (each cluster tag counts once at most).
        matched_tags = 0
        for cluster_tag in tags_lower:
            for skill_tag in skill.tags:
                if skill_tag and (skill_tag in cluster_tag or cluster_tag in skill_tag):
                    matched_tags += 1
                    break

        # Specialized skills (those that declare tags) require at least one
        # tag match. Without this gate, a high-priority specialized skill
        # could win on situacao alone — e.g. "tutela urgência" being picked
        # for a generic "mero expediente".
        if skill.tags and matched_tags == 0:
            continue

        score = 0
        if situacao and situacao in skill.situacoes:
            score += 10
        score += matched_tags * 3
        score += skill.priority

        if score > best_score:
            best = skill
            best_score = score

    if best and best_score >= MIN_MATCH_SCORE:
        return best
    return None


def format_skill_for_prompt(skill: Skill) -> str:
    """Wrap a skill body for injection into the LLM system message."""
    return (
        f"\n\n---\n🧠 **SKILL ATIVADA: {skill.name}**\n"
        f"O sistema selecionou esta skill com base no tipo de processo do cluster. "
        f"Aplique as diretrizes abaixo ao redigir a minuta:\n\n"
        f"{skill.body}\n---"
    )
