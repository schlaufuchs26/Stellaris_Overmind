"""
Decision Engine — Stellaris 4.4.6 LLM AI Overhaul

Receives a ruleset + personality + known game state + triggering event,
queries the LLM, and produces exactly one validated macro action.

The prompt includes:
  - Composite ruleset (ethics/civics/traits/origin)
  - Personality profile (war willingness, trade focus, etc.)
  - Game-phase priorities (economy, fleet, expansion guidance)
  - Fleet composition meta (weapon verdicts, fleet templates)
  - Current game state (fog-of-war filtered)
  - Triggering event (if any)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from engine.llm_provider import LLMProvider
from engine.prompt_cache import PromptCache
from engine.meta_loader import load_meta
from engine.ruleset_generator import (
    ALLOWED_ACTIONS,
    get_espionage_phase_priority,
    get_fleet_template,
    get_phase_priorities,
    get_weapon_meta,
)
from engine.strategic_knowledge import (
    get_megastructure_guidance,
    get_policy_guidance,
    get_starbase_guidance,
    get_tech_priorities,
    get_tradition_guidance,
)

# Prompt budget — approximate token count limits
# Qwen2.5-Omni-3B context: 4096 tokens; 7B: 8192 tokens
# ~4 chars per token on average
MAX_PROMPT_CHARS = 12000  # ~3000 tokens, leaves room for response
MAX_COLONIES_IN_PROMPT = 10
MAX_EMPIRES_IN_PROMPT = 8
MAX_FLEETS_IN_PROMPT = 5


@dataclass
class Directive:
    """A single macro‑strategic directive produced by the LLM."""

    action: str
    target: str | None = None
    reason: str = ""
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target": self.target,
            "reason": self.reason,
            "parameters": self.parameters,
        }


def build_static_prompt(ruleset: dict, personality: dict, state: dict) -> str:
    """Build the static context block of the decision prompt.

    Everything that does not change between ticks for a given ruleset and
    game phase: lead-in constraints, versioned meta, allowed actions,
    compact ruleset, personality profile, phase priorities, and the
    tradition/policy/tech/starbase/fleet/weapon guidance. The changing
    game state is appended separately by build_state_prompt, so the static
    block can be sent once per phase and cached (ticket #792).
    """
    year = state.get("year", 2200)
    phase = get_phase_priorities(year)
    meta = load_meta(str(ruleset.get("version", "")))
    fleet_tmpl = get_fleet_template(year) if meta.get("fleet_templates") else None
    weapons = get_weapon_meta() if meta.get("weapon_verdicts") else []
    espionage_phase = get_espionage_phase_priority(year)

    compact_ruleset = _compact_ruleset(ruleset)

    ethics = state.get("empire", {}).get("ethics", [])
    tradition_guide = get_tradition_guidance(
        year, ethics=ethics, adopted=state.get("traditions", []),
    )
    policy_guide = get_policy_guidance(year, ethics)
    tech_guide = get_tech_priorities(year)
    sb_guide = get_starbase_guidance(year)
    mega_names = [m["name"] for m in get_megastructure_guidance(year)]

    # Lead-in trimmed to the constraints that are not duplicated elsewhere
    # (ticket #792): the role intro, "exactly ONE action", and the format
    # footer live in ALLOWED ACTIONS / the reply-format block below, so
    # restating them wastes tokens.
    sections = [
        "Stellaris 4.4.6 strategic advisor. Cite ruleset elements in your reason. "
        "Never use information the empire does not know (fog-of-war). Never "
        "reference mechanics that do not exist in Stellaris 4.4.6.",
        "",
        f"VERSIONED META ({meta.get('version', ruleset.get('version', '?'))}):",
        meta.get("meta_rules_domestic", "No curated domestic meta is available."),
        meta.get("meta_rules_military", "No curated military meta is available."),
        "",
        f"ALLOWED ACTIONS: {', '.join(ALLOWED_ACTIONS)}",
        "",
        "EMPIRE RULESET:",
        json.dumps(compact_ruleset, indent=2, default=str),
        "",
        "PERSONALITY PROFILE:",
        json.dumps(personality, indent=2, default=str),
        "",
        f"GAME PHASE: {phase['phase']} | FOCUS: {phase.get('economy_focus', '')}",
        "",
        "ESPIONAGE: "
        f"priority={espionage_phase.get('priority', 'low')} | "
        f"{espionage_phase.get('notes', '')}",
        "",
        f"TRADITIONS: recommended={tradition_guide.get('recommended_trees', [])}",
        f"POLICY RECOMMENDATIONS: {json.dumps(policy_guide.get('recommended', {}))}",
        f"TECH PRIORITIES: {json.dumps(tech_guide.get('meta_notes', []))}",
        f"STARBASE: {sb_guide.get('priority', '')} | {sb_guide.get('notes', '')}",
    ]

    if fleet_tmpl is not None:
        sections.extend([
            "",
            f"FLEET: {json.dumps(fleet_tmpl.composition)} | {fleet_tmpl.notes}",
        ])

    if weapons:
        sections.append(f"WEAPON META: {json.dumps(weapons)}")

    if state.get("empire", {}).get("is_nomadic"):
        nomad = meta.get("nomads", {})
        sections.append(
            f"NOMADS: {json.dumps(nomad.get('operational_rules', []))}"
        )

    if mega_names:
        sections.append(f"MEGASTRUCTURES: consider building {mega_names}")

    return "\n".join(sections)


def _compact_ruleset(ruleset: dict) -> dict:
    """Drop raw data tables from the ruleset, keeping computed values."""
    return {
        k: v for k, v in ruleset.items()
        if k in ("version", "base", "modifiers", "overrides", "government",
                 "meta_tier", "meta_strategy")
    }


def _ruleset_fingerprint(ruleset: dict) -> str:
    """Stable identity of the compact ruleset, so a mid-game reform
    (new ethics/civics → new ruleset) invalidates the static-prompt cache
    even when the version string is unchanged."""
    compact = _compact_ruleset(ruleset)
    raw = json.dumps(compact, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def build_state_prompt(state: dict, event: str | None) -> str:
    """Build the changing part of the decision prompt: the CURRENT STATE
    block (compacted game state), the triggering event, and the reply
    format footer. The footer rides with the state part so full prompts
    keep their historical layout (format instruction last) and state-only
    deltas stay self-describing for endpoints without conversation memory.
    """
    # Truncate state to fit prompt budget
    compact_state = _compact_state(state)

    parts = [
        "",
        "CURRENT STATE:",
        json.dumps(compact_state, indent=2),
    ]
    if event:
        parts.append(f"\nTRIGGERING EVENT: {event}")
    parts.append(
        "\nRespond in EXACTLY this format:\n"
        "ACTION: <one action from the allowed list>\n"
        "TARGET: <target or NONE>\n"
        "REASON: <must cite ruleset elements and meta rules>"
    )
    return "\n".join(parts)


def build_prompt(
    ruleset: dict,
    personality: dict,
    state: dict,
    event: str | None,
    *,
    static_prompt: str | None = None,
) -> str:
    """Construct the LLM prompt from ruleset, personality, state, and event.

    The prompt is structured to give the LLM maximum context while
    constraining output to exactly one action in the required format.
    Large game states are truncated to fit within the model's context window.

    The prompt splits into a static part (ruleset, personality, meta,
    phase guidance; see build_static_prompt) and the changing game state
    (build_state_prompt). Pass a cached static_prompt to skip rebuilding
    the static sections; the result then contains only the state part,
    which is all that changes between ticks (ticket #792).
    """
    if static_prompt is None:
        static_prompt = build_static_prompt(ruleset, personality, state)
    return static_prompt + build_state_prompt(state, event)


def cached_prompt(
    cache: PromptCache,
    cache_key: str,
    ruleset: dict,
    personality: dict,
    state: dict,
    event: str | None,
    sent_static: dict[str, str],
) -> str:
    """Build the prompt for one decision, sending the static context only
    when it is new (first decision for this key, game-phase change, or
    ruleset reform). Every other tick returns only the changing game
    state, so an endpoint with conversation memory (the fuchs Overmind
    bridge) sees the static context once and a small delta each turn
    (ticket #792).

    `sent_static` tracks which static block was already sent per cache
    key; the caller owns it (and clears it on cache invalidation) so the
    PromptCache itself stays a pure prefix cache.
    """
    year = state.get("year", 2200)
    phase = get_phase_priorities(year)["phase"]
    version = str(ruleset.get("version", ""))
    static = cache.get_or_build(
        cache_key, phase, version + ":" + _ruleset_fingerprint(ruleset),
        lambda: build_static_prompt(ruleset, personality, state),
    )
    dynamic = build_state_prompt(state, event)
    if sent_static.get(cache_key) == static:
        return dynamic
    sent_static[cache_key] = static
    return static + dynamic


def _compact_state(state: dict) -> dict:
    """Truncate game state to fit within prompt budget.

    Keeps the most strategically relevant data:
    - Full economy (with monthly net)
    - Top N fleets by power
    - Top N known empires (hostile first)
    - Colony details (first N with stats)
    - Technology summary
    - Active policies, edicts, traditions, wars
    """
    compact: dict = {
        "year": state.get("year"),
        "month": state.get("month"),
        "empire": state.get("empire", {}),
        "economy": state.get("economy", {}),
    }

    # Fleets: keep top N by power
    fleets = state.get("fleets", [])
    if isinstance(fleets, list):
        fleets_sorted = sorted(fleets, key=lambda f: f.get("power", 0), reverse=True)
        compact["fleets"] = fleets_sorted[:MAX_FLEETS_IN_PROMPT]
        if len(fleets) > MAX_FLEETS_IN_PROMPT:
            compact["total_fleets"] = len(fleets)
            compact["total_fleet_power"] = sum(f.get("power", 0) for f in fleets)

    # Colonies: count + first N
    colonies = state.get("colonies", [])
    if isinstance(colonies, list):
        compact["colony_count"] = len(colonies)
        compact["colonies"] = colonies[:MAX_COLONIES_IN_PROMPT]

    # Known empires: prioritize hostile, keep top N
    empires = state.get("known_empires", [])
    if isinstance(empires, list):
        # Sort: hostile first, then by name
        hostile = [e for e in empires if e.get("attitude") in ("hostile", "Hostile")]
        others = [e for e in empires if e.get("attitude") not in ("hostile", "Hostile")]
        sorted_empires = hostile + others
        compact["known_empires"] = sorted_empires[:MAX_EMPIRES_IN_PROMPT]
        if len(empires) > MAX_EMPIRES_IN_PROMPT:
            compact["total_known_empires"] = len(empires)

    # Technology: count + current research
    tech = state.get("technology", {})
    if isinstance(tech, dict):
        compact["tech_count"] = tech.get("count", 0)
        in_progress = tech.get("in_progress", {})
        if in_progress:
            compact["researching"] = in_progress

    # Traditions
    traditions = state.get("traditions", [])
    if isinstance(traditions, list) and traditions:
        compact["traditions"] = traditions

    # Ascension perks
    perks = state.get("ascension_perks", [])
    if isinstance(perks, list) and perks:
        compact["ascension_perks"] = perks

    # Active policies (compact: just policy→selected map)
    policies = state.get("policies", [])
    if isinstance(policies, list) and policies:
        compact["policies"] = {
            p["policy"]: p["selected"]
            for p in policies
            if isinstance(p, dict) and "policy" in p
        }

    # Edicts
    edicts = state.get("edicts", [])
    if isinstance(edicts, list) and edicts:
        compact["edicts"] = edicts

    # Wars
    wars = state.get("wars", [])
    if isinstance(wars, list) and wars:
        compact["wars"] = wars

    # Starbases (upgraded only — outposts excluded in save_reader)
    starbases = state.get("starbases", [])
    if isinstance(starbases, list) and starbases:
        compact["starbases"] = starbases

    # Leaders (compact: class + level only)
    leaders = state.get("leaders", [])
    if isinstance(leaders, list) and leaders:
        compact["leaders"] = [
            {"class": ld.get("class", ""), "level": ld.get("level", 0)}
            for ld in leaders
            if isinstance(ld, dict)
        ]

    # Naval capacity
    nav_cap = state.get("naval_capacity", {})
    if isinstance(nav_cap, dict) and nav_cap:
        compact["naval_capacity"] = nav_cap

    return compact


def parse_llm_response(raw: str) -> Directive:
    """Parse the LLM's structured response into a Directive."""
    action = ""
    target = None
    reason = ""

    for line in raw.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("ACTION:"):
            action = line.split(":", 1)[1].strip().upper()
        elif upper.startswith("TARGET:"):
            val = line.split(":", 1)[1].strip()
            target = None if val.upper() == "NONE" else val
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if action not in ALLOWED_ACTIONS:
        raise ValueError(
            f"Invalid action '{action}'. Must be one of {ALLOWED_ACTIONS}"
        )

    return Directive(action=action, target=target, reason=reason)


def decide(
    ruleset: dict,
    state: dict,
    event: str | None = None,
    *,
    personality: dict | None = None,
    llm_callable=None,
    provider: LLMProvider | None = None,
) -> Directive:
    """Run one decision cycle.

    Parameters
    ----------
    ruleset : dict
        Composite ruleset from ``ruleset_generator.generate_ruleset``.
    state : dict
        JSON‑serialisable game state snapshot (fog‑of‑war filtered).
    event : str | None
        Optional triggering event identifier.
    personality : dict | None
        Personality profile from ``personality_shards.build_personality``.
        If *None*, a neutral profile is used.
    llm_callable : callable | None
        Legacy: ``llm_callable(prompt: str) -> str``.
        Prefer *provider* instead.
    provider : LLMProvider | None
        An ``LLMProvider`` instance.  Takes precedence over *llm_callable*.
        If neither is given, a stub response is used.
    """
    if personality is None:
        personality = {
            "war_willingness": 0.5,
            "expansion_drive": 0.5,
            "tech_focus": 0.5,
            "unity_focus": 0.5,
            "diplomatic_openness": 0.5,
            "trade_focus": 0.3,
            "economic_style": "balanced",
            "risk_tolerance": 0.5,
            "ascension_preference": "any",
            "crisis_preparedness": 0.3,
            "fleet_doctrine": "balanced",
            "leader_weights": {},
        }

    prompt = build_prompt(ruleset, personality, state, event)

    if provider is not None:
        # Use the provider interface (preferred path)
        from engine.llm_provider import LLMProviderError

        try:
            response = provider.complete(prompt)
            raw_response = response.text
        except LLMProviderError:
            raw_response = (
                "ACTION: CONSOLIDATE\n"
                "TARGET: NONE\n"
                "REASON: LLM provider error; defaulting to safe posture."
            )
    elif llm_callable is not None:
        raw_response = llm_callable(prompt)
    else:
        raw_response = (
            "ACTION: CONSOLIDATE\n"
            "TARGET: NONE\n"
            "REASON: No LLM connected; defaulting to safe posture "
            "per 4.3 meta (stability is scarce, consolidate first)."
        )

    return parse_llm_response(raw_response)
