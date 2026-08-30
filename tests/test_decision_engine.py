"""Tests for decision_engine — Stellaris 4.4.6."""

from __future__ import annotations

import pytest

from engine.decision_engine import (
    ALLOWED_ACTIONS,
    Directive,
    build_prompt,
    build_state_prompt,
    build_static_prompt,
    cached_prompt,
    decide,
    parse_llm_response,
)
from engine.personality_shards import build_personality
from engine.ruleset_generator import generate_ruleset


class TestParseResponse:

    def test_valid_response(self) -> None:
        raw = "ACTION: BUILD_FLEET\nTARGET: Sol\nREASON: Militarist ethic demands fleet."
        d = parse_llm_response(raw)
        assert d.action == "BUILD_FLEET"
        assert d.target == "Sol"
        assert "Militarist" in d.reason

    def test_none_target(self) -> None:
        raw = "ACTION: CONSOLIDATE\nTARGET: NONE\nREASON: Stability needed."
        d = parse_llm_response(raw)
        assert d.target is None

    def test_invalid_action_raises(self) -> None:
        raw = "ACTION: NUKE_PLANET\nTARGET: NONE\nREASON: Test."
        with pytest.raises(ValueError, match="Invalid action"):
            parse_llm_response(raw)

    def test_case_insensitive_parsing(self) -> None:
        raw = "action: expand\ntarget: none\nreason: Need space."
        d = parse_llm_response(raw)
        assert d.action == "EXPAND"

    def test_extra_whitespace(self) -> None:
        raw = "  ACTION:  BUILD_FLEET  \n  TARGET:  Sol  \n  REASON:  Fleet needed.  "
        d = parse_llm_response(raw)
        assert d.action == "BUILD_FLEET"
        assert d.target == "Sol"


class TestBuildPrompt:

    def test_prompt_contains_meta_rules(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        prompt = build_prompt(rs, p, early_game_state, None)
        assert "VERSIONED META (4.4.6)" in prompt
        assert "specialize colonies to their designation" in prompt
        assert "Disruptors are DEAD" not in prompt
        assert "ALLOWED ACTIONS" in prompt
        assert "4.4.6" in prompt

    def test_prompt_contains_fleet_meta(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        prompt = build_prompt(rs, p, early_game_state, None)
        assert "FLEET" in prompt
        assert "corvette" in prompt.lower()

    def test_prompt_contains_phase(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        prompt = build_prompt(rs, p, early_game_state, None)
        assert "early" in prompt.lower()

    def test_prompt_contains_event(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        prompt = build_prompt(rs, p, early_game_state, "WAR_DECLARED")
        assert "WAR_DECLARED" in prompt


class TestPromptSplit:

    def test_full_prompt_is_static_plus_state(
        self, une_empire: dict, early_game_state: dict,
    ) -> None:
        """build_prompt must equal build_static_prompt + build_state_prompt,
        so the split never changes what the LLM sees on a full send."""
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        full = build_prompt(rs, p, early_game_state, "WAR_DECLARED")
        static = build_static_prompt(rs, p, early_game_state)
        state = build_state_prompt(early_game_state, "WAR_DECLARED")
        assert full == static + state
        assert "CURRENT STATE:" in state
        assert "WAR_DECLARED" in state
        assert "CURRENT STATE:" not in static
        assert "EMPIRE RULESET" in static
        assert "PERSONALITY PROFILE" in static

    def test_state_prompt_is_much_smaller_than_full(
        self, une_empire: dict, early_game_state: dict,
    ) -> None:
        """The whole point of the split: the per-tick delta is small."""
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        full = build_prompt(rs, p, early_game_state, None)
        state = build_state_prompt(early_game_state, None)
        assert len(state) < len(full) * 0.5

    def test_cached_prompt_sends_static_once(
        self, une_empire: dict, early_game_state: dict,
    ) -> None:
        """cached_prompt sends the static block on the first call, then only
        the state delta while the phase and ruleset are unchanged."""
        from engine.prompt_cache import PromptCache
        from engine.decision_engine import cached_prompt

        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        cache = PromptCache()
        sent: dict[str, str] = {}

        # Same phase (early), different state: full then delta
        later_state = dict(early_game_state, year=2230, month=1)
        later_state["economy"] = dict(early_game_state["economy"], alloys=90)
        p1 = cached_prompt(cache, "single", rs, p, early_game_state, None, sent)
        p2 = cached_prompt(cache, "single", rs, p, later_state, None, sent)
        assert "EMPIRE RULESET" in p1
        assert "CURRENT STATE:" in p2
        assert "EMPIRE RULESET" not in p2
        assert "PERSONALITY PROFILE" not in p2
        assert cache.stats["hits"] == 1

    def test_cached_prompt_resends_on_phase_change(
        self, une_empire: dict, early_game_state: dict,
    ) -> None:
        """A game-phase transition must re-send the static block."""
        from engine.prompt_cache import PromptCache
        from engine.decision_engine import cached_prompt

        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        cache = PromptCache()
        sent: dict[str, str] = {}

        early = dict(early_game_state, year=2210)  # early
        late = dict(early_game_state, year=2400)   # late
        p1 = cached_prompt(cache, "single", rs, p, early, None, sent)
        p2 = cached_prompt(cache, "single", rs, p, late, None, sent)
        assert "EMPIRE RULESET" in p1
        assert "EMPIRE RULESET" in p2  # re-sent on phase change
        assert cache.stats["misses"] == 2

    def test_cached_prompt_resends_on_ruleset_change(
        self, une_empire: dict, early_game_state: dict,
    ) -> None:
        """A ruleset reform (new ethics/civics) must re-send the static block."""
        from engine.prompt_cache import PromptCache
        from engine.decision_engine import cached_prompt

        rs1 = generate_ruleset(**une_empire)
        rs2 = generate_ruleset(**{**une_empire, "ethics": ["Fanatic Militarist"]})
        p = build_personality(**une_empire)
        cache = PromptCache()
        sent: dict[str, str] = {}

        p1 = cached_prompt(cache, "single", rs1, p, early_game_state, None, sent)
        p2 = cached_prompt(cache, "single", rs2, p, early_game_state, None, sent)
        assert "EMPIRE RULESET" in p1
        assert "EMPIRE RULESET" in p2
        assert cache.stats["misses"] == 2


class TestDecide:

    def test_stub_returns_consolidate(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        d = decide(rs, early_game_state)
        assert d.action == "CONSOLIDATE"

    def test_custom_llm_callable(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)

        def fake_llm(prompt: str) -> str:
            return "ACTION: BUILD_FLEET\nTARGET: Sol\nREASON: Militarist ethic base."

        d = decide(rs, early_game_state, llm_callable=fake_llm)
        assert d.action == "BUILD_FLEET"
        assert d.target == "Sol"

    def test_personality_passed(self, une_empire: dict, early_game_state: dict) -> None:
        rs = generate_ruleset(**une_empire)
        p = build_personality(**une_empire)
        d = decide(rs, early_game_state, personality=p)
        assert isinstance(d, Directive)

    def test_to_dict(self) -> None:
        d = Directive(action="EXPAND", target="Alpha Centauri", reason="Expansion drive.")
        dd = d.to_dict()
        assert dd["action"] == "EXPAND"
        assert dd["target"] == "Alpha Centauri"
