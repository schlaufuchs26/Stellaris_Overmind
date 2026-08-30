"""Tests for game_loop — Stellaris 4.4.6."""

from __future__ import annotations

import pytest

from engine.bridge import BridgeConfig
from engine.game_loop import EmpireConfig, GameLoopController, LoopStats
from engine.llm_provider import LLMResponse, StubProvider


@pytest.fixture
def empire() -> EmpireConfig:
    return EmpireConfig(
        ethics=["Militarist", "Materialist"],
        civics=["Technocracy"],
        traits=["Intelligent"],
        origin="Prosperous Unification",
        government="Oligarchy",
    )


@pytest.fixture
def controller(empire: EmpireConfig, tmp_path) -> GameLoopController:
    from engine.recorder import GameRecorder

    config = BridgeConfig(bridge_dir=tmp_path / "bridge")
    recorder = GameRecorder(replay_dir=tmp_path / "replays")
    return GameLoopController(
        empire=empire,
        provider=StubProvider(),
        bridge_config=config,
        recorder=recorder,
    )


class TestControllerInit:

    def test_ruleset_generated(self, controller: GameLoopController) -> None:
        assert controller._ruleset["version"] == "4.4.6"

    def test_personality_generated(self, controller: GameLoopController) -> None:
        assert "tech_focus" in controller._personality

    def test_stats_initialized(self, controller: GameLoopController) -> None:
        assert controller.stats.decisions_made == 0


class TestTickOnce:

    def test_stub_produces_consolidate(
        self, controller: GameLoopController, early_game_state: dict,
    ) -> None:
        d = controller.tick_once(early_game_state)
        assert d is not None
        assert d.action == "CONSOLIDATE"

    def test_increments_stats(
        self, controller: GameLoopController, early_game_state: dict,
    ) -> None:
        controller.tick_once(early_game_state)
        assert controller.stats.decisions_made == 1

    def test_records_decision(
        self, controller: GameLoopController, early_game_state: dict,
    ) -> None:
        # tick_once doesn't record (only _tick does), but stats update
        controller.tick_once(early_game_state)
        assert controller.stats.last_action == "CONSOLIDATE"


class RecordingProvider(StubProvider):
    """StubProvider that records every prompt it is asked to complete."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return super().complete(prompt)


class TestPromptCaching:

    def test_static_sent_once_then_state_delta(
        self, empire: EmpireConfig, tmp_path,
    ) -> None:
        """Two ticks in the same phase: the first prompt carries the full
        static context, the second carries only the state delta (ticket
        #792)."""
        from engine.recorder import GameRecorder

        provider = RecordingProvider()
        config = BridgeConfig(bridge_dir=tmp_path / "bridge")
        recorder = GameRecorder(replay_dir=tmp_path / "replays")
        controller = GameLoopController(
            empire=empire,
            provider=provider,
            bridge_config=config,
            recorder=recorder,
        )

        state1 = {
            "version": "4.4.6", "year": 2210, "month": 3,
            "empire": {"ethics": ["Militarist", "Materialist"], "civics": ["Technocracy"]},
            "economy": {"energy": 100, "minerals": 200, "alloys": 30},
            "colonies": ["Earth"], "known_empires": [],
            "fleets": [{"name": "1st", "power": 1500}],
        }
        state2 = dict(state1, year=2211, month=6)
        state2["economy"] = {"energy": 120, "minerals": 210, "alloys": 45}
        state2["fleets"] = [{"name": "1st", "power": 1800}]

        assert controller.tick_once(state1) is not None
        assert controller.tick_once(state2) is not None

        assert len(provider.prompts) == 2
        assert "EMPIRE RULESET" in provider.prompts[0]
        assert "PERSONALITY PROFILE" in provider.prompts[0]
        assert "CURRENT STATE:" in provider.prompts[0]
        # Second tick: no static context, just the state delta
        assert "EMPIRE RULESET" not in provider.prompts[1]
        assert "PERSONALITY PROFILE" not in provider.prompts[1]
        assert "CURRENT STATE:" in provider.prompts[1]
        assert '"year": 2211' in provider.prompts[1]

    def test_static_resent_after_phase_change(
        self, empire: EmpireConfig, tmp_path,
    ) -> None:
        """Crossing into a new game phase re-sends the static context."""
        from engine.recorder import GameRecorder

        provider = RecordingProvider()
        config = BridgeConfig(bridge_dir=tmp_path / "bridge")
        recorder = GameRecorder(replay_dir=tmp_path / "replays")
        controller = GameLoopController(
            empire=empire,
            provider=provider,
            bridge_config=config,
            recorder=recorder,
        )

        state = {
            "version": "4.4.6", "year": 2210, "month": 3,
            "empire": {"ethics": ["Militarist", "Materialist"], "civics": ["Technocracy"]},
            "economy": {"energy": 100}, "colonies": ["Earth"],
            "known_empires": [], "fleets": [],
        }
        assert controller.tick_once(state) is not None
        late = dict(state, year=2400, month=1)  # late game
        assert controller.tick_once(late) is not None

        assert len(provider.prompts) == 2
        assert "EMPIRE RULESET" in provider.prompts[0]
        assert "EMPIRE RULESET" in provider.prompts[1]  # re-sent


class TestRulesetRefresh:

    def test_refresh_on_civic_change(
        self, controller: GameLoopController, early_game_state: dict,
    ) -> None:
        # Simulate government reform
        state = dict(early_game_state)
        state["empire"] = {
            "ethics": ["Fanatic Militarist"],
            "civics": ["Distinguished Admiralty"],
            "origin": "Prosperous Unification",
            "government": "Oligarchy",
        }
        controller._maybe_refresh_ruleset(state)
        assert controller._ruleset["base"].get("fire_rate_mult") == 0.20

    def test_no_refresh_when_unchanged(
        self, controller: GameLoopController, early_game_state: dict,
    ) -> None:
        state = dict(early_game_state)
        state["empire"] = {
            "ethics": ["Materialist", "Militarist"],
            "civics": ["Technocracy"],
            "origin": "Prosperous Unification",
            "government": "Oligarchy",
        }
        old_ruleset = controller._ruleset
        controller._maybe_refresh_ruleset(state)
        # Ruleset should be the same object (not regenerated)
        assert controller._ruleset is old_ruleset
