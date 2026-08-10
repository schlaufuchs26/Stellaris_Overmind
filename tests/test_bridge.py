"""Tests for bridge — Stellaris 4.4.6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.bridge import (
    AI_EVENT_IDS,
    BridgeConfig,
    BridgeReader,
    BridgeWriter,
    UnifiedBridge,
    build_ai_event_command,
    is_ai_event_command,
)


@pytest.fixture
def bridge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ai_bridge"
    d.mkdir()
    return d


@pytest.fixture
def bridge_config(bridge_dir: Path) -> BridgeConfig:
    return BridgeConfig(bridge_dir=bridge_dir, save_dir=Path(""))


class TestBridgeWriter:

    def test_does_not_expose_direct_console_execution(self) -> None:
        assert not hasattr(BridgeWriter, "write_console_commands")

    def test_write_directive(self, bridge_config: BridgeConfig) -> None:
        writer = BridgeWriter(bridge_config)
        writer.write_directive({"action": "EXPAND", "target": "Sol"})
        path = bridge_config.bridge_dir / "directive.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["action"] == "EXPAND"

    def test_write_is_atomic(self, bridge_config: BridgeConfig) -> None:
        """No .tmp file should remain after write."""
        writer = BridgeWriter(bridge_config)
        writer.write_directive({"action": "DEFEND"})
        tmp = bridge_config.bridge_dir / "directive.tmp"
        assert not tmp.exists()

    def test_clear_directive(self, bridge_config: BridgeConfig) -> None:
        writer = BridgeWriter(bridge_config)
        writer.write_directive({"action": "EXPAND"})
        writer.clear_directive()
        assert not (bridge_config.bridge_dir / "directive.json").exists()

    def test_clear_nonexistent_ok(self, bridge_config: BridgeConfig) -> None:
        writer = BridgeWriter(bridge_config)
        writer.clear_directive()  # should not raise

    def test_writes_targeted_ai_event_command(self, bridge_config: BridgeConfig) -> None:
        command_dir = bridge_config.bridge_dir / "commands"
        bridge_config.command_dir = command_dir
        writer = BridgeWriter(bridge_config)

        writer.write_directive_for(42, {"action": "EXPAND"})

        command_path = command_dir / "overmind_directive_42.command"
        assert command_path.read_text(encoding="utf-8") == "event overmind.101 42\n"
        assert not command_path.with_suffix(".tmp").exists()

    def test_rejects_invalid_ai_event_command(self, bridge_config: BridgeConfig) -> None:
        writer = BridgeWriter(bridge_config)

        with pytest.raises(ValueError, match="Unsupported AI directive action"):
            writer.write_directive_for(42, {"action": "INVALID"})

        assert not (bridge_config.bridge_dir / "directive_42.json").exists()


class TestAIEventCommands:

    def test_mod_declares_each_allowlisted_action_event(self) -> None:
        event_file = (
            Path(__file__).parent.parent
            / "mod/stellaris_overmind/events/overmind_events.txt"
        )
        content = event_file.read_text(encoding="utf-8")

        for action, event_id in AI_EVENT_IDS.items():
            assert f"id = overmind.{event_id}" in content
            assert f"overmind_action_{action.lower()} = yes" in content

    def test_builds_allowlisted_command(self) -> None:
        assert build_ai_event_command(3, "ESPIONAGE") == "event overmind.111 3"

    @pytest.mark.parametrize("command", [
        "event overmind.101 3",
        "event overmind.111 42",
    ])
    def test_recognizes_allowlisted_command(self, command: str) -> None:
        assert is_ai_event_command(command)

    @pytest.mark.parametrize("command", [
        "effect add_resource = { alloys = 100 }",
        "event overmind.101 0",
        "event overmind.112 3",
        "event overmind.101 3; play 3",
    ])
    def test_rejects_non_allowlisted_command(self, command: str) -> None:
        assert not is_ai_event_command(command)


class TestBridgeReader:

    def test_no_snapshot_returns_none(self, bridge_config: BridgeConfig) -> None:
        reader = BridgeReader(bridge_config)
        assert reader.read_snapshot() is None

    def test_read_snapshot(self, bridge_config: BridgeConfig) -> None:
        snap_path = bridge_config.bridge_dir / "state_snapshot.json"
        snap_path.write_text(json.dumps({"year": 2230, "month": 6}))
        reader = BridgeReader(bridge_config)
        data = reader.read_snapshot()
        assert data is not None
        assert data["year"] == 2230

    def test_no_double_read(self, bridge_config: BridgeConfig) -> None:
        snap_path = bridge_config.bridge_dir / "state_snapshot.json"
        snap_path.write_text(json.dumps({"year": 2230}))
        reader = BridgeReader(bridge_config)
        assert reader.read_snapshot() is not None
        assert reader.read_snapshot() is None  # same file, not re-read

    def test_read_ack(self, bridge_config: BridgeConfig) -> None:
        ack_path = bridge_config.bridge_dir / "ack.json"
        ack_path.write_text(json.dumps({"status": "ok"}))
        reader = BridgeReader(bridge_config)
        ack = reader.read_ack()
        assert ack is not None
        assert ack["status"] == "ok"

    def test_corrupt_json_returns_none(self, bridge_config: BridgeConfig) -> None:
        snap_path = bridge_config.bridge_dir / "state_snapshot.json"
        snap_path.write_text("{invalid json")
        reader = BridgeReader(bridge_config)
        assert reader.read_snapshot() is None


class TestUnifiedBridge:

    def test_json_mode_when_no_save_dir(self, bridge_config: BridgeConfig) -> None:
        config = BridgeConfig(
            save_dir=Path("/nonexistent_path_xyz"),
            bridge_dir=bridge_config.bridge_dir,
        )
        bridge = UnifiedBridge(config)
        assert bridge.mode == "json"

    def test_autosave_mode_when_save_dir_exists(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "save games"
        save_dir.mkdir()
        config = BridgeConfig(save_dir=save_dir, bridge_dir=tmp_path / "bridge")
        bridge = UnifiedBridge(config)
        assert bridge.mode == "autosave"
