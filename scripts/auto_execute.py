"""Inject constrained Overmind country events into Stellaris.

The bridge writes one ``overmind_directive_<country_id>.command`` file per AI
directive. Each file must contain exactly one allowlisted ``event overmind.*``
command. This script rejects arbitrary console text and never switches player
control or executes direct build/resource effects.

Usage:
    python scripts/auto_execute.py

    # With custom Stellaris data directory
    python scripts/auto_execute.py --stellaris-dir "C:/Users/.../Paradox Interactive/Stellaris"

Requirements:
    - Stellaris must be running (non-Ironman, non-multiplayer)
    - The game console must be accessible (` key)
    - Windows only (uses ctypes for window activation)

Note: This is optional. AI-mode directives remain pending until this injector
or another compatible transport is running.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import logging
import time
from pathlib import Path

from engine.bridge import is_ai_event_command

log = logging.getLogger(__name__)

# Windows API constants
SW_RESTORE = 9
KEYEVENTF_KEYUP = 0x0002
VK_RETURN = 0x0D
VK_OEM_3 = 0xC0  # backtick/tilde key (console toggle)


def find_stellaris_window() -> int:
    """Find the Stellaris game window handle."""
    user32 = ctypes.windll.user32

    hwnd = user32.FindWindowW(None, "Stellaris")
    if hwnd:
        return hwnd

    # Try alternate titles
    for title in ("Stellaris ", "stellaris"):
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd

    return 0


def activate_window(hwnd: int) -> bool:
    """Bring Stellaris window to foreground."""
    user32 = ctypes.windll.user32

    # Restore if minimized
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    # Bring to front
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)

    return user32.GetForegroundWindow() == hwnd


def send_key(vk: int, delay: float = 0.05) -> None:
    """Send a single key press + release."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(delay)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(delay)


def send_text(text: str, delay: float = 0.03) -> None:
    """Type a string using SendInput."""
    user32 = ctypes.windll.user32
    for char in text:
        # Use VkKeyScan to get virtual key for each character
        vk_result = user32.VkKeyScanW(ord(char))
        vk = vk_result & 0xFF
        shift = (vk_result >> 8) & 1

        if shift:
            user32.keybd_event(0x10, 0, 0, 0)  # Shift down

        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(delay)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

        if shift:
            user32.keybd_event(0x10, 0, KEYEVENTF_KEYUP, 0)

        time.sleep(delay)


def execute_console_command(hwnd: int, command: str) -> bool:
    """Open Stellaris console, type command, execute, close console."""
    if not activate_window(hwnd):
        log.warning("Could not activate Stellaris window")
        return False

    time.sleep(0.2)

    # Open console (backtick key)
    send_key(VK_OEM_3, delay=0.1)
    time.sleep(0.3)

    # Type command
    send_text(command)
    time.sleep(0.1)

    # Press Enter
    send_key(VK_RETURN, delay=0.1)
    time.sleep(0.3)

    # Close console (backtick key again)
    send_key(VK_OEM_3, delay=0.1)

    return True


def watch_and_execute(
    stellaris_dir: Path,
    poll_interval: float = 2.0,
) -> None:
    """Inject each pending, allowlisted AI directive event exactly once."""
    log.info("Auto-execute watching: %s", stellaris_dir / "overmind_directive_*.command")
    log.info("Make sure Stellaris is running and the console is accessible")
    log.info("Press Ctrl+C to stop")

    while True:
        try:
            for command_path in sorted(stellaris_dir.glob("overmind_directive_*.command")):
                command = command_path.read_text(encoding="utf-8").strip()
                if not is_ai_event_command(command):
                    rejected_path = command_path.with_suffix(".rejected")
                    command_path.replace(rejected_path)
                    log.error("Rejected unsafe directive command: %s", command_path.name)
                    continue

                hwnd = find_stellaris_window()
                if hwnd == 0:
                    log.warning("Stellaris window not found — is the game running?")
                    break

                if execute_console_command(hwnd, command):
                    command_path.unlink()
                    log.info("Injected directive: %s", command)
                else:
                    log.warning("Failed to inject directive: %s", command)

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            log.info("Shutting down auto-execute")
            break
        except Exception:
            log.exception("Error in auto-execute loop")
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-execute Overmind directives in Stellaris console",
    )
    parser.add_argument(
        "--stellaris-dir", type=Path,
        default=None,
        help="Stellaris user data directory (auto-detected if not set)",
    )
    parser.add_argument(
        "--poll", type=float, default=2.0,
        help="Polling interval in seconds",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [auto-exec] %(message)s",
        datefmt="%H:%M:%S",
    )

    stellaris_dir = args.stellaris_dir
    if stellaris_dir is None:
        # Auto-detect common Stellaris user data paths
        for candidate in [
            Path.home() / "OneDrive/Documents/Paradox Interactive/Stellaris",
            Path.home() / "Documents/Paradox Interactive/Stellaris",
        ]:
            if candidate.exists():
                stellaris_dir = candidate
                break
        if stellaris_dir is None:
            log.error("Cannot find Stellaris directory. Use --stellaris-dir.")
            raise SystemExit(1)

    if not stellaris_dir.exists():
        log.error("Stellaris directory not found: %s", stellaris_dir)
        raise SystemExit(1)

    watch_and_execute(stellaris_dir, args.poll)


if __name__ == "__main__":
    main()
