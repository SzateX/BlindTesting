from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Optional

import pyautogui

from .sounddevice_adapter import SounddeviceAdapter, SounddeviceConfig
from .whisper_adapter import WhisperAdapter, WhisperAdapterConfig


@dataclass
class BlindTestConfig:
    """Runtime settings for keyboard/listening loops and text matching."""

    click_delay_sec: float = 0.2
    max_clicks: int = 20
    match_case: bool = False
    match_mode: str = "contains"  # contains | regex | fuzzy
    fuzzy_threshold: float = 0.8  # 0..1


class BlindTestInterface:
    """High-level facade for desktop actions and speech-driven assertions."""

    def __init__(
        self,
        whisper_config: Optional[WhisperAdapterConfig] = None,
        sound_config: Optional[SounddeviceConfig] = None,
        test_config: Optional[BlindTestConfig] = None,
    ) -> None:
        """Create adapters and bind runtime matching configuration."""
        self.whisper = WhisperAdapter(whisper_config)
        self.sound = SounddeviceAdapter(sound_config)
        self.config = test_config or BlindTestConfig()

    def press(self, key: str) -> None:
        """Press a single keyboard key."""
        pyautogui.press(key)

    def hotkey(self, *keys: str) -> None:
        """Press a keyboard shortcut (keys pressed together)."""
        pyautogui.hotkey(*keys)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type text with optional per-character delay."""
        pyautogui.write(text, interval=interval)

    def listen_once(self, duration_sec: Optional[float] = None) -> str:
        """Capture one audio chunk and return its transcription."""
        audio, sr = (
            self.sound.record_for_duration(duration_sec)
            if duration_sec is not None
            else self.sound.record_until_silence()
        )
        if audio.size == 0:
            return ""
        return self.whisper.transcribe((audio, sr))

    def listen_for(self, duration_sec: float) -> str:
        """Capture audio for a fixed duration and transcribe it."""
        return self.listen_once(duration_sec=duration_sec)

    def repeat_press_until_heard(
        self,
        key: str,
        expected: Optional[str] = None,
        max_clicks: Optional[int] = None,
        delay_sec: Optional[float] = None,
        listen_duration_sec: Optional[float] = None,
        match_mode: Optional[str] = None,
        match_case: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
    ) -> str:
        """
        Press a key repeatedly until expected text is heard (or until max clicks).
        Returns the last transcription (empty string if nothing heard).
        """
        clicks = 0
        max_clicks = max_clicks if max_clicks is not None else self.config.max_clicks
        delay_sec = delay_sec if delay_sec is not None else self.config.click_delay_sec

        last_text = ""
        while clicks < max_clicks:
            self.press(key)
            last_text = self.listen_once(duration_sec=listen_duration_sec)
            if expected is None:
                return last_text
            if self._match_text(
                last_text,
                expected,
                match_mode=match_mode,
                match_case=match_case,
                fuzzy_threshold=fuzzy_threshold,
            ):
                return last_text
            clicks += 1
            time.sleep(delay_sec)
        return last_text

    def repeat_hotkey_until_heard(
        self,
        keys: tuple[str, ...],
        expected: Optional[str] = None,
        max_clicks: Optional[int] = None,
        delay_sec: Optional[float] = None,
        listen_duration_sec: Optional[float] = None,
        match_mode: Optional[str] = None,
        match_case: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
    ) -> str:
        """
        Press a hotkey repeatedly until expected text is heard (or until max clicks).
        Returns the last transcription (empty string if nothing heard).
        """
        clicks = 0
        max_clicks = max_clicks if max_clicks is not None else self.config.max_clicks
        delay_sec = delay_sec if delay_sec is not None else self.config.click_delay_sec

        last_text = ""
        while clicks < max_clicks:
            self.hotkey(*keys)
            last_text = self.listen_once(duration_sec=listen_duration_sec)
            if expected is None:
                return last_text
            if self._match_text(
                last_text,
                expected,
                match_mode=match_mode,
                match_case=match_case,
                fuzzy_threshold=fuzzy_threshold,
            ):
                return last_text
            clicks += 1
            time.sleep(delay_sec)
        return last_text

    def wait_until_heard(
        self,
        expected: str,
        timeout_sec: float = 10.0,
        poll_delay_sec: float = 0.2,
        listen_duration_sec: Optional[float] = None,
        match_mode: Optional[str] = None,
        match_case: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
    ) -> str:
        """
        Listen repeatedly until expected text is heard or timeout expires.
        Returns the matching transcription, or the last heard transcription.
        """
        deadline = time.monotonic() + timeout_sec
        last_text = ""
        while time.monotonic() < deadline:
            last_text = self.listen_once(duration_sec=listen_duration_sec)
            if self._match_text(
                last_text,
                expected,
                match_mode=match_mode,
                match_case=match_case,
                fuzzy_threshold=fuzzy_threshold,
            ):
                return last_text
            time.sleep(poll_delay_sec)
        return last_text

    def _match_text(
        self,
        text: str,
        expected: str,
        match_mode: Optional[str] = None,
        match_case: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
    ) -> bool:
        """Compare recognized text with expected text using selected match mode."""
        if match_case is None:
            match_case = self.config.match_case
        if not match_case:
            text = text.lower()
            expected = expected.lower()

        mode = match_mode or self.config.match_mode
        if mode == "regex":
            return re.search(expected, text) is not None
        if mode == "fuzzy":
            threshold = (
                fuzzy_threshold
                if fuzzy_threshold is not None
                else self.config.fuzzy_threshold
            )
            ratio = SequenceMatcher(None, expected, text).ratio()
            return ratio >= threshold
        # default: contains
        return expected in text
    
