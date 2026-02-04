from __future__ import annotations

import logging
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import numpy as np

# SoundCard 0.4.x still calls numpy.fromstring in binary mode, which NumPy 2+
# removed. This shim keeps loopback recording working on newer NumPy versions.
_ORIG_FROMSTRING = np.fromstring


def _compat_fromstring(string, dtype=float, count=-1, sep=""):
    """Compatibility shim for SoundCard on NumPy 2.x."""
    if sep == "":
        try:
            # np.fromstring(binary) used to return a fresh array. Keep that
            # behavior so SoundCard does not read from a reused backing buffer.
            return np.frombuffer(string, dtype=dtype, count=count).copy()
        except (TypeError, ValueError):
            pass
    return _ORIG_FROMSTRING(string, dtype=dtype, count=count, sep=sep)


if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    np.fromstring = _compat_fromstring

import soundcard as sc

logger = logging.getLogger(__name__)


@dataclass
class SounddeviceConfig:
    """Settings for loopback capture and optional postprocessing."""

    sample_rate: Optional[int] = None
    channels: int = 1
    blocksize: int = 4096
    device: Optional[Union[int, str]] = None  # speaker index or name fragment
    start_threshold: float = 0.0015
    silence_threshold: float = 0.001
    max_silence_sec: float = 0.8
    max_record_sec: float = 20.0
    start_timeout_sec: float = 8.0
    pre_roll_sec: float = 0.8
    min_audio_after_start_sec: float = 0.0
    target_rms: float = 0.03
    max_gain: float = 8.0
    apply_postprocess: bool = False
    debug_dump_dir: Optional[str] = None
    auto_probe_sample_rate: bool = True


class SounddeviceAdapter:
    """
    Loopback adapter backed by soundcard (WASAPI loopback on Windows).

    The adapter records directly from a loopback recorder in a synchronous loop,
    which avoids a background queue/thread that can block shutdown.
    """

    def __init__(self, config: Optional[SounddeviceConfig] = None) -> None:
        """Initialize adapter state and apply default config values."""
        self.config = config or SounddeviceConfig()
        self._speaker = None
        self._loopback_mic = None
        self._active_recorder = None
        self._running = False
        self._debug_dump_index = 0

    @staticmethod
    def list_devices() -> list[dict]:
        """List output devices that can be used for loopback capture."""
        default = sc.default_speaker()
        default_name = str(default.name) if default is not None else ""
        items = []
        for idx, sp in enumerate(sc.all_speakers()):
            items.append(
                {
                    "index": idx,
                    "name": sp.name,
                    "id": sp.id,
                    "hostapi": "soundcard",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000.0,
                    "is_default_input": False,
                    "is_default_output": sp.name == default_name,
                }
            )
        return items

    @staticmethod
    def list_loopback_candidates() -> list[dict]:
        """Return loopback candidates (alias of `list_devices`)."""
        return SounddeviceAdapter.list_devices()

    def _resolve_speaker(self):
        """Resolve speaker from config (default, index, or name match)."""
        if self.config.device is None:
            sp = sc.default_speaker()
            if sp is None:
                raise RuntimeError("No default speaker found for loopback capture.")
            return sp

        if isinstance(self.config.device, int):
            speakers = sc.all_speakers()
            idx = self.config.device
            if idx < 0 or idx >= len(speakers):
                raise ValueError(f"Invalid speaker index: {idx}")
            return speakers[idx]

        query = str(self.config.device).strip().lower()
        if query.isdigit():
            speakers = sc.all_speakers()
            idx = int(query)
            if idx < 0 or idx >= len(speakers):
                raise ValueError(f"Invalid speaker index: {idx}")
            return speakers[idx]

        for sp in sc.all_speakers():
            if query in sp.name.lower():
                return sp

        raise ValueError(f"No speaker matches '{self.config.device}'")

    def _resolve_loopback_mic(self):
        """Resolve loopback microphone for the selected speaker."""
        sp = self._resolve_speaker()
        self._speaker = sp

        # Some setups are more stable with speaker name, others with id.
        for value in (str(sp.name), str(sp.id)):
            mic = sc.get_microphone(id=value, include_loopback=True)
            if mic is not None:
                return mic

        raise RuntimeError(f"Loopback microphone not found for speaker '{sp.name}'")

    def _choose_sample_rate(self, mic) -> int:
        """Choose a stable sample rate, optionally probing common values."""
        if self.config.sample_rate is not None:
            return int(self.config.sample_rate)
        if not self.config.auto_probe_sample_rate:
            return 48000

        channels = max(1, self.config.channels)
        candidates = (48000, 44100, 32000, 24000, 16000)
        for sr in candidates:
            try:
                with mic.recorder(
                    samplerate=sr,
                    channels=channels,
                    blocksize=self.config.blocksize,
                ) as rec:
                    chunk = rec.record(numframes=min(512, self.config.blocksize))
                if chunk is not None and len(chunk) > 0:
                    logger.debug("Selected loopback sample rate: %s", sr)
                    return sr
            except Exception as exc:
                logger.debug("Sample rate probe failed for %s Hz: %s", sr, exc)

        logger.warning("Sample rate probe failed; falling back to 48000 Hz")
        return 48000

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        """Return root-mean-square level for the given waveform."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)))

    @staticmethod
    def _to_mono(chunk: np.ndarray) -> np.ndarray:
        """Convert waveform chunk to mono float32."""
        if chunk.ndim == 1:
            return chunk.astype(np.float32, copy=False)
        if chunk.shape[1] == 1:
            return chunk[:, 0].astype(np.float32, copy=False)
        return chunk.mean(axis=1, dtype=np.float32)

    def get_selected_stream_info(self) -> dict:
        """Return metadata for the active or selected loopback stream."""
        mic = self._resolve_loopback_mic()
        sp = self._speaker
        return {
            "device": self.config.device,
            "name": sp.name if sp is not None else str(mic),
            "hostapi": "soundcard",
            "sample_rate": int(self.config.sample_rate or 48000),
            "channels": max(1, self.config.channels),
            "loopback_enabled": True,
            "backend": "soundcard",
        }

    def start_loopback(self) -> None:
        """Start loopback recording if recorder is not already active."""
        if self._active_recorder is not None:
            return

        self._loopback_mic = self._resolve_loopback_mic()
        sample_rate = self._choose_sample_rate(self._loopback_mic)
        channels = max(1, self.config.channels)

        self._active_recorder = self._loopback_mic.recorder(
            samplerate=sample_rate,
            channels=channels,
            blocksize=self.config.blocksize,
        )
        self._active_recorder.__enter__()
        self._running = True

        self.config.sample_rate = sample_rate
        self.config.channels = channels
        logger.debug(
            "Started soundcard loopback (speaker=%s, sample_rate=%s, channels=%s)",
            self._speaker.name if self._speaker is not None else str(self._loopback_mic),
            sample_rate,
            channels,
        )

    def stop(self) -> None:
        """Stop loopback recording and release recorder resources."""
        self._running = False
        if self._active_recorder is not None:
            self._active_recorder.__exit__(None, None, None)
            self._active_recorder = None
        self._loopback_mic = None
        self._speaker = None

    def read_chunks(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield continuous audio chunks from the active loopback stream."""
        self.start_loopback()
        sample_rate = int(self.config.sample_rate or 16000)

        while self._running and self._active_recorder is not None:
            chunk = self._active_recorder.record(numframes=self.config.blocksize)
            if chunk is None or len(chunk) == 0:
                continue
            yield np.asarray(chunk, dtype=np.float32), sample_rate

    def record_for_duration(self, duration_sec: float) -> Tuple[np.ndarray, int]:
        """
        Record loopback audio for a fixed duration in seconds.
        """
        self.start_loopback()
        if self._active_recorder is None:
            return np.zeros((0,), dtype=np.float32), int(self.config.sample_rate or 48000)

        duration_sec = max(0.0, float(duration_sec))
        sample_rate = int(self.config.sample_rate or 48000)
        if duration_sec == 0.0:
            return np.zeros((0,), dtype=np.float32), sample_rate

        rec = self._active_recorder
        target_frames = max(1, int(duration_sec * sample_rate))
        collected = 0
        frames: list[np.ndarray] = []

        while collected < target_frames:
            frames_left = target_frames - collected
            numframes = min(self.config.blocksize, max(1, frames_left))
            block = rec.record(numframes=numframes)
            if block is None or len(block) == 0:
                continue
            mono = self._to_mono(np.asarray(block, dtype=np.float32))
            frames.append(mono.copy())
            collected += len(mono)

        raw_audio = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
        raw_audio = np.nan_to_num(raw_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_audio = np.clip(raw_audio, -1.0, 1.0)
        self._maybe_dump_audio(raw_audio, sample_rate, "raw")

        audio = (
            self._postprocess_audio(raw_audio)
            if self.config.apply_postprocess
            else raw_audio
        )
        self._maybe_dump_audio(audio, sample_rate, "processed")
        logger.debug("Captured fixed-duration loopback audio duration=%.2fs", len(audio) / sample_rate)
        return audio, sample_rate

    def record_until_silence(self) -> Tuple[np.ndarray, int]:
        """Record speech until trailing silence or timeout is detected."""
        self.start_loopback()
        if self._active_recorder is None:
            return np.zeros((0,), dtype=np.float32), int(self.config.sample_rate or 48000)

        rec = self._active_recorder
        sample_rate = int(self.config.sample_rate or 48000)
        pre_roll_blocks = max(1, int(self.config.pre_roll_sec * sample_rate / self.config.blocksize))
        silence_blocks_needed = max(
            1,
            int(self.config.max_silence_sec * sample_rate / self.config.blocksize),
        )

        pre: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
        frames: list[np.ndarray] = []
        started = False
        silent_blocks = 0

        start_time = time.monotonic()
        start_deadline = start_time + self.config.start_timeout_sec
        stop_deadline = start_time + self.config.max_record_sec

        while True:
            block = rec.record(numframes=self.config.blocksize)
            if block is None or len(block) == 0:
                if time.monotonic() > stop_deadline:
                    break
                continue

            mono = self._to_mono(np.asarray(block, dtype=np.float32))
            pre.append(mono.copy())
            level = self._rms(mono)
            now = time.monotonic()

            if not started:
                if level >= self.config.start_threshold:
                    started = True
                    frames.extend(list(pre))
                    silent_blocks = 0
                elif now > start_deadline:
                    logger.debug("No loopback audio before start timeout.")
                    return np.zeros((0,), dtype=np.float32), sample_rate
            else:
                frames.append(mono.copy())
                if level < self.config.silence_threshold:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

                if silent_blocks >= silence_blocks_needed:
                    break

            if now > stop_deadline:
                break

        if not frames:
            return np.zeros((0,), dtype=np.float32), sample_rate

        raw_audio = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
        raw_audio = np.nan_to_num(raw_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_audio = np.clip(raw_audio, -1.0, 1.0)
        self._maybe_dump_audio(raw_audio, sample_rate, "raw")

        audio = (
            self._postprocess_audio(raw_audio)
            if self.config.apply_postprocess
            else raw_audio
        )
        self._maybe_dump_audio(audio, sample_rate, "processed")
        logger.debug("Captured loopback audio duration=%.2fs", len(audio) / sample_rate)
        return audio, sample_rate

    def _maybe_dump_audio(self, audio: np.ndarray, sample_rate: int, label: str) -> None:
        """Write diagnostic WAV files when debug dump directory is configured."""
        dump_dir = self.config.debug_dump_dir
        if not dump_dir:
            return

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.size == 0:
            return

        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._debug_dump_index += 1
        filename = f"{timestamp}_{self._debug_dump_index:04d}_{label}.wav"
        out_path = out_dir / filename

        pcm16 = np.clip(arr, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())

        logger.debug("Wrote debug audio: %s", out_path)

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply DC removal and RMS normalization to captured audio."""
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)

        audio = audio.astype(np.float32, copy=False)
        audio = audio - float(np.mean(audio))

        rms = self._rms(audio)
        if rms > 1e-8:
            gain = min(self.config.target_rms / rms, self.config.max_gain)
            audio = np.clip(audio * gain, -1.0, 1.0)
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        out_rms = self._rms(audio)
        logger.debug("Postprocessed audio (rms=%.6f, peak=%.6f)", out_rms, peak)
        return audio
