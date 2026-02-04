from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import whisper


AudioInput = Union[
    Dict[str, Any],  # {"array": np.ndarray, "sampling_rate": int}
    tuple,  # (np.ndarray, sampling_rate)
]

logger = logging.getLogger(__name__)


@dataclass
class WhisperAdapterConfig:
    """Model and inference options for Whisper transcription."""

    model: str = "turbo"
    device: str = "cuda"
    fp16: Optional[bool] = False  # Changed to False - fp16=True causes NaN/hallucinations on some GPUs
    language: str = "pl"
    task: str = "transcribe"
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    normalize_audio: bool = True
    target_rms: float = 0.04
    max_gain: float = 20.0


class WhisperAdapter:
    """Thin adapter that normalizes audio and runs Whisper transcription."""

    def __init__(self, config: Optional[WhisperAdapterConfig] = None) -> None:
        """Initialize Whisper model and runtime device settings."""
        self.config = config or WhisperAdapterConfig()
        self.device = self.config.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        # fp16 requires CUDA; default to False to avoid NaN issues on some GPUs
        if self.config.fp16 is None:
            self.fp16 = False  # Safe default - fp16=True can cause NaN on some hardware
        else:
            self.fp16 = bool(self.config.fp16) and self.device.startswith("cuda")

        self._model = whisper.load_model(self.config.model, device=self.device)
        logger.debug(
            "Whisper initialized (model=%s, device=%s, fp16=%s)",
            self.config.model,
            self.device,
            self.fp16,
        )

    @staticmethod
    def _to_mono_float32(array: np.ndarray) -> np.ndarray:
        """Convert arbitrary waveform input to clipped mono float32."""
        if array.ndim > 1:
            array = array.mean(axis=1)
        array = np.asarray(array, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(array, -1.0, 1.0)

    @staticmethod
    def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
        """Resample waveform with linear interpolation to target sample rate."""
        if src_rate == dst_rate or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        x_old = np.arange(audio.size, dtype=np.float32)
        new_len = max(1, int(round(audio.size * (dst_rate / float(src_rate)))))
        x_new = np.linspace(0, audio.size - 1, num=new_len, dtype=np.float32)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Optionally normalize waveform loudness before inference."""
        if not self.config.normalize_audio or audio.size == 0:
            return audio
        audio = audio - float(audio.mean())
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        if rms > 1e-8 and rms < self.config.target_rms:
            gain = min(self.config.target_rms / rms, self.config.max_gain)
            audio = audio * gain
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def transcribe(self, audio: AudioInput, sample_rate: Optional[int] = None) -> str:
        """Transcribe audio input and return recognized text."""
        if isinstance(audio, tuple):
            array, sr = audio
            sr = sample_rate or sr
        elif isinstance(audio, dict):
            array = audio.get("array")
            sr = sample_rate if sample_rate is not None else audio.get("sampling_rate")
        else:
            raise TypeError("audio must be a tuple or dict with array and sampling_rate")

        if array is None:
            logger.debug("Whisper transcribe skipped: empty audio array")
            return ""
        if sr is None:
            sr = 16000

        waveform = self._to_mono_float32(np.asarray(array))
        waveform = self._normalize_audio(waveform)
        waveform = self._resample_linear(waveform, int(sr), 16000)
        duration_s = float(waveform.size) / 16000.0 if waveform.size else 0.0
        rms = float(np.sqrt(np.mean(waveform * waveform))) if waveform.size else 0.0
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0

        logger.debug(
            "Whisper transcribe start (device=%s, fp16=%s, language=%s, sr=%s->16000, duration=%.2fs, rms=%.6f, peak=%.6f)",
            self.device,
            self.fp16,
            self.config.language,
            sr,
            duration_s,
            rms,
            peak,
        )

        result = self._model.transcribe(
            waveform,
            fp16=self.fp16,
            language=self.config.language,
            task=self.config.task,
            temperature=self.config.temperature,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        text = str(result.get("text", "")).strip()
        logger.debug("Whisper transcription: %r", text[:300])
        return text
