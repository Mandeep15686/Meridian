"""Whisper ASR wrapper with silence-based chunking and transcript assembly."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.registry import MODELS

logger = logging.getLogger(__name__)

HF_INFERENCE_BASE = os.getenv(
    "HF_INFERENCE_ENDPOINT",
    "https://router.huggingface.co",
).rstrip("/")

HF_ASR_URL = f"{HF_INFERENCE_BASE}/models/{MODELS.asr}"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"


@dataclass
class Transcription:
    full_text: str
    segments: list[TranscriptSegment]
    language: str = "en"
    duration_seconds: float = 0.0
    word_count: int = 0

    @property
    def compliance_statements(self) -> list[str]:
        """Return sentences that contain regulatory keywords."""
        keywords = {
            "retain",
            "retention",
            "delete",
            "deletion",
            "gdpr",
            "consent",
            "lawful basis",
            "dpo",
            "data protection",
            "personal data",
            "third party",
            "transfer",
            "breach",
            "notification",
            "soc",
            "audit",
            "compliance",
            "regulation",
            "policy",
            "privacy",
        }
        results = []
        for sentence in self.full_text.split(". "):
            if any(kw in sentence.lower() for kw in keywords):
                results.append(sentence.strip())
        return results


class ASRModel:
    """
    Whisper large-v3 transcription via the HuggingFace Inference API.

    Handles:
    - Audio normalisation to 16kHz mono WAV (required by Whisper)
    - Silence-based chunking (VAD with webrtcvad)
    - Chunk upload and transcript assembly with timestamps
    - Optional initial prompt for domain vocabulary bias
    """

    SAMPLE_RATE = 16_000
    CHUNK_DURATION_S = settings.AUDIO_CHUNK_DURATION_S  # 30 seconds

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {settings.HF_API_TOKEN}",
        }

    async def transcribe(self, audio_path: Path, language: str = "en") -> Transcription:
        """
        Transcribe an audio file end-to-end.

        Args:
            audio_path: Path to the audio file (any format supported by ffmpeg).
            language: ISO 639-1 language code.

        Returns:
            Assembled Transcription with full text, segments, and duration.
        """
        import librosa
        import soundfile as sf

        # ── 1. Normalise to 16kHz mono WAV ────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = Path(tmp_wav.name)

        try:
            audio, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
            sf.write(str(wav_path), audio, self.SAMPLE_RATE, subtype="PCM_16")
            duration = len(audio) / self.SAMPLE_RATE

            # ── 2. Split at silence boundaries ────────────────────────────────
            chunks = self._split_on_silence(wav_path, duration)

            # ── 3. Transcribe each chunk ───────────────────────────────────────
            segments: list[TranscriptSegment] = []
            for chunk_bytes, chunk_start in chunks:
                text = await self._transcribe_chunk(chunk_bytes, language)
                if text:
                    # Estimate end from chunk duration
                    chunk_duration = len(chunk_bytes) / (self.SAMPLE_RATE * 2)  # 16-bit
                    segments.append(
                        TranscriptSegment(
                            start=chunk_start,
                            end=chunk_start + chunk_duration,
                            text=text.strip(),
                        )
                    )

        finally:
            wav_path.unlink(missing_ok=True)

        full_text = " ".join(seg.text for seg in segments)
        return Transcription(
            full_text=full_text,
            segments=segments,
            language=language,
            duration_seconds=duration,
            word_count=len(full_text.split()),
        )

    def _split_on_silence(
        self, wav_path: Path, total_duration: float
    ) -> list[tuple[bytes, float]]:
        """
        Split a WAV file at silence boundaries using webrtcvad.

        Returns list of (pcm_bytes, start_time_seconds).
        """
        try:
            import wave

            import webrtcvad

            vad = webrtcvad.Vad(mode=2)  # aggressiveness 0–3
            frame_duration_ms = 30
            sample_rate = self.SAMPLE_RATE
            frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit

            with wave.open(str(wav_path), "rb") as wf:
                pcm_data = wf.readframes(wf.getnframes())

            frames = [
                pcm_data[i : i + frame_size]
                for i in range(0, len(pcm_data) - frame_size, frame_size)
            ]

            chunks: list[tuple[bytes, float]] = []
            current_chunk: list[bytes] = []
            chunk_start = 0.0
            silence_frames = 0
            silence_threshold = int(
                1.0 * 1000 / frame_duration_ms
            )  # 1 second of silence

            for i, frame in enumerate(frames):
                if len(frame) < frame_size:
                    break
                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    current_chunk.append(frame)
                    silence_frames = 0
                else:
                    silence_frames += 1
                    current_chunk.append(frame)
                    chunk_time = (i * frame_duration_ms) / 1000
                    # Flush chunk if silence or approaching max chunk duration
                    if (
                        silence_frames >= silence_threshold
                        or chunk_time - chunk_start >= self.CHUNK_DURATION_S
                    ) and current_chunk:
                        chunks.append((b"".join(current_chunk), chunk_start))
                        chunk_start = chunk_time
                        current_chunk = []
                        silence_frames = 0

            # Flush remaining
            if current_chunk:
                chunks.append((b"".join(current_chunk), chunk_start))

            return chunks if chunks else [(pcm_data, 0.0)]

        except ImportError:
            logger.warning(
                "webrtcvad not available — falling back to fixed-size chunks"
            )
            return self._fixed_size_chunks(wav_path)

    def _fixed_size_chunks(self, wav_path: Path) -> list[tuple[bytes, float]]:
        """Fallback: split into fixed 30-second chunks."""
        import wave

        frame_count = self.SAMPLE_RATE * self.CHUNK_DURATION_S
        chunks: list[tuple[bytes, float]] = []

        with wave.open(str(wav_path), "rb") as wf:
            idx = 0
            while True:
                frames = wf.readframes(frame_count)
                if not frames:
                    break
                start = idx * self.CHUNK_DURATION_S
                chunks.append((frames, float(start)))
                idx += 1

        return chunks

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8), reraise=True
    )
    async def _transcribe_chunk(self, pcm_bytes: bytes, language: str) -> str:
        """Send a single PCM chunk to the Whisper HF endpoint."""
        # Wrap raw PCM in a minimal WAV container
        import wave

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        wav_buf.seek(0)

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                HF_ASR_URL,
                headers={
                    **self._headers,
                    "Content-Type": "audio/wav",
                },
                params={
                    "language": language,
                    "initial_prompt": settings.WHISPER_INITIAL_PROMPT,
                },
                content=wav_buf.read(),
            )

        if response.status_code == 503:
            raise httpx.HTTPStatusError(
                "Whisper model loading", request=response.request, response=response
            )

        if response.status_code != 200:
            logger.error(
                "Whisper API error %d: %s", response.status_code, response.text[:200]
            )
            return ""

        data = response.json()
        return cast(str, data.get("text", ""))
