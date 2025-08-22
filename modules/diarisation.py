# modules/diarization.py

import whisperx
import tempfile
import os
import torch
from typing import List, Tuple, BinaryIO
from modules.logger import logger


class DiarizationService:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = whisperx.load_model("medium", device=self.device)
            logger.info(f"✅ WhisperX model loaded on {self.device}")
        except Exception as e:
            logger.exception("❌ Failed to load WhisperX model.")
            raise RuntimeError("Could not initialize WhisperX model.") from e

    def transcribe_and_diarize(self, file: BinaryIO, extension: str = "mp3") -> List[Tuple[str, str]]:
        """
        Transcribe audio and segment by speaker.
        Returns a list of (speaker, text) tuples.
        """
        if extension not in ['mp3', 'wav', 'm4a', 'webm']:
            logger.error(f"Unsupported audio format: {extension}")
            raise ValueError(f"Unsupported audio format: .{extension}")

        tmp_path = None
        try:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            logger.info(f"Transcribing and diarizing: {tmp_path}")
            audio = whisperx.load_audio(tmp_path)
            transcription = self.model.transcribe(audio, batch_size=16)

            if "segments" not in transcription:
                logger.warning("⚠️ No segments found in transcription.")
                return [("Unknown", transcription.get("text", ""))]

            # Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=self.device)
            aligned_transcription = whisperx.align(transcription["segments"], model_a, metadata, audio, self.device)

            # Diarization
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=self.device)
            diarize_segments = diarize_model(tmp_path)

            # Assign speaker labels
            result = whisperx.assign_word_speakers(diarize_segments, aligned_transcription["word_segments"])

            # Group words by speaker
            speaker_map = {}
            for word_info in result["word_segments"]:
                speaker = word_info.get("speaker", "Unknown")
                speaker_map.setdefault(speaker, []).append(word_info["text"])

            speaker_chunks = [(speaker, " ".join(words)) for speaker, words in speaker_map.items()]
            logger.info(f"🗣️ Diarization result: {len(speaker_chunks)} speakers detected.")
            return speaker_chunks

        except Exception as e:
            logger.exception("❌ Diarization failed.")
            return [("Unknown", "Diarization failed.")]

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Deleted temporary file: {tmp_path}")
