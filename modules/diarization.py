# modules/diarization.py
import whisperx
import tempfile
import os
import torch
import functools
from typing import List, Tuple, BinaryIO, Dict, Optional, Literal, NamedTuple
from dataclasses import dataclass
from modules.logger import logger
import numpy as np
from pathlib import Path
from modules.meeting_context import MeetingContext
# Type definitions
ModelSize = Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

@dataclass
class SpeakerStats:
    """Statistics for individual speakers."""
    speaker_id: str
    word_count: int
    talk_time_seconds: float
    talk_time_percentage: float
    segment_count: int

@dataclass
class DiarizationResult:
    """Complete diarization result with speaker segments and statistics."""
    speaker_segments: List[Tuple[str, str]]  # (speaker_label, text)
    speaker_stats: List[SpeakerStats]
    total_duration: float
    total_words: int

class AudioChunk(NamedTuple):
    """Represents a chunk of audio for processing."""
    start_time: float
    end_time: float
    audio_data: np.ndarray

# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, any] = {}
_ALIGN_MODEL_CACHE: Dict[str, Tuple[any, any]] = {}
_DIARIZATION_MODEL_CACHE: Dict[str, any] = {}

@functools.lru_cache(maxsize=4)
def get_cached_whisper_model(model_size: ModelSize, device: str):
    """Cache WhisperX models globally to avoid reloading."""
    cache_key = f"{model_size}_{device}"
    
    if cache_key not in _MODEL_CACHE:
        try:
            logger.info(f"🔄 Loading WhisperX model ({model_size}) on {device}...")
            model = whisperx.load_model(model_size, device=device)
            _MODEL_CACHE[cache_key] = model
            logger.info(f"✅ WhisperX model ({model_size}) cached successfully")
        except Exception as e:
            logger.exception(f"❌ Failed to load WhisperX model ({model_size})")
            raise RuntimeError(f"Could not load WhisperX model ({model_size})") from e
    
    return _MODEL_CACHE[cache_key]

@functools.lru_cache(maxsize=8)
def get_cached_align_model(language_code: str, device: str):
    """Cache alignment models by language."""
    cache_key = f"{language_code}_{device}"
    
    if cache_key not in _ALIGN_MODEL_CACHE:
        try:
            logger.info(f"🔄 Loading alignment model for {language_code}...")
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            _ALIGN_MODEL_CACHE[cache_key] = (model_a, metadata)
            logger.info(f"✅ Alignment model for {language_code} cached successfully")
        except Exception as e:
            logger.exception(f"❌ Failed to load alignment model for {language_code}")
            raise RuntimeError(f"Could not load alignment model for {language_code}") from e
    
    return _ALIGN_MODEL_CACHE[cache_key]

@functools.lru_cache(maxsize=2)
def get_cached_diarization_model(device: str, auth_token: Optional[str] = None):
    """Cache diarization models."""
    cache_key = f"diarization_{device}_{bool(auth_token)}"
    
    if cache_key not in _DIARIZATION_MODEL_CACHE:
        try:
            logger.info(f"🔄 Loading diarization model on {device}...")
            model = whisperx.DiarizationPipeline(use_auth_token=auth_token, device=device)
            _DIARIZATION_MODEL_CACHE[cache_key] = model
            logger.info("✅ Diarization model cached successfully")
        except Exception as e:
            logger.exception("❌ Failed to load diarization model")
            raise RuntimeError("Could not load diarization model") from e
    
    return _DIARIZATION_MODEL_CACHE[cache_key]

class DiarizationService:
    def __init__(
        self, 
        device: Optional[str] = None, 
        model_size: ModelSize = "medium",
        auth_token: Optional[str] = None,
        chunk_length_minutes: float = 25.0
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.auth_token = auth_token
        self.chunk_length_seconds = chunk_length_minutes * 60
        
        # Load and cache models during initialization
        try:
            self.model = get_cached_whisper_model(self.model_size, self.device)
            logger.info(f"✅ DiarizationService initialized with {model_size} model on {self.device}")
        except Exception as e:
            logger.exception("❌ Failed to initialize DiarizationService")
            raise

    def _chunk_audio(self, audio_path: str, audio_data: np.ndarray) -> List[AudioChunk]:
        """Split long audio into manageable chunks."""
        sample_rate = 16000  # WhisperX standard sample rate
        total_duration = len(audio_data) / sample_rate
        
        if total_duration <= self.chunk_length_seconds:
            return [AudioChunk(0, total_duration, audio_data)]
        
        chunks = []
        chunk_samples = int(self.chunk_length_seconds * sample_rate)
        
        for i in range(0, len(audio_data), chunk_samples):
            start_time = i / sample_rate
            end_time = min((i + chunk_samples) / sample_rate, total_duration)
            chunk_data = audio_data[i:i + chunk_samples]
            chunks.append(AudioChunk(start_time, end_time, chunk_data))
        
        logger.info(f"📦 Split {total_duration:.1f}s audio into {len(chunks)} chunks")
        return chunks

    def _relabel_speakers(self, speaker_segments: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Convert SPEAKER_0, SPEAKER_1 to Speaker 1, Speaker 2."""
        speaker_mapping = {}
        next_speaker_num = 1
        
        relabeled = []
        for original_speaker, text in speaker_segments:
            if original_speaker not in speaker_mapping:
                if original_speaker.startswith("SPEAKER_"):
                    speaker_mapping[original_speaker] = f"Speaker {next_speaker_num}"
                    next_speaker_num += 1
                else:
                    speaker_mapping[original_speaker] = original_speaker
            
            new_label = speaker_mapping[original_speaker]
            relabeled.append((new_label, text))
        
        return relabeled

    def _calculate_speaker_stats(
        self, 
        word_segments: List[Dict], 
        total_duration: float
    ) -> List[SpeakerStats]:
        """Calculate comprehensive speaker statistics."""
        speaker_data = {}
        
        for word_info in word_segments:
            speaker = word_info.get("speaker", "Unknown")
            word = word_info.get("word", "")
            start_time = word_info.get("start", 0)
            end_time = word_info.get("end", 0)
            duration = end_time - start_time
            
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    "words": [],
                    "total_time": 0.0,
                    "segments": 0
                }
            
            speaker_data[speaker]["words"].append(word)
            speaker_data[speaker]["total_time"] += duration
            speaker_data[speaker]["segments"] += 1
        
        # Convert to SpeakerStats objects
        stats = []
        total_words = sum(len(data["words"]) for data in speaker_data.values())
        
        for speaker, data in speaker_data.items():
            # Relabel speaker names
            if speaker.startswith("SPEAKER_"):
                try:
                    speaker_num = int(speaker.split("_")[1]) + 1
                    display_name = f"Speaker {speaker_num}"
                except (IndexError, ValueError):
                    display_name = speaker
            else:
                display_name = speaker
            
            talk_time_percentage = (data["total_time"] / total_duration * 100) if total_duration > 0 else 0
            
            stats.append(SpeakerStats(
                speaker_id=display_name,
                word_count=len(data["words"]),
                talk_time_seconds=data["total_time"],
                talk_time_percentage=talk_time_percentage,
                segment_count=data["segments"]
            ))
        
        # Sort by talk time (descending)
        stats.sort(key=lambda x: x.talk_time_seconds, reverse=True)
        return stats

    def _process_single_chunk(
        self, 
        chunk: AudioChunk, 
        tmp_path: str, 
        chunk_index: int
    ) -> Tuple[List[Dict], str]:
        """Process a single audio chunk."""
        try:
            logger.info(f"🔄 Processing chunk {chunk_index + 1}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
            
            # Transcribe
            transcription = self.model.transcribe(chunk.audio_data, batch_size=16)
            
            if "segments" not in transcription or not transcription["segments"]:
                logger.warning(f"⚠️ No segments found in chunk {chunk_index + 1}")
                return [], transcription.get("language", "en")
            
            # Align
            language = transcription.get("language", "en")
            model_a, metadata = get_cached_align_model(language, self.device)
            aligned_transcription = whisperx.align(
                transcription["segments"], model_a, metadata, chunk.audio_data, self.device
            )
            
            # Diarize
            diarize_model = get_cached_diarization_model(self.device, self.auth_token)
            diarize_segments = diarize_model(tmp_path, min_speakers=1, max_speakers=10)
            
            # Assign speakers
            result = whisperx.assign_word_speakers(diarize_segments, aligned_transcription["word_segments"])
            
            # Adjust timestamps for chunk offset
            word_segments = result.get("word_segments", [])
            for segment in word_segments:
                if "start" in segment:
                    segment["start"] += chunk.start_time
                if "end" in segment:
                    segment["end"] += chunk.start_time
            
            return word_segments, language
            
        except Exception as e:
            logger.exception(f"❌ Failed to process chunk {chunk_index + 1}")
            return [], "en"

    def transcribe_and_diarize(
        self, 
        file: BinaryIO, 
        extension: str,
        context: MeetingContext
    ) -> MeetingContext:
        """
        Perform speaker diarization using WhisperX and enrich an existing MeetingContext
        with speaker segments and speaker-level statistics.
        """

        if extension not in ['mp3', 'wav', 'm4a', 'webm', 'flac', 'ogg']:
            logger.error(f"Unsupported audio format: {extension}")
            raise ValueError(f"Unsupported audio format: .{extension}")
        
        tmp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                file.seek(0)
                tmp.write(file.read())
                tmp_path = tmp.name

            
            logger.info(f"🎵 Processing audio file: {tmp_path}")
            
            # Load audio
            audio_data = whisperx.load_audio(tmp_path)
            total_duration = len(audio_data) / 16000  # WhisperX uses 16kHz
            
            # Chunk audio for long files
            chunks = self._chunk_audio(tmp_path, audio_data)
            
            # Process all chunks
            all_word_segments = []
            detected_language = "en"
            # Process each chunk directly using the original audio file + time offsets
            for i, chunk in enumerate(chunks):
                word_segments, language = self._process_single_chunk(
                    chunk,
                    tmp_path,
                    i
                )
                all_word_segments.extend(word_segments)
                detected_language = language

            # Handle case where no speech was detected
            if not all_word_segments:
                logger.warning("⚠️ No word segments found in any chunks")

                context.speaker_segments = [("Unknown", "No speech detected")]
                context.speaker_stats = []

                context.metadata["duration"] = total_duration
                context.metadata["total_words"] = 0
                context.metadata["num_speakers"] = 0
                context.metadata["warning"] = "No speech detected"

                return context

            
            # Group words by speaker
            speaker_map = {}
            for word_info in all_word_segments:
                speaker = word_info.get("speaker", "Unknown")
                word_text = word_info.get("word", "").strip()
                if word_text:  # Only add non-empty words
                    speaker_map.setdefault(speaker, []).append(word_text)
            
            # Create speaker segments
            speaker_segments = [(speaker, " ".join(words)) for speaker, words in speaker_map.items()]
            
            # Relabel speakers to human-friendly names
            speaker_segments = self._relabel_speakers(speaker_segments)
            
            # Calculate statistics
            speaker_stats = self._calculate_speaker_stats(all_word_segments, total_duration)
            total_words = sum(stat.word_count for stat in speaker_stats)
            
            logger.info(f"🗣️ Diarization complete: {len(speaker_segments)} speakers, {total_words} words, {total_duration:.1f}s")
            
            context.speaker_segments = speaker_segments
            context.speaker_stats = speaker_stats

            context.metadata["duration"] = total_duration
            context.metadata["total_words"] = total_words
            context.metadata["num_speakers"] = len(speaker_segments)

            return context

            
        except Exception as e:
            logger.exception("❌ Diarization failed")

            context.speaker_segments = [("Unknown", "Diarization failed")]
            context.speaker_stats = []

            context.metadata["error"] = "Diarization failed"

            return context

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"🗑️ Cleaned up temporary file: {tmp_path}")

    def get_speaker_summary(self, context: MeetingContext) -> str:
        if not context.speaker_stats:
            return "No speakers detected."

        total_duration = context.metadata.get("duration", 0)
        total_words = context.metadata.get("total_words", 0)

        summary_lines = [
            f"📊 Speaker Analysis ({total_duration:.1f}s total, {total_words} words):\n"
        ]

        for i, stats in enumerate(context.speaker_stats, 1):
            pct_words = (stats.word_count / total_words * 100) if total_words else 0
            summary_lines.append(
                f"{i}. {stats.speaker_id}: "
                f"{stats.word_count} words ({pct_words:.1f}%), "
                f"{stats.talk_time_seconds:.1f}s ({stats.talk_time_percentage:.1f}%), "
                f"{stats.segment_count} segments"
            )

        return "\n".join(summary_lines)

    def clear_cache(self):
        """Clear all cached models to free memory."""
        global _MODEL_CACHE, _ALIGN_MODEL_CACHE, _DIARIZATION_MODEL_CACHE
        _MODEL_CACHE.clear()
        _ALIGN_MODEL_CACHE.clear()
        _DIARIZATION_MODEL_CACHE.clear()
        
        # Clear function caches
        get_cached_whisper_model.cache_clear()
        get_cached_align_model.cache_clear()
        get_cached_diarization_model.cache_clear()
        
        logger.info("🧹 All diarization model caches cleared")