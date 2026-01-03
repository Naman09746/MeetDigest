# modules/transcriber.py

import tempfile
import os
import math
from typing import BinaryIO, Optional, List, Dict, Union, Literal
from dataclasses import dataclass
from modules.logger import logger
import streamlit as st
from modules.meeting_context import MeetingContext
SUPPORTED_AUDIO_EXTENSIONS = ['mp3', 'wav', 'm4a', 'webm', 'flac', 'ogg']
DEFAULT_MODEL_NAME = "base"  # Can be "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
CHUNK_DURATION = 300  # 5 minutes in seconds
OVERLAP_DURATION = 30  # 30 seconds overlap between chunks

# Available Whisper models with their characteristics
WHISPER_MODELS = {
    "tiny": {"size": "~39 MB", "speed": "fastest", "accuracy": "lowest"},
    "base": {"size": "~74 MB", "speed": "fast", "accuracy": "good"},
    "small": {"size": "~244 MB", "speed": "medium", "accuracy": "better"},
    "medium": {"size": "~769 MB", "speed": "slow", "accuracy": "very good"},
    "large": {"size": "~1550 MB", "speed": "slowest", "accuracy": "best"},
    "large-v2": {"size": "~1550 MB", "speed": "slowest", "accuracy": "best"},
    "large-v3": {"size": "~1550 MB", "speed": "slowest", "accuracy": "best"}
}

# Common languages for Whisper
LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi"
}

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed audio with timing information."""
    start: float
    end: float
    text: str
    id: Optional[int] = None
    
    def duration(self) -> float:
        """Get duration of the segment in seconds."""
        return self.end - self.start

@dataclass 
class TranscriptionResult:
    """Complete transcription result with segments and metadata."""
    text: str
    segments: List[TranscriptionSegment]
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None

TranscriptionEngine = Literal["whisper", "faster-whisper"]


@st.cache_resource
def load_whisper_model(model_name: str = DEFAULT_MODEL_NAME, engine: TranscriptionEngine = "whisper"):
    """
    Load Whisper model once and cache it.
    
    Args:
        model_name: Size of the model to load
        engine: Which Whisper implementation to use
    """
    try:
        if engine == "faster-whisper":
            from faster_whisper import WhisperModel
            import torch
            
            # Use GPU if available for faster-whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            model = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4
            )
            logger.info(f"✅ Faster-Whisper model '{model_name}' loaded on {device}")
        else:
            import whisper
            model = whisper.load_model(model_name)
            logger.info(f"✅ Whisper model '{model_name}' loaded successfully")
            
        return model, engine
    except ImportError as e:
        if engine == "faster-whisper":
            logger.warning("⚠️ faster-whisper not available, falling back to standard whisper")
            import whisper
            model = whisper.load_model(model_name)
            return model, "whisper"
        else:
            raise RuntimeError(f"Whisper library not available: {e}")
    except Exception as e:
        logger.exception("❌ Failed to load Whisper model")
        raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}")


def get_audio_duration(file_path: str) -> Optional[float]:
    """Get audio file duration in seconds."""
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except ImportError:
        logger.warning("⚠️ librosa not available, cannot determine audio duration")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Could not determine audio duration: {e}")
        return None


def split_audio_chunks(file_path: str, chunk_duration: int = CHUNK_DURATION, 
                      overlap_duration: int = OVERLAP_DURATION) -> List[Dict]:
    """
    Split audio into overlapping chunks for processing long files.
    
    Returns:
        List of chunk info dicts with start_time, end_time, and file_path
    """
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(file_path)
        total_duration = len(audio) / 1000  # Convert to seconds
        
        if total_duration <= chunk_duration:
            return [{"start_time": 0, "end_time": total_duration, "file_path": file_path}]
        
        chunks = []
        chunk_start = 0
        chunk_num = 0
        
        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_duration, total_duration)
            
            # Extract chunk with overlap
            start_ms = int(chunk_start * 1000)
            end_ms = int(chunk_end * 1000)
            chunk_audio = audio[start_ms:end_ms]
            
            # Save chunk to temp file
            chunk_path = f"{file_path}_chunk_{chunk_num}.wav"
            chunk_audio.export(chunk_path, format="wav")
            
            chunks.append({
                "start_time": chunk_start,
                "end_time": chunk_end,
                "file_path": chunk_path
            })
            
            chunk_start = chunk_end - overlap_duration
            chunk_num += 1
            
        logger.info(f"🔪 Split audio into {len(chunks)} chunks")
        return chunks
        
    except ImportError:
        logger.warning("⚠️ pydub not available, cannot split long audio files")
        return [{"start_time": 0, "end_time": None, "file_path": file_path}]
    except Exception as e:
        logger.warning(f"⚠️ Could not split audio: {e}")
        return [{"start_time": 0, "end_time": None, "file_path": file_path}]


def merge_overlapping_segments(segments: List[TranscriptionSegment], 
                             overlap_duration: int = OVERLAP_DURATION) -> List[TranscriptionSegment]:
    """Merge segments from overlapping chunks, removing duplicates."""
    if len(segments) <= 1:
        return segments
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        previous = merged[-1]
        
        # If there's significant overlap, merge the segments
        if current.start < previous.end - overlap_duration/2:
            # Merge by extending the previous segment
            merged_text = previous.text
            if not previous.text.endswith(' '):
                merged_text += ' '
            merged_text += current.text
            
            merged[-1] = TranscriptionSegment(
                start=previous.start,
                end=current.end,
                text=merged_text,
                id=previous.id
            )
        else:
            merged.append(current)
    
    return merged


def transcribe_with_whisper(file_path: str, model, language: Optional[str] = None) -> Dict:
    """Transcribe using standard Whisper."""
    import whisper
    
    options = {}
    if language and language != "auto":
        options["language"] = language
    
    result = model.transcribe(file_path, **options)
    return result


def transcribe_with_faster_whisper(file_path: str, model, language: Optional[str] = None) -> Dict:
    """Transcribe using faster-whisper."""
    options = {}
    if language and language != "auto":
        options["language"] = language
    
    segments, info = model.transcribe(file_path, **options)
    
    # Convert to whisper-like format
    segments_list = []
    full_text = ""
    
    for segment in segments:
        segments_list.append({
            "id": segment.id,
            "start": segment.start,
            "end": segment.end, 
            "text": segment.text
        })
        full_text += segment.text + " "
    
    return {
        "text": full_text.strip(),
        "segments": segments_list,
        "language": info.language,
        "language_probability": info.language_probability
    }


def transcribe_audio_chunk(chunk_path: str, model, engine: str, 
                          chunk_start_time: float = 0,
                          language: Optional[str] = None) -> List[TranscriptionSegment]:
    """Transcribe a single audio chunk."""
    try:
        if engine == "faster-whisper":
            result = transcribe_with_faster_whisper(chunk_path, model, language)
        else:
            result = transcribe_with_whisper(chunk_path, model, language)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                id=seg.get("id"),
                start=seg["start"] + chunk_start_time,
                end=seg["end"] + chunk_start_time,
                text=seg["text"].strip()
            ))
        
        return segments
        
    except Exception as e:
        logger.error(f"❌ Failed to transcribe chunk {chunk_path}: {e}")
        return []


def transcribe_audio(
    file: BinaryIO, 
    extension: str = 'mp3',
    model_name: str = DEFAULT_MODEL_NAME,
    engine: TranscriptionEngine = "whisper",
    language: Optional[str] = None,
    enable_chunking: bool = True,
    chunk_duration: int = CHUNK_DURATION,
    return_segments: bool = True
) -> MeetingContext:
    """
    Transcribe audio using Whisper model with advanced options
    and return a MeetingContext for downstream pipeline stages.
    """
    extension = extension.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.error(f"❌ Unsupported audio format: .{extension}")
        raise ValueError(
            f"Unsupported audio format: .{extension}. "
            f"Supported: {SUPPORTED_AUDIO_EXTENSIONS}"
        )

    tmp_path = None
    chunk_files = []

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        logger.info(
            f"🔊 Transcribing audio: {tmp_path} "
            f"(model={model_name}, engine={engine})"
        )

        # Get audio duration
        duration = get_audio_duration(tmp_path)
        if duration:
            logger.info(f"📏 Audio duration: {duration:.1f} seconds")

        # Load Whisper / Faster-Whisper model
        model, actual_engine = load_whisper_model(model_name, engine)

        # Decide whether to chunk
        should_chunk = (
            enable_chunking
            and duration
            and duration > chunk_duration
            and actual_engine == "whisper"
        )

        if should_chunk:
            logger.info(
                f"🔪 Long audio detected, splitting into {chunk_duration}s chunks"
            )

            chunks = split_audio_chunks(
                tmp_path, chunk_duration, OVERLAP_DURATION
            )
            chunk_files.extend(
                [c["file_path"] for c in chunks if c["file_path"] != tmp_path]
            )

            all_segments: List[TranscriptionSegment] = []

            for i, chunk in enumerate(chunks):
                logger.info(
                    f"🔄 Processing chunk {i + 1}/{len(chunks)}"
                )

                chunk_segments = transcribe_audio_chunk(
                    chunk["file_path"],
                    model,
                    actual_engine,
                    chunk["start_time"],
                    language
                )
                all_segments.extend(chunk_segments)

            segments = merge_overlapping_segments(
                all_segments, OVERLAP_DURATION
            )
        else:
            segments = transcribe_audio_chunk(
                tmp_path, model, actual_engine, 0, language
            )

        # Build full text
        full_text = " ".join(seg.text for seg in segments).strip()

        # -------------------------
        # EMPTY TRANSCRIPTION CASE
        # -------------------------
        if not full_text:
            logger.warning("⚠️ Transcription returned empty text")

            context = MeetingContext()
            context.raw_text = ""
            context.metadata["duration"] = duration
            context.metadata["model"] = model_name
            context.metadata["engine"] = actual_engine
            context.metadata["error"] = "Empty transcription"
            context.metadata["source"] = "audio"
            context.metadata["file_extension"] = extension

            return context

        logger.info(
            f"✅ Transcription completed: "
            f"{len(segments)} segments, "
            f"{len(full_text)} characters"
        )

        # -------------------------
        # SUCCESS CASE
        # -------------------------
        context = MeetingContext()
        context.raw_text = full_text

        context.metadata["duration"] = duration
        context.metadata["model"] = model_name
        context.metadata["engine"] = actual_engine
        context.metadata["language"] = language
        context.metadata["source"] = "audio"
        context.metadata["file_extension"] = extension

        if return_segments:
            context.metadata["segments"] = segments

        return context

    except Exception as e:
        logger.exception("❌ Audio transcription failed")
        raise RuntimeError(f"Audio transcription failed: {e}")

    finally:
        # Cleanup temp files
        for file_path in [tmp_path] + chunk_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"🧹 Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not delete temp file {file_path}: {e}"
                    )


def get_model_info() -> Dict[str, Dict]:
    """Get information about available Whisper models."""
    return WHISPER_MODELS.copy()


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages for transcription."""
    return LANGUAGES.copy()


# Streamlit UI helpers
def create_transcription_sidebar():
    """Create sidebar controls for transcription settings."""
    with st.sidebar.expander("🎙️ Transcription Settings", expanded=False):
        model_name = st.selectbox(
            "Whisper Model",
            options=list(WHISPER_MODELS.keys()),
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower"
        )
        
        engine = st.selectbox(
            "Engine",
            options=["whisper", "faster-whisper"],
            index=0,
            help="faster-whisper is optimized for speed and GPU usage"
        )
        
        language = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=0,  # Default to "auto"
            help="Language hint for better accuracy"
        )
        
        enable_chunking = st.checkbox(
            "Enable chunking for long files",
            value=True,
            help="Split long audio files into chunks for better memory usage"
        )
        
        if enable_chunking:
            chunk_duration = st.slider(
                "Chunk duration (seconds)",
                min_value=60,
                max_value=600,
                value=300,
                step=30,
                help="Duration of each audio chunk"
            )
        else:
            chunk_duration = CHUNK_DURATION
            
        return_segments = st.checkbox(
            "Return detailed segments",
            value=True,
            help="Return transcription with timing information for each segment"
        )
    
    return {
        "model_name": model_name,
        "engine": engine,
        "language": language if language != "auto" else None,
        "enable_chunking": enable_chunking,
        "chunk_duration": chunk_duration,
        "return_segments": return_segments
    }