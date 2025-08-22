# modules/transcriber.py

import tempfile
import os
from typing import BinaryIO
from modules.logger import logger
import streamlit as st

SUPPORTED_AUDIO_EXTENSIONS = ['mp3', 'wav', 'm4a', 'webm']
DEFAULT_MODEL_NAME = "base"  # Can be "small", "medium", "large"

@st.cache_resource
def load_whisper_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load Whisper model once and cache it.
    """
    import whisper
    try:
        model = whisper.load_model(model_name)
        logger.info(f"✅ Whisper model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.exception("❌ Failed to load Whisper model.")
        raise RuntimeError(f"Failed to load Whisper model '{model_name}'. Check your environment.") from e


def transcribe_audio(file: BinaryIO, extension: str = 'mp3', model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Transcribe audio using Whisper model.
    Returns plain text transcription.
    """
    extension = extension.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.error(f"❌ Unsupported audio format: .{extension}")
        raise ValueError(f"Unsupported audio format: .{extension}")

    tmp_path = None
    try:
        # Save the uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        logger.info(f"🔊 Transcribing audio file: {tmp_path} using model '{model_name}'")

        # Load model
        model = load_whisper_model(model_name)

        # Perform transcription
        result = model.transcribe(tmp_path)

        text = result.get("text", "").strip()
        if not text:
            logger.warning("⚠️ Transcription returned empty text.")
            return "No transcribed text found."

        logger.info("✅ Transcription completed successfully.")
        return text

    except Exception as e:
        logger.exception("❌ Audio transcription failed.")
        raise RuntimeError("Audio transcription failed. Please try again with a valid file.") from e

    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"🧹 Deleted temporary file: {tmp_path}")
