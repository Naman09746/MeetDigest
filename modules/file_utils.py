# modules/file_utils.py

from typing import BinaryIO
import os
from modules.logger import logger

# Supported formats
SUPPORTED_TEXT_FORMATS = ['txt', 'vtt', 'srt']
SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'm4a', 'webm']
SUPPORTED_ALL = SUPPORTED_TEXT_FORMATS + SUPPORTED_AUDIO_FORMATS


def get_file_extension(file) -> str:
    """
    Extract lowercase file extension from uploaded file.
    """
    try:
        filename = file.name if hasattr(file, 'name') else str(file)
        extension = os.path.splitext(filename)[1][1:].lower()
        logger.debug(f"File extension detected: .{extension}")
        return extension
    except Exception as e:
        logger.exception("❌ Failed to extract file extension.")
        return ""


def is_supported_file(file) -> bool:
    """
    Check if the uploaded file has a supported extension.
    """
    ext = get_file_extension(file)
    is_supported = ext in SUPPORTED_ALL
    logger.info(f"File supported: {is_supported} (.{ext})")
    return is_supported


def is_audio_file(file) -> bool:
    """
    Check if the uploaded file is an audio type.
    """
    ext = get_file_extension(file)
    return ext in SUPPORTED_AUDIO_FORMATS


def is_text_file(file) -> bool:
    """
    Check if the uploaded file is a text type.
    """
    ext = get_file_extension(file)
    return ext in SUPPORTED_TEXT_FORMATS


def safe_read(file: BinaryIO) -> str:
    """
    Safely read and decode a text-based file as UTF-8.
    Returns the content as a string.
    """
    try:
        content = file.read().decode('utf-8')
        logger.debug("File read and decoded successfully.")
        return content
    except UnicodeDecodeError:
        logger.error("❌ File is not valid UTF-8.")
        raise ValueError("Uploaded file is not a valid UTF-8 text file.")
    except Exception as e:
        logger.exception("❌ Failed to read uploaded file.")
        raise
