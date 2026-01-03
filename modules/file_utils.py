"""
modules/file_utils.py

Utility functions for handling text-based meeting inputs.
This module is intentionally kept simple and robust, as Streamlit
uploads already load files into memory.
"""

import logging
from typing import Optional

import pysrt
import webvtt

# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Text File Reader
# ---------------------------------------------------------------------

def read_text_file(file) -> str:
    """
    Read a plain text (.txt) file uploaded via Streamlit.

    Parameters
    ----------
    file : UploadedFile
        Streamlit uploaded file object

    Returns
    -------
    str
        Full text content of the file
    """
    try:
        content = file.read().decode("utf-8")

        if not content.strip():
            logger.warning("⚠️ Text file is empty")

        logger.info(f"✅ Read text file: {len(content)} characters")
        return content

    except UnicodeDecodeError:
        logger.exception("❌ Text file encoding error")
        raise ValueError(
            "Text file encoding not supported. Please upload a UTF-8 encoded file."
        )

    except Exception as e:
        logger.exception("❌ Failed to read text file")
        raise ValueError("Could not read text file") from e


# ---------------------------------------------------------------------
# SRT Subtitle Parser
# ---------------------------------------------------------------------

def parse_srt(file) -> str:
    """
    Parse a .srt subtitle file into plain text.

    Parameters
    ----------
    file : UploadedFile
        Streamlit uploaded file object

    Returns
    -------
    str
        Combined subtitle text
    """
    try:
        raw_content = file.read().decode("utf-8")
        subtitles = pysrt.from_string(raw_content)

        text = " ".join(sub.text for sub in subtitles)

        if not text.strip():
            logger.warning("⚠️ SRT file parsed but contains no text")

        logger.info(f"✅ Parsed SRT file: {len(text)} characters")
        return text

    except Exception as e:
        logger.exception("❌ Failed to parse SRT file")
        raise ValueError("Could not parse .srt file") from e


# ---------------------------------------------------------------------
# VTT Subtitle Parser
# ---------------------------------------------------------------------

def parse_vtt(file) -> str:
    """
    Parse a .vtt subtitle file into plain text.

    Parameters
    ----------
    file : UploadedFile
        Streamlit uploaded file object

    Returns
    -------
    str
        Combined subtitle text
    """
    try:
        captions = webvtt.read_buffer(file)

        text = " ".join(caption.text for caption in captions)

        if not text.strip():
            logger.warning("⚠️ VTT file parsed but contains no text")

        logger.info(f"✅ Parsed VTT file: {len(text)} characters")
        return text

    except Exception as e:
        logger.exception("❌ Failed to parse VTT file")
        raise ValueError("Could not parse .vtt file") from e


# ---------------------------------------------------------------------
# Optional helper (used by app logic)
# ---------------------------------------------------------------------

def is_supported_text_extension(extension: str) -> bool:
    """
    Check if file extension is supported for text parsing.

    Parameters
    ----------
    extension : str
        File extension without dot

    Returns
    -------
    bool
    """
    return extension.lower() in {"txt", "srt", "vtt"}
