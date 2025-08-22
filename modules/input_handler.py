# modules/input_handler.py

import re
from modules.logger import logger
from modules.file_utils import safe_read


def read_text_file(file) -> str:
    """Read plain text transcript file with UTF-8 decoding."""
    try:
        content = safe_read(file)
        return content
    except Exception as e:
        logger.error("❌ Failed to read text file.")
        raise


def parse_vtt(file) -> str:
    """Parse a .vtt subtitle file and extract plain transcript text."""
    try:
        content = safe_read(file)
        lines = content.splitlines()
        text_lines = []

        for line in lines:
            # Skip timestamps like 00:00:01.000 --> 00:00:04.000
            if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$", line):
                continue
            if line.strip() == '' or line.strip().isdigit() or line.startswith("WEBVTT"):
                continue
            text_lines.append(line)

        return " ".join(text_lines)

    except Exception as e:
        logger.exception("❌ Failed to parse VTT file.")
        raise ValueError("Could not parse .vtt file.") from e


def parse_srt(file) -> str:
    """Parse a .srt subtitle file and extract plain transcript text."""
    try:
        content = safe_read(file)
        lines = content.splitlines()
        text_lines = []

        for line in lines:
            # Skip timestamps like 00:00:01,000 --> 00:00:04,000
            if re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", line):
                continue
            if line.strip() == '' or line.strip().isdigit():
                continue
            text_lines.append(line)

        return " ".join(text_lines)

    except Exception as e:
        logger.exception("❌ Failed to parse SRT file.")
        raise ValueError("Could not parse .srt file.") from e


def transcribe_audio(file):
    """Deprecated placeholder."""
    raise NotImplementedError("Use transcriber.py for audio transcription.")
