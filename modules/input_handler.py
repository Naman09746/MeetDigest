# modules/input_handler.py

import re
from typing import List, Optional
from dataclasses import dataclass
from modules.logger import logger

# ---------------------------------------------------------
# Optional subtitle libraries
# ---------------------------------------------------------

try:
    import pysrt
    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False
    logger.warning("⚠️ pysrt not available, falling back to regex parsing for SRT")

try:
    import webvtt
    HAS_WEBVTT = True
except ImportError:
    HAS_WEBVTT = False
    logger.warning("⚠️ webvtt-py not available, falling back to regex parsing for VTT")


# ---------------------------------------------------------
# Data structures
# ---------------------------------------------------------

@dataclass
class SubtitleEntry:
    start_time: str
    end_time: str
    text: str
    speaker: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    index: Optional[int] = None

    def duration_seconds(self) -> float:
        if self.start_seconds is not None and self.end_seconds is not None:
            return self.end_seconds - self.start_seconds
        return 0.0


@dataclass
class ParsedSubtitles:
    entries: List[SubtitleEntry]
    plain_text: str
    total_duration: float
    entry_count: int
    detected_speakers: List[str]


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def _read_uploaded_file(file) -> str:
    """
    Read Streamlit uploaded file safely as UTF-8 text.
    """
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        logger.exception("❌ Failed to read uploaded file")
        raise ValueError("Could not read uploaded file") from e


def time_to_seconds(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    try:
        h, m, s = time_str.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return 0.0


def detect_speaker_from_text(text: str):
    patterns = [
        r'^([A-Z][A-Z\s]+):\s*(.+)$',
        r'^\[([^\]]+)\]:\s*(.+)$',
        r'^([^:]+):\s*(.+)$',
        r'^\-\s*([^:]+):\s*(.+)$'
    ]

    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    return None, text


# ---------------------------------------------------------
# TXT
# ---------------------------------------------------------

def read_text_file(file) -> str:
    """
    Read plain text transcript.
    """
    content = _read_uploaded_file(file)
    logger.info(f"✅ Read TXT file: {len(content)} characters")
    return content


# ---------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------

def parse_srt(file) -> ParsedSubtitles:
    content = _read_uploaded_file(file)

    if HAS_PYSRT:
        return parse_srt_with_pysrt(content)

    logger.info("🔧 Falling back to regex-based SRT parsing")
    return parse_srt_with_regex(content)


def parse_srt_with_pysrt(content: str) -> ParsedSubtitles:
    subs = pysrt.from_string(content)
    entries = []
    speakers = set()

    for sub in subs:
        text = re.sub(r'<[^>]+>', '', sub.text).strip()
        speaker, clean_text = detect_speaker_from_text(text)
        if speaker:
            speakers.add(speaker)

        start = f"{sub.start.hours:02d}:{sub.start.minutes:02d}:{sub.start.seconds:02d},{sub.start.milliseconds:03d}"
        end = f"{sub.end.hours:02d}:{sub.end.minutes:02d}:{sub.end.seconds:02d},{sub.end.milliseconds:03d}"

        entry = SubtitleEntry(
            start_time=start,
            end_time=end,
            text=clean_text,
            speaker=speaker,
            start_seconds=time_to_seconds(start),
            end_seconds=time_to_seconds(end),
            index=sub.index
        )
        entries.append(entry)

    plain_text = " ".join(e.text for e in entries)
    total_duration = entries[-1].end_seconds if entries else 0.0

    return ParsedSubtitles(
        entries=entries,
        plain_text=plain_text,
        total_duration=total_duration,
        entry_count=len(entries),
        detected_speakers=list(speakers)
    )


def parse_srt_with_regex(content: str) -> ParsedSubtitles:
    lines = content.splitlines()
    entries = []
    speakers = set()

    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            start, end = lines[i].split("-->")
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines)
            speaker, clean_text = detect_speaker_from_text(text)
            if speaker:
                speakers.add(speaker)

            entry = SubtitleEntry(
                start_time=start.strip(),
                end_time=end.strip(),
                text=clean_text,
                speaker=speaker,
                start_seconds=time_to_seconds(start),
                end_seconds=time_to_seconds(end)
            )
            entries.append(entry)
        i += 1

    plain_text = " ".join(e.text for e in entries)
    total_duration = entries[-1].end_seconds if entries else 0.0

    return ParsedSubtitles(entries, plain_text, total_duration, len(entries), list(speakers))


# ---------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------

def parse_vtt(file) -> ParsedSubtitles:
    content = _read_uploaded_file(file)

    if HAS_WEBVTT:
        return parse_vtt_with_webvtt(content)

    logger.info("🔧 Falling back to regex-based VTT parsing")
    return parse_vtt_with_regex(content)


def parse_vtt_with_webvtt(content: str) -> ParsedSubtitles:
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name

    try:
        vtt = webvtt.read(path)
        entries = []
        speakers = set()

        for i, cap in enumerate(vtt):
            speaker, clean_text = detect_speaker_from_text(cap.text)
            if speaker:
                speakers.add(speaker)

            entry = SubtitleEntry(
                start_time=cap.start,
                end_time=cap.end,
                text=clean_text,
                speaker=speaker,
                start_seconds=time_to_seconds(cap.start),
                end_seconds=time_to_seconds(cap.end),
                index=i + 1
            )
            entries.append(entry)

        plain_text = " ".join(e.text for e in entries)
        total_duration = entries[-1].end_seconds if entries else 0.0

        return ParsedSubtitles(entries, plain_text, total_duration, len(entries), list(speakers))

    finally:
        os.remove(path)


def parse_vtt_with_regex(content: str) -> ParsedSubtitles:
    lines = content.splitlines()
    entries = []
    speakers = set()

    for line in lines:
        if "-->" in line:
            start, end = line.split("-->")
            continue
        if line.strip():
            speaker, clean_text = detect_speaker_from_text(line)
            if speaker:
                speakers.add(speaker)

            entry = SubtitleEntry(
                start_time=start.strip(),
                end_time=end.strip(),
                text=clean_text,
                speaker=speaker,
                start_seconds=time_to_seconds(start),
                end_seconds=time_to_seconds(end)
            )
            entries.append(entry)

    plain_text = " ".join(e.text for e in entries)
    total_duration = entries[-1].end_seconds if entries else 0.0

    return ParsedSubtitles(entries, plain_text, total_duration, len(entries), list(speakers))


# ---------------------------------------------------------
# Legacy helpers (unchanged)
# ---------------------------------------------------------

def parse_srt_legacy(file) -> str:
    return parse_srt(file).plain_text


def parse_vtt_legacy(file) -> str:
    return parse_vtt(file).plain_text
