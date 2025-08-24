# modules/input_handler.py
import re
from typing import List, Dict, Optional, Union, NamedTuple
from dataclasses import dataclass
from modules.logger import logger
from modules.file_utils import read_text_file_smart, analyze_file, FileType

# Try to import proper subtitle libraries
try:
    import pysrt
    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False
    logger.warning("⚠️ pysrt not available, falling back to regex parsing for SRT files")

try:
    import webvtt
    HAS_WEBVTT = True  
except ImportError:
    HAS_WEBVTT = False
    logger.warning("⚠️ webvtt-py not available, falling back to regex parsing for VTT files")

@dataclass
class SubtitleEntry:
    """Structured subtitle entry with timing and optional speaker info."""
    start_time: str  # Original format (e.g., "00:01:23,456" or "00:01:23.456")
    end_time: str
    text: str
    speaker: Optional[str] = None
    start_seconds: Optional[float] = None  # Converted to seconds for easier processing
    end_seconds: Optional[float] = None
    index: Optional[int] = None  # Original subtitle index

    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.start_seconds is not None and self.end_seconds is not None:
            return self.end_seconds - self.start_seconds
        return 0.0

@dataclass
class ParsedSubtitles:
    """Complete parsed subtitle data with both structured and plain text."""
    entries: List[SubtitleEntry]
    plain_text: str
    total_duration: float
    entry_count: int
    detected_speakers: List[str]
    
    def get_text_by_timerange(self, start_seconds: float, end_seconds: float) -> str:
        """Extract text within a specific time range."""
        matching_entries = [
            entry for entry in self.entries
            if entry.start_seconds is not None and entry.end_seconds is not None
            and entry.start_seconds >= start_seconds and entry.end_seconds <= end_seconds
        ]
        return " ".join(entry.text for entry in matching_entries)

def time_to_seconds(time_str: str) -> float:
    """Convert subtitle time format to seconds."""
    # Handle both SRT format (00:01:23,456) and VTT format (00:01:23.456)
    time_str = time_str.replace(',', '.')
    
    try:
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid time format: {time_str}")
        
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError) as e:
        logger.warning(f"⚠️ Could not parse time '{time_str}': {e}")
        return 0.0

def detect_speaker_from_text(text: str) -> tuple[Optional[str], str]:
    """
    Detect speaker labels from subtitle text.
    Common patterns: "Speaker 1: Hello", "[John]: Hello", "MARY: Hello"
    """
    # Pattern for speaker labels
    speaker_patterns = [
        r'^([A-Z][A-Z\s]+):\s*(.+)$',  # "JOHN DOE: text"
        r'^\[([^\]]+)\]:\s*(.+)$',     # "[Speaker]: text"  
        r'^([^:]+):\s*(.+)$',          # "Name: text"
        r'^\-\s*([^:]+):\s*(.+)$',     # "- Speaker: text"
    ]
    
    for pattern in speaker_patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            speaker = match.group(1).strip()
            clean_text = match.group(2).strip()
            return speaker, clean_text
    
    return None, text

def parse_srt_with_pysrt(content: str) -> ParsedSubtitles:
    """Parse SRT using pysrt library."""
    try:
        import tempfile
        import os
        
        # pysrt expects a file, so create a temporary one
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            subtitles = pysrt.open(tmp_path, encoding='utf-8')
            entries = []
            detected_speakers = set()
            
            for subtitle in subtitles:
                # Convert pysrt time to string format
                start_time = f"{subtitle.start.hours:02d}:{subtitle.start.minutes:02d}:{subtitle.start.seconds:02d},{subtitle.start.milliseconds:03d}"
                end_time = f"{subtitle.end.hours:02d}:{subtitle.end.minutes:02d}:{subtitle.end.seconds:02d},{subtitle.end.milliseconds:03d}"
                
                # Clean up text (remove HTML-like tags, extra whitespace)
                text = re.sub(r'<[^>]+>', '', subtitle.text)
                text = ' '.join(text.split())  # Normalize whitespace
                
                # Detect speaker
                speaker, clean_text = detect_speaker_from_text(text)
                if speaker:
                    detected_speakers.add(speaker)
                
                # Convert times to seconds
                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)
                
                entry = SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=clean_text,
                    speaker=speaker,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    index=subtitle.index
                )
                entries.append(entry)
            
            # Generate plain text
            plain_text = " ".join(entry.text for entry in entries)
            total_duration = entries[-1].end_seconds if entries else 0.0
            
            return ParsedSubtitles(
                entries=entries,
                plain_text=plain_text,
                total_duration=total_duration,
                entry_count=len(entries),
                detected_speakers=list(detected_speakers)
            )
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.warning(f"⚠️ pysrt parsing failed: {e}, falling back to regex")
        return parse_srt_with_regex(content)

def parse_vtt_with_webvtt(content: str) -> ParsedSubtitles:
    """Parse VTT using webvtt-py library."""
    try:
        import tempfile
        import os
        
        # webvtt expects a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            vtt = webvtt.read(tmp_path)
            entries = []
            detected_speakers = set()
            
            for i, caption in enumerate(vtt):
                # Clean up text
                text = ' '.join(caption.text.split())  # Normalize whitespace
                
                # Detect speaker
                speaker, clean_text = detect_speaker_from_text(text)
                if speaker:
                    detected_speakers.add(speaker)
                
                # Convert times to seconds
                start_seconds = time_to_seconds(caption.start)
                end_seconds = time_to_seconds(caption.end)
                
                entry = SubtitleEntry(
                    start_time=caption.start,
                    end_time=caption.end,
                    text=clean_text,
                    speaker=speaker,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    index=i + 1
                )
                entries.append(entry)
            
            # Generate plain text
            plain_text = " ".join(entry.text for entry in entries)
            total_duration = entries[-1].end_seconds if entries else 0.0
            
            return ParsedSubtitles(
                entries=entries,
                plain_text=plain_text,
                total_duration=total_duration,
                entry_count=len(entries),
                detected_speakers=list(detected_speakers)
            )
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.warning(f"⚠️ webvtt parsing failed: {e}, falling back to regex")
        return parse_vtt_with_regex(content)

def parse_srt_with_regex(content: str) -> ParsedSubtitles:
    """Fallback SRT parsing using regex (enhanced version of original)."""
    try:
        lines = content.splitlines()
        entries = []
        detected_speakers = set()
        current_entry = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if line is a subtitle index
            if line.isdigit():
                current_entry = {"index": int(line)}
                i += 1
                continue
            
            # Check if line is a timestamp
            time_pattern = r'^(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})$'
            time_match = re.match(time_pattern, line)
            
            if time_match:
                current_entry.update({
                    "start_time": time_match.group(1),
                    "end_time": time_match.group(2)
                })
                
                # Collect text lines until next empty line or end
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                # Join and clean text
                text = " ".join(text_lines)
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                
                # Detect speaker
                speaker, clean_text = detect_speaker_from_text(text)
                if speaker:
                    detected_speakers.add(speaker)
                
                # Create entry
                start_seconds = time_to_seconds(current_entry["start_time"])
                end_seconds = time_to_seconds(current_entry["end_time"])
                
                entry = SubtitleEntry(
                    start_time=current_entry["start_time"],
                    end_time=current_entry["end_time"],
                    text=clean_text,
                    speaker=speaker,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    index=current_entry.get("index")
                )
                entries.append(entry)
                continue
            
            i += 1
        
        # Generate plain text
        plain_text = " ".join(entry.text for entry in entries)
        total_duration = entries[-1].end_seconds if entries else 0.0
        
        return ParsedSubtitles(
            entries=entries,
            plain_text=plain_text,
            total_duration=total_duration,
            entry_count=len(entries),
            detected_speakers=list(detected_speakers)
        )
        
    except Exception as e:
        logger.exception("❌ Failed to parse SRT with regex")
        raise ValueError("Could not parse .srt file") from e

def parse_vtt_with_regex(content: str) -> ParsedSubtitles:
    """Fallback VTT parsing using regex (enhanced version of original)."""
    try:
        lines = content.splitlines()
        entries = []
        detected_speakers = set()
        
        i = 0
        entry_index = 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip WEBVTT header, empty lines, and NOTE lines
            if (not line or line.startswith("WEBVTT") or 
                line.startswith("NOTE") or line.startswith("STYLE")):
                i += 1
                continue
            
            # Check if line is a timestamp
            time_pattern = r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}).*$'
            time_match = re.match(time_pattern, line)
            
            if time_match:
                start_time = time_match.group(1)
                end_time = time_match.group(2)
                
                # Collect text lines until next empty line or timestamp
                text_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:  # Empty line ends the entry
                        break
                    if re.match(time_pattern, next_line):  # Next timestamp
                        i -= 1  # Back up so we process this timestamp next
                        break
                    text_lines.append(next_line)
                    i += 1
                
                # Join and clean text
                text = " ".join(text_lines)
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/WebVTT tags
                
                # Detect speaker
                speaker, clean_text = detect_speaker_from_text(text)
                if speaker:
                    detected_speakers.add(speaker)
                
                # Create entry
                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)
                
                entry = SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=clean_text,
                    speaker=speaker,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    index=entry_index
                )
                entries.append(entry)
                entry_index += 1
            
            i += 1
        
        # Generate plain text
        plain_text = " ".join(entry.text for entry in entries)
        total_duration = entries[-1].end_seconds if entries else 0.0
        
        return ParsedSubtitles(
            entries=entries,
            plain_text=plain_text,
            total_duration=total_duration,
            entry_count=len(entries),
            detected_speakers=list(detected_speakers)
        )
        
    except Exception as e:
        logger.exception("❌ Failed to parse VTT with regex")
        raise ValueError("Could not parse .vtt file") from e

def read_text_file(file) -> str:
    """Read plain text transcript file with smart reading strategy."""
    try:
        file_info = analyze_file(file)
        
        if file_info.file_type != FileType.TXT:
            logger.warning(f"⚠️ Expected .txt file, got {file_info.file_type.name}")
        
        # Use smart reading (streaming for large files)
        content = read_text_file_smart(file)
        
        # If we got a generator (streaming), join it
        if hasattr(content, '__next__'):  # It's a generator
            logger.info("📖 Joining streamed text content")
            content = "".join(content)
        
        logger.info(f"✅ Read text file: {len(content)} characters")
        return content
        
    except Exception as e:
        logger.error("❌ Failed to read text file")
        raise

def parse_vtt(file) -> ParsedSubtitles:
    """Parse a .vtt subtitle file with structured output."""
    try:
        file_info = analyze_file(file)
        logger.info(f"📄 Parsing VTT file: {file_info.size_human}")
        
        # Use smart reading
        content = read_text_file_smart(file)
        if hasattr(content, '__next__'):
            content = "".join(content)
        
        # Use proper library if available, otherwise regex
        if HAS_WEBVTT:
            logger.info("🔧 Using webvtt-py library")
            result = parse_vtt_with_webvtt(content)
        else:
            logger.info("🔧 Using regex fallback")
            result = parse_vtt_with_regex(content)
        
        logger.info(f"✅ VTT parsed: {result.entry_count} entries, {result.total_duration:.1f}s duration")
        if result.detected_speakers:
            logger.info(f"🗣️ Speakers detected: {', '.join(result.detected_speakers)}")
        
        return result
        
    except Exception as e:
        logger.exception("❌ Failed to parse VTT file")
        raise ValueError("Could not parse .vtt file") from e

def parse_srt(file) -> ParsedSubtitles:
    """Parse a .srt subtitle file with structured output."""
    try:
        file_info = analyze_file(file)
        logger.info(f"📄 Parsing SRT file: {file_info.size_human}")
        
        # Use smart reading
        content = read_text_file_smart(file)
        if hasattr(content, '__next__'):
            content = "".join(content)
        
        # Use proper library if available, otherwise regex
        if HAS_PYSRT:
            logger.info("🔧 Using pysrt library")
            result = parse_srt_with_pysrt(content)
        else:
            logger.info("🔧 Using regex fallback")
            result = parse_srt_with_regex(content)
        
        logger.info(f"✅ SRT parsed: {result.entry_count} entries, {result.total_duration:.1f}s duration")
        if result.detected_speakers:
            logger.info(f"🗣️ Speakers detected: {', '.join(result.detected_speakers)}")
        
        return result
        
    except Exception as e:
        logger.exception("❌ Failed to parse SRT file")
        raise ValueError("Could not parse .srt file") from e

# Legacy compatibility functions
def parse_vtt_legacy(file) -> str:
    """Legacy function that returns only plain text."""
    result = parse_vtt(file)
    return result.plain_text

def parse_srt_legacy(file) -> str:
    """Legacy function that returns only plain text.""" 
    result = parse_srt(file)
    return result.plain_text

def transcribe_audio(file):
    """Deprecated placeholder - use transcriber.py instead."""
    raise NotImplementedError("Use transcriber.py for audio transcription.")

# Utility functions
def merge_subtitle_entries(entries: List[SubtitleEntry], max_gap_seconds: float = 2.0) -> List[SubtitleEntry]:
    """Merge consecutive entries from the same speaker with small gaps."""
    if not entries:
        return entries
    
    merged = []
    current = entries[0]
    
    for next_entry in entries[1:]:
        # Check if we should merge
        should_merge = (
            current.speaker == next_entry.speaker and  # Same speaker (or both None)
            current.end_seconds is not None and
            next_entry.start_seconds is not None and
            (next_entry.start_seconds - current.end_seconds) <= max_gap_seconds
        )
        
        if should_merge:
            # Merge entries
            current = SubtitleEntry(
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=current.text + " " + next_entry.text,
                speaker=current.speaker,
                start_seconds=current.start_seconds,
                end_seconds=next_entry.end_seconds,
                index=current.index
            )
        else:
            merged.append(current)
            current = next_entry
    
    merged.append(current)
    return merged

def format_subtitle_summary(parsed: ParsedSubtitles) -> str:
    """Generate a human-readable summary of parsed subtitles."""
    lines = [
        f"📊 Subtitle Summary:",
        f"  • {parsed.entry_count} entries",
        f"  • {parsed.total_duration:.1f} seconds duration", 
        f"  • {len(parsed.plain_text)} characters of text"
    ]
    
    if parsed.detected_speakers:
        lines.append(f"  • Speakers: {', '.join(parsed.detected_speakers)}")
    
    return "\n".join(lines)