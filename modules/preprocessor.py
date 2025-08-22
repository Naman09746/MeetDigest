# modules/preprocessor.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
from modules.logger import logger

# Download tokenizer if not already
nltk.download('punkt', quiet=True)

# Optional filler words to remove (customizable)
FILLER_WORDS = [
    "um", "uh", "you know", "like", "I mean", "so", "well", "okay", "right", "hmm"
]

# Regex patterns (precompiled for speed)
SRT_TIMESTAMP = re.compile(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}')
VTT_TIMESTAMP = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}')
FILLER_PATTERN = re.compile(rf'\b({"|".join(map(re.escape, FILLER_WORDS))})\b', re.IGNORECASE)
SPEAKER_PATTERN = re.compile(r'(Speaker\s*\d+:|[A-Z][a-z]+:)', re.IGNORECASE)


def clean_transcript(text: str) -> str:
    """
    Clean transcript by removing timestamps, filler words, and excess whitespace.
    """
    if not text.strip():
        logger.warning("⚠️ Received empty transcript for cleaning.")
        return ""

    try:
        # Remove timestamps
        text = SRT_TIMESTAMP.sub('', text)
        text = VTT_TIMESTAMP.sub('', text)

        # Remove filler words
        text = FILLER_PATTERN.sub('', text)

        # Normalize whitespace, remove standalone numbers, and strip
        text = re.sub(r'\n+', '\n', text)                # Collapse multiple newlines
        text = re.sub(r'\s+', ' ', text)                 # Normalize whitespace
        text = re.sub(r'\b\d+\b', '', text)              # Remove standalone numbers
        text = text.encode('ascii', 'ignore').decode()   # Remove non-ASCII noise
        cleaned = text.strip()

        logger.info("✅ Transcript cleaned successfully.")
        return cleaned

    except Exception as e:
        logger.exception("❌ Failed to clean transcript.")
        return text


def segment_by_speaker(text: str) -> List[Tuple[str, str]]:
    """
    Segment transcript into speaker-utterance pairs.
    Returns a list of (speaker, utterance) tuples.
    """
    if not text.strip():
        return [("Unknown", "")]

    try:
        segments = SPEAKER_PATTERN.split(text)
        structured = []

        for i in range(1, len(segments), 2):
            speaker = segments[i].strip().rstrip(':')
            utterance = segments[i + 1].strip() if i + 1 < len(segments) else ''
            if utterance:
                structured.append((speaker, utterance))

        if not structured:
            logger.warning("⚠️ No speaker segmentation found, returning entire transcript.")
            return [("Unknown", text)]

        logger.info(f"🎤 Segmented transcript into {len(structured)} speaker blocks.")
        return structured

    except Exception as e:
        logger.exception("❌ Failed to segment speakers.")
        return [("Unknown", text)]


def chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """
    Split transcript into sentence-based chunks of ~max_tokens.
    Ensures semantic completeness.
    """
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            token_estimate = len(sent.split())  # You can replace with actual token counter if needed

            if current_length + token_estimate > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_length = token_estimate
            else:
                current_chunk.append(sent)
                current_length += token_estimate

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"🧩 Split transcript into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logger.exception("❌ Failed to chunk transcript.")
        return [text]
