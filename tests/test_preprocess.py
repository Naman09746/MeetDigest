# tests/test_preprocess.py

import pytest
from modules import preprocessor


def test_clean_transcript_removes_filler_and_timestamps():
    raw = """
    00:00:00,000 --> 00:00:05,000
    Speaker 1: Um, I think we should like review the plan.
    """
    cleaned = preprocessor.clean_transcript(raw)
    assert "um" not in cleaned.lower()
    assert "like" not in cleaned.lower()
    assert "00:00" not in cleaned

def test_segment_by_speaker():
    text = "Speaker 1: Hello there. Speaker 2: Hi, how are you?"
    segments = preprocessor.segment_by_speaker(text)
    assert isinstance(segments, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in segments)

def test_chunk_text_token_limit():
    text = "This is a test sentence. " * 50  # ~250 words
    chunks = preprocessor.chunk_text(text, max_tokens=50)
    assert all(len(chunk.split()) <= 50 for chunk in chunks)
    assert len(chunks) > 1
