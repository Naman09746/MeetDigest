# tests/test_transcriber.py

import pytest
from unittest.mock import patch, MagicMock
from modules import transcriber
from unittest import mock
import sys

# Mock whisper before importing transcriber
sys.modules['whisper'] = mock.MagicMock()

from modules import transcriber
class DummyFile:
    def __init__(self, data=b"FAKE AUDIO"):
        self.data = data

    def read(self):
        return self.data

@patch("modules.transcriber.load_whisper_model")
def test_transcribe_audio_success(mock_model_loader):
    dummy_model = MagicMock()
    dummy_model.transcribe.return_value = {"text": "Hello world"}
    mock_model_loader.return_value = dummy_model

    dummy_file = DummyFile()

    text = transcriber.transcribe_audio(dummy_file, extension="mp3", model_name="base")
    assert "Hello world" in text
    mock_model_loader.assert_called_once()

@patch("modules.transcriber.load_whisper_model")
def test_transcribe_audio_unsupported_format(mock_model_loader):
    dummy_file = DummyFile()
    with pytest.raises(ValueError):
        transcriber.transcribe_audio(dummy_file, extension="flac")
