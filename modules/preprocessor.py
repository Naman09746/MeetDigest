# modules/preprocessor.py
import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from modules.logger import logger
from modules.date_utils import normalize_dates
import streamlit as st
from unidecode import unidecode
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spaCy model for better sentence segmentation
# @st.cache_resource
# def load_spacy_model():
#     try:
#         return spacy.load("en_core_web_sm")
#     except OSError:
#         logger.warning("spaCy model not found, falling back to NLTK")
#         return None
    


# Instead of using Streamlit cache inside the core module
from functools import lru_cache

@lru_cache(maxsize=1)
def load_spacy_model():
    """Load spaCy model once and cache in memory."""
    return spacy.load("en_core_web_sm")

import spacy
nlp = load_spacy_model()

class OutputFormat(Enum):
    RAW_TEXT = "raw_text"
    SENTENCE_LIST = "sentence_list"
    SPEAKER_TAGGED = "speaker_tagged"
    STRUCTURED = "structured"

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class CleaningConfig:
    """Configuration for text cleaning options."""
    
    remove_filler_words: bool = True
    remove_repetitions: bool = True
    normalize_numbers: bool = True
    normalize_dates: bool = True
    remove_timestamps: bool = True
    remove_diarization_noise: bool = True
    fix_speech_artifacts: bool = True
    standardize_punctuation: bool = True
    remove_non_ascii: bool = True
    preserve_speaker_info: bool = True
    min_sentence_length: int = 3

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_sentence_length < 1:
            raise ValueError("min_sentence_length must be >= 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleaningConfig":
        """Create a config from dictionary, with fallback to defaults."""
        return cls(**{**asdict(cls()), **data})
    
    def update(self, **kwargs) -> None:
        """Update config values dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config option: {key}")


@dataclass
class ProcessingResult:
    """Result object containing processed text and metadata"""
    cleaned_text: str
    sentence_list: List[str] = None
    speaker_segments: List[Tuple[str, str]] = None
    chunks: List[str] = None
    cleaning_stats: Dict = None
    processing_notes: List[str] = None
    
    def __post_init__(self):
        if self.sentence_list is None:
            self.sentence_list = []
        if self.speaker_segments is None:
            self.speaker_segments = []
        if self.chunks is None:
            self.chunks = []
        if self.cleaning_stats is None:
            self.cleaning_stats = {}
        if self.processing_notes is None:
            self.processing_notes = []

# Enhanced patterns for speech cleaning
SPEECH_PATTERNS = {
    # Filler words and hesitations
    'filler_words': [
        "um", "uh", "uhm", "er", "ah", "oh", "eh", "mm", "hmm", "hm",
        "you know", "i mean", "like", "so", "well", "okay", "right",
        "basically", "actually", "literally", "honestly", "obviously",
        "sort of", "kind of", "i guess", "let me see", "let's see"
    ],
    
    # Repetitions and stuttering
    'repetitions': [
        r'\b(\w+)\s+\1\b',  # Word repetitions: "the the", "and and"
        r'\b(\w)\1{3,}\b',  # Character repetitions: "sooo", "yeahhh"
        r'\b(\w{1,3})-\1\b'  # Stuttering: "th-th-the", "w-w-what"
    ],
    
    # Speech artifacts
    'artifacts': [
        r'\[.*?\]',  # [background noise], [inaudible], [laughter]
        r'\(.*?\)',  # (background noise), (coughing)
        r'<.*?>',    # <unclear>, <crosstalk>
        r'\*.*?\*',  # *noise*, *silence*
        r'--+',      # Multiple dashes
        r'\.{3,}',   # Multiple dots ...
        r'\?{2,}',   # Multiple question marks
        r'!{2,}',    # Multiple exclamation marks
    ],
    
    # Diarization noise
    'diarization_noise': [
        r'SPEAKER_?\d+:?',
        r'Speaker\s*\d+:?',
        r'spk_?\d+:?',
        r'S\d+:?',
        r'Unknown\s*Speaker:?',
        r'Male\s*Voice:?',
        r'Female\s*Voice:?'
    ],
    
    # Timestamps and technical artifacts
    'timestamps': [
        r'\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}',  # SRT/VTT
        r'\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}',  # Individual timestamps
        r'\[\d{1,2}:\d{2}:\d{2}\]',  # Bracketed timestamps
        r'<\d{1,2}:\d{2}:\d{2}>',   # Angle bracket timestamps
        r'\btimestamp:\s*\d+',       # Technical timestamp markers
        r'\d+\s*\n\d{2}:\d{2}:\d{2}',  # Subtitle numbering
    ]
}

# Number and date normalization patterns
NUMBER_PATTERNS = {
    'spelled_numbers': {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
        'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
        'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
    },
    'ordinals': {
        'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
        'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
        'ninth': '9th', 'tenth': '10th'
    }
}

# Compile patterns for performance
COMPILED_PATTERNS = {}
for category, patterns in SPEECH_PATTERNS.items():
    if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
        if category == 'filler_words':
            # Create word boundary pattern for filler words
            COMPILED_PATTERNS[category] = re.compile(
                r'\b(' + '|'.join(map(re.escape, patterns)) + r')\b', 
                re.IGNORECASE
            )
        else:
            # Compile as regex patterns
            COMPILED_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    else:
        COMPILED_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

class EnhancedPreprocessor:
    """Enhanced text preprocessor with comprehensive speech cleaning"""
    
    def __init__(self):
        self.nlp = nlp
        self.stats = {}
    
    def _track_stat(self, key: str, value: int = 1):
        """Track cleaning statistics"""
        self.stats[key] = self.stats.get(key, 0) + value
    
    def _clean_speech_artifacts(self, text: str, config: CleaningConfig) -> str:
        """Remove speech artifacts and diarization noise"""
        original_text = text
        
        # Remove timestamps
        if config.remove_timestamps:
            for pattern in COMPILED_PATTERNS['timestamps']:
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('timestamps_removed', matches)
                text = pattern.sub('', text)
        
        # Remove diarization noise
        if config.remove_diarization_noise:
            for pattern in COMPILED_PATTERNS['diarization_noise']:
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('diarization_noise_removed', matches)
                text = pattern.sub('', text)
        
        # Remove speech artifacts
        if config.fix_speech_artifacts:
            for pattern in COMPILED_PATTERNS['artifacts']:
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('artifacts_removed', matches)
                text = pattern.sub('', text)
        
        # Remove filler words
        if config.remove_filler_words:
            matches = len(COMPILED_PATTERNS['filler_words'].findall(text))
            if matches > 0:
                self._track_stat('filler_words_removed', matches)
            text = COMPILED_PATTERNS['filler_words'].sub('', text)
        
        # Fix repetitions
        if config.remove_repetitions:
            for pattern in COMPILED_PATTERNS['repetitions']:
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('repetitions_fixed', matches)
                text = pattern.sub(r'\1', text)  # Keep only one instance
        
        return text
    
    def _normalize_numbers_and_dates(self, text: str, config: CleaningConfig) -> str:
        """Normalize numbers and dates"""
        if not (config.normalize_numbers or config.normalize_dates):
            return text
        
        # Normalize spelled-out numbers
        if config.normalize_numbers:
            for word, number in NUMBER_PATTERNS['spelled_numbers'].items():
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('numbers_normalized', matches)
                text = pattern.sub(number, text)
            
            # Normalize ordinals
            for word, ordinal in NUMBER_PATTERNS['ordinals'].items():
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                matches = len(pattern.findall(text))
                if matches > 0:
                    self._track_stat('ordinals_normalized', matches)
                text = pattern.sub(ordinal, text)
        
        # Normalize dates using date_utils
        if config.normalize_dates:
            try:
                # This would integrate with your date_utils module
                normalized_text = normalize_dates(text)
                if normalized_text != text:
                    self._track_stat('dates_normalized', 1)
                text = normalized_text
            except Exception as e:
                logger.warning(f"Date normalization failed: {e}")
        
        return text
    
    def _standardize_punctuation_and_spacing(self, text: str, config: CleaningConfig) -> str:
        """Standardize punctuation and spacing"""
        if not config.standardize_punctuation:
            return text
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        
        # Fix quotes
        text = re.sub(r'[""]', '"', text)  # Normalize quotes
        text = re.sub(r"['']", "'", text)  # Normalize apostrophes
        
        # Fix multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
        # Remove trailing punctuation duplicates
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text
    
    

    def _remove_non_ascii_and_cleanup(self, text: str, config: CleaningConfig) -> str:
        """Final cleanup and non-ASCII handling (robust)."""
    
        if config.remove_non_ascii:
            # First transliterate accented characters into ASCII
            # e.g., "José" → "Jose", "Müller" → "Muller"
            text = unidecode(text)

            # Remove emojis & unusual symbols, but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:()\'"-]', '', text)

            self._track_stat('non_ascii_removed', 1)
    
        # Normalize whitespace (collapse multiple spaces/newlines)
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text

    
    def _segment_sentences(self, text: str, config: CleaningConfig) -> List[str]:
        """Segment text into sentences using spaCy or NLTK"""
        try:
            if self.nlp:
                # Use spaCy for better sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                # Fallback to NLTK
                sentences = sent_tokenize(text)
            
            # Filter short sentences if configured
            if config.min_sentence_length > 0:
                sentences = [s for s in sentences if len(s.split()) >= config.min_sentence_length]
                
            return sentences
            
        except Exception as e:
            logger.warning(f"Sentence segmentation failed: {e}")
            # Fallback: split on common sentence endings
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _extract_speaker_segments(self, text: str, config: CleaningConfig) -> List[Tuple[str, str]]:
        """Extract speaker segments with improved patterns"""
        if not config.preserve_speaker_info:
            return [("Unknown", text)]
        
        # Enhanced speaker patterns
        speaker_patterns = [
            r'^([A-Z][a-zA-Z\s]+):\s*(.+)$',  # "John Smith: Hello there"
            r'^([A-Z]+):\s*(.+)$',             # "JOHN: Hello there"
            r'^(\w+):\s*(.+)$',                # "john: Hello there"
            r'(Speaker\s*\d+):\s*(.+)',        # "Speaker 1: Hello there"
            r'(Male|Female|Voice\s*\d+):\s*(.+)'  # "Male: Hello there"
        ]
        
        segments = []
        lines = text.split('\n')
        current_speaker = "Unknown"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker patterns
            speaker_found = False
            for pattern in speaker_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous segment
                    if current_text:
                        segments.append((current_speaker, ' '.join(current_text)))
                        current_text = []
                    
                    # Start new segment
                    current_speaker = match.group(1).strip()
                    current_text = [match.group(2).strip()]
                    speaker_found = True
                    break
            
            if not speaker_found:
                current_text.append(line)
        
        # Add final segment
        if current_text:
            segments.append((current_speaker, ' '.join(current_text)))
        
        # If no speakers found, return entire text
        if not segments:
            return [("Unknown", text)]
        
        logger.info(f"🎤 Extracted {len(segments)} speaker segments")
        return segments
    
    def clean_transcript(self, text: str, config: CleaningConfig = None) -> ProcessingResult:
        """
        Comprehensive transcript cleaning with configurable options.
        More robust error handling and consistent pipeline.
        """
        if not text or not text.strip():
            logger.warning("⚠️ Received empty transcript for cleaning.")
            return ProcessingResult(
                cleaned_text="",
                processing_notes=["Empty input text"]
            )
    
        if config is None:
            config = CleaningConfig()
    
        # Reset stats
        self.stats = {}
        processing_notes = []
        original_length = len(text)

        try:
            # Stage 1: Clean speech artifacts
            try:
                logger.debug("🧹 Stage 1: Cleaning speech artifacts")
                text = self._clean_speech_artifacts(text, config)
            except Exception as e:
                logger.error(f"⚠️ Artifact cleaning failed: {e}")
                processing_notes.append("Artifact cleaning skipped")

            # Stage 2: Normalize numbers and dates
            try:
                logger.debug("🔢 Stage 2: Normalizing numbers and dates")
                text = self._normalize_numbers_and_dates(text, config)
            except Exception as e:
                logger.error(f"⚠️ Number/date normalization failed: {e}")
                processing_notes.append("Number/date normalization skipped")

            # Stage 3: Standardize punctuation and spacing
            try:
                logger.debug("✏️ Stage 3: Standardizing punctuation")
                text = self._standardize_punctuation_and_spacing(text, config)
            except Exception as e:
                logger.error(f"⚠️ Punctuation standardization failed: {e}")
                processing_notes.append("Punctuation standardization skipped")

            # Stage 4: Final cleanup (non-ASCII, whitespace)
            try:
                logger.debug("🔧 Stage 4: Final cleanup")
                cleaned_text = self._remove_non_ascii_and_cleanup(text, config)
            except Exception as e:
                logger.error(f"⚠️ Final cleanup failed: {e}")
                cleaned_text = text
                processing_notes.append("Final cleanup skipped")

            # Generate additional formats
            sentence_list, speaker_segments, chunks = [], [], []
            try:
                sentence_list = self._segment_sentences(cleaned_text, config)
            except Exception as e:
                logger.error(f"⚠️ Sentence segmentation failed: {e}")
                processing_notes.append("Sentence segmentation skipped")

            try:
                speaker_segments = self._extract_speaker_segments(cleaned_text, config)
            except Exception as e:
                logger.error(f"⚠️ Speaker extraction failed: {e}")
                processing_notes.append("Speaker extraction skipped")

            try:
                chunks = self._intelligent_chunk(cleaned_text)
            except Exception as e:
                logger.error(f"⚠️ Chunking failed: {e}")
                processing_notes.append("Chunking skipped")

            # Stats
            cleaning_stats = {
                **self.stats,
                'original_length': original_length,
                'cleaned_length': len(cleaned_text),
                'compression_ratio': len(cleaned_text) / max(original_length, 1),
                'sentences_extracted': len(sentence_list),
                'speaker_segments': len(speaker_segments),
                'chunks_created': len(chunks)
            }

            # Processing notes
            if cleaning_stats['compression_ratio'] < 0.5:
                processing_notes.append("⚠️ Significant compression - possible over-cleaning")
            if len(sentence_list) < 3:
                processing_notes.append("⚠️ Very few sentences extracted")
            if len(speaker_segments) > 1:
                processing_notes.append(f"Multi-speaker conversation detected ({len(speaker_segments)} speakers)")

            logger.info(f"✅ Transcript cleaned: {original_length} → {len(cleaned_text)} chars")

            return ProcessingResult(
                cleaned_text=cleaned_text,
                sentence_list=sentence_list,
                speaker_segments=speaker_segments,
                chunks=chunks,
                cleaning_stats=cleaning_stats,
                processing_notes=processing_notes
            )

        except Exception as e:
            logger.exception("❌ Cleaning pipeline crashed.")
            return ProcessingResult(
                cleaned_text=text,
                processing_notes=[f"Cleaning failed: {str(e)}"]
            )

    
    def _intelligent_chunk(self, text: str, max_tokens: int = 450) -> List[str]:
        """Create intelligent chunks with sentence boundaries"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                sentences = sent_tokenize(text)
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                # Estimate tokens (rough approximation: 1.3 * words)
                token_estimate = int(len(sent.split()) * 1.3)
                
                if current_length + token_estimate > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_length = token_estimate
                else:
                    current_chunk.append(sent)
                    current_length += token_estimate
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            logger.info(f"🧩 Created {len(chunks)} intelligent chunks")
            return chunks
            
        except Exception as e:
            logger.exception("❌ Failed to create chunks")
            return [text]

# Global preprocessor instance
_preprocessor_instance = None

def get_preprocessor() -> EnhancedPreprocessor:
    """Get singleton preprocessor instance"""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = EnhancedPreprocessor()
    return _preprocessor_instance

# Backward compatible functions
def clean_transcript(text: str) -> str:
    """
    Backward compatible function for existing code
    """
    preprocessor = get_preprocessor()
    result = preprocessor.clean_transcript(text)
    return result.cleaned_text

def segment_by_speaker(text: str) -> List[Tuple[str, str]]:
    """
    Backward compatible function for speaker segmentation
    """
    preprocessor = get_preprocessor()
    config = CleaningConfig(preserve_speaker_info=True)
    result = preprocessor.clean_transcript(text, config)
    return result.speaker_segments

def chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """
    Backward compatible function for text chunking
    """
    preprocessor = get_preprocessor()
    return preprocessor._intelligent_chunk(text, max_tokens)

# Enhanced API functions
def enhanced_clean_transcript(text: str, 
                            remove_filler_words: bool = True,
                            normalize_numbers: bool = True,
                            normalize_dates: bool = True,
                            output_format: str = "structured") -> Union[str, Dict, List]:
    """
    Enhanced cleaning with configurable options and multiple output formats
    """
    config = CleaningConfig(
        remove_filler_words=remove_filler_words,
        normalize_numbers=normalize_numbers,
        normalize_dates=normalize_dates
    )
    
    preprocessor = get_preprocessor()
    result = preprocessor.clean_transcript(text, config)
    
    try:
        format_enum = OutputFormat(output_format)
    except ValueError:
        format_enum = OutputFormat.STRUCTURED
    
    if format_enum == OutputFormat.RAW_TEXT:
        return result.cleaned_text
    elif format_enum == OutputFormat.SENTENCE_LIST:
        return result.sentence_list
    elif format_enum == OutputFormat.SPEAKER_TAGGED:
        return result.speaker_segments
    else:  # STRUCTURED
        return {
            'cleaned_text': result.cleaned_text,
            'sentences': result.sentence_list,
            'speaker_segments': result.speaker_segments,
            'chunks': result.chunks,
            'stats': result.cleaning_stats,
            'notes': result.processing_notes
        }

def get_cleaning_stats(text: str) -> Dict:
    """Get detailed cleaning statistics"""
    preprocessor = get_preprocessor()
    result = preprocessor.clean_transcript(text)
    return result.cleaning_stats

# Utility functions
def get_available_output_formats() -> List[str]:
    """Get available output formats"""
    return [fmt.value for fmt in OutputFormat]

def create_custom_config(**kwargs) -> CleaningConfig:
    """Create custom cleaning configuration"""
    return CleaningConfig(**kwargs)


# 🔎 Strengths of Your Current Code
# Config-driven cleaning with CleaningConfig → makes the pipeline modular.
# Pipeline stages clearly separated (artifacts → numbers/dates → punctuation → cleanup → segmentation → speaker extraction → chunking).
# Error handling & logging at every stage → resilient.
# Stats tracking → great for debugging and analytics.
# Multiple output formats via OutputFormat + enhanced_clean_transcript.
# Speaker-aware segmentation with fallback.
# Singleton preprocessor instance → avoids reloading spaCy.
# This is already production-grade.
# ⚠️ Weaknesses / Potential Improvements
# 1. Tokenization for Chunking
# Current _intelligent_chunk still uses rough token estimates (len(words) × 1.3).
# That’s not accurate enough for transformer limits.
# 👉 You already have the alternative (GPT2TokenizerFast) version I gave you earlier — that’s better.
# Suggestion: replace _intelligent_chunk with tokenizer-aware version, but keep a fallback to your current method if HuggingFace isn’t installed.
# 2. Number Normalization
# Right now you only handle spelled-out numbers up to “million”.
# Doesn’t cover larger words (“billion”, “trillion”) or compound forms (“twenty-one”, “ninety-five”).
# Also doesn’t handle decimal words (“point five”).
# Suggestion: extend NUMBER_PATTERNS or integrate a text2num library for robustness.
# 3. Date Normalization
# You depend on normalize_dates from date_utils.
# If that function fails (format mismatch, non-English dates, etc.), you just log a warning.
# Suggestion: wrap it in a fallback parser (like dateparser lib) to handle “next Tuesday”, “5th of May 2024”, etc.
# 4. Sentence Segmentation
# spaCy is good, but for transcripts, it sometimes breaks too aggressively (splits mid-speech).
# Suggestion: allow configurable segmenters → regex (fast), spaCy (accurate), NLTK (lightweight).
# 5. Speaker Extraction
# Your regex-based speaker detection works, but:
# It may miss inline formats like:
# SPEAKER_1 Hello, how are you?
# Or transcripts with no colon separators.
# Suggestion: add regex for “SPEAKER X text” without colon.
# 6. Cleaning Pipeline
# _clean_speech_artifacts removes filler words globally.
# This may over-remove words when they are legit (e.g., “Well, that’s true” → “that’s true”).
# Suggestion: only remove fillers when standalone or at start of sentence.
# 7. Unicode Cleanup
# _remove_non_ascii_and_cleanup → removes emojis entirely.
# Depending on use case, you might want to preserve emojis or convert them (e.g., 🙂 → “smile”).
# Suggestion: make this configurable (preserve/remove/convert).
