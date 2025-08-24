# modules/summarizer.py
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Union
import torch
import streamlit as st
from modules.logger import logger
from dataclasses import dataclass
from enum import Enum
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

warnings.filterwarnings("ignore", category=FutureWarning)

class SummaryStrategy(Enum):
    ABSTRACTIVE = "abstractive"
    EXTRACTIVE = "extractive"
    HYBRID = "hybrid"
    MAP_REDUCE = "map_reduce"

class SummarizerModel(Enum):
    DISTILBART = "sshleifer/distilbart-cnn-12-6"
    BART_LARGE = "facebook/bart-large-cnn"
    PEGASUS = "google/pegasus-xsum"
    T5_SMALL = "t5-small"
    FLAN_T5_BASE = "google/flan-t5-base"

@dataclass
class SummaryConfig:
    """Configuration for summarization parameters"""
    strategy: SummaryStrategy = SummaryStrategy.MAP_REDUCE
    model: SummarizerModel = SummarizerModel.DISTILBART
    max_chunk_tokens: int = 512
    max_summary_tokens: int = 150
    min_summary_tokens: int = 30
    extractive_ratio: float = 0.3  # For hybrid: ratio of sentences to extract
    overlap_sentences: int = 2  # Overlap between chunks for context
    quality_threshold: float = 0.5  # Minimum quality score for summaries
    
    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy.value,
            'model': self.model.value,
            'max_chunk_tokens': self.max_chunk_tokens,
            'max_summary_tokens': self.max_summary_tokens,
            'min_summary_tokens': self.min_summary_tokens,
            'extractive_ratio': self.extractive_ratio,
            'overlap_sentences': self.overlap_sentences,
            'quality_threshold': self.quality_threshold
        }

@dataclass
class SummaryResult:
    """Result object containing summary and metadata"""
    summary: str
    strategy_used: str
    model_used: str
    chunk_count: int
    success_rate: float
    processing_time: float
    quality_score: float = 0.0
    key_sentences: List[str] = None
    
    def __post_init__(self):
        if self.key_sentences is None:
            self.key_sentences = []

class EnhancedSummarizer:
    """Enhanced summarizer with multiple strategies and fallback mechanisms"""
    
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self.stop_words = set(stopwords.words('english'))
    
    @st.cache_resource
    def _load_model(_self, model_name: str):
        """Load and cache summarization model"""
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"🔄 Loading model: {model_name} on {'GPU' if device == 0 else 'CPU'}")
        
        try:
            if model_name.startswith("t5") or "flan-t5" in model_name:
                # T5 models need text2text-generation pipeline
                return pipeline("text2text-generation", model=model_name, device=device)
            else:
                return pipeline("summarization", model=model_name, device=device)
        except Exception as e:
            logger.exception(f"❌ Failed to load model {model_name}")
            raise RuntimeError(f"Could not load model {model_name}") from e
    
    @st.cache_resource
    def _load_tokenizer(_self, model_name: str):
        """Load and cache tokenizer for token counting"""
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {model_name}, using default")
            return None
    
    def _get_model(self, model_name: str):
        """Get cached model instance"""
        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)
            self._tokenizers[model_name] = self._load_tokenizer(model_name)
        return self._models[model_name], self._tokenizers[model_name]
    
    def _count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens in text"""
        if tokenizer:
            try:
                return len(tokenizer.encode(text, truncation=False))
            except:
                pass
        # Fallback: approximate token count
        return len(text.split()) * 1.3  # Rough approximation
    
    def _intelligent_chunk_text(self, text: str, max_tokens: int, 
                               overlap_sentences: int, tokenizer) -> List[str]:
        """Intelligently chunk text with sentence boundaries and overlap"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence, tokenizer)
            
            # If adding this sentence would exceed limit, save current chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap for context continuity
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_tokens = sum(self._count_tokens(s, tokenizer) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.debug(f"📄 Created {len(chunks)} intelligent chunks with overlap")
        return chunks
    
    def _extractive_summarize(self, text: str, ratio: float = 0.3) -> List[str]:
        """Extract key sentences using TF-IDF and cosine similarity"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 3:
                return sentences
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate sentence importance scores
            similarity_matrix = cosine_similarity(sentence_vectors)
            scores = np.mean(similarity_matrix, axis=1)
            
            # Select top sentences
            num_sentences = max(1, int(len(sentences) * ratio))
            top_indices = scores.argsort()[-num_sentences:]
            top_indices.sort()  # Maintain original order
            
            key_sentences = [sentences[i] for i in top_indices]
            logger.debug(f"🔍 Extracted {len(key_sentences)} key sentences")
            return key_sentences
            
        except Exception as e:
            logger.warning(f"Extractive summarization failed: {e}")
            # Fallback: return first few sentences
            sentences = sent_tokenize(text)
            return sentences[:max(1, int(len(sentences) * ratio))]
    
    def _summarize_single_chunk(self, chunk: str, model, tokenizer, config: SummaryConfig) -> Tuple[str, float]:
        """Summarize a single chunk with quality assessment"""
        if not chunk.strip():
            return "", 0.0
        
        try:
            # Prepare input based on model type
            if config.model.value.startswith("t5") or "flan-t5" in config.model.value:
                input_text = f"summarize: {chunk}"
                result = model(
                    input_text,
                    max_length=config.max_summary_tokens,
                    min_length=config.min_summary_tokens,
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]["generated_text"]
            else:
                result = model(
                    chunk,
                    max_length=config.max_summary_tokens,
                    min_length=config.min_summary_tokens,
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]["summary_text"]
            
            # Assess quality (simple heuristic)
            quality = self._assess_summary_quality(chunk, summary)
            return summary.strip(), quality
            
        except Exception as e:
            logger.warning(f"Failed to summarize chunk: {e}")
            return "[Summarization failed for this section]", 0.0
    
    def _assess_summary_quality(self, original: str, summary: str) -> float:
        """Assess summary quality using simple heuristics"""
        if not summary or summary.startswith("[") or len(summary) < 10:
            return 0.0
        
        # Basic quality indicators
        score = 0.5  # Base score
        
        # Length appropriateness (should be significantly shorter)
        compression_ratio = len(summary) / max(len(original), 1)
        if 0.1 <= compression_ratio <= 0.5:
            score += 0.2
        
        # Contains meaningful content (not just stopwords)
        meaningful_words = [w for w in summary.lower().split() if w not in self.stop_words]
        if len(meaningful_words) >= 5:
            score += 0.2
        
        # Proper sentence structure
        if summary.endswith('.') or summary.endswith('!') or summary.endswith('?'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _map_reduce_summarize(self, text: str, config: SummaryConfig) -> SummaryResult:
        """Map-reduce summarization: chunk -> summarize -> combine -> re-summarize"""
        import time
        start_time = time.time()
        
        try:
            model, tokenizer = self._get_model(config.model.value)
            
            # Step 1: Intelligent chunking
            chunks = self._intelligent_chunk_text(
                text, config.max_chunk_tokens, config.overlap_sentences, tokenizer
            )
            
            if not chunks:
                return SummaryResult(
                    summary="No content to summarize",
                    strategy_used=config.strategy.value,
                    model_used=config.model.value,
                    chunk_count=0,
                    success_rate=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Summarize each chunk
            logger.info(f"📝 Starting map-reduce summarization of {len(chunks)} chunks")
            chunk_summaries = []
            successful_chunks = 0
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                summary, quality = self._summarize_single_chunk(chunk, model, tokenizer, config)
                
                if quality >= config.quality_threshold:
                    chunk_summaries.append(summary)
                    successful_chunks += 1
                else:
                    logger.warning(f"Chunk {i+1} summary quality too low ({quality:.2f})")
                    # Use extractive summary as fallback
                    key_sentences = self._extractive_summarize(chunk, 0.5)
                    chunk_summaries.append(' '.join(key_sentences))
                    successful_chunks += 0.5  # Partial success
            
            if not chunk_summaries:
                return SummaryResult(
                    summary="Failed to generate any summaries",
                    strategy_used=config.strategy.value,
                    model_used=config.model.value,
                    chunk_count=len(chunks),
                    success_rate=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Combine and re-summarize if needed
            combined_summary = '\n'.join(chunk_summaries)
            
            # If combined summary is still too long, re-summarize
            if self._count_tokens(combined_summary, tokenizer) > config.max_summary_tokens * 2:
                logger.info("🔄 Re-summarizing combined chunks")
                final_summary, final_quality = self._summarize_single_chunk(
                    combined_summary, model, tokenizer, config
                )
            else:
                final_summary = combined_summary
                final_quality = successful_chunks / len(chunks)
            
            processing_time = time.time() - start_time
            success_rate = successful_chunks / len(chunks)
            
            logger.info(f"✅ Map-reduce summarization completed in {processing_time:.2f}s")
            
            return SummaryResult(
                summary=final_summary,
                strategy_used=config.strategy.value,
                model_used=config.model.value,
                chunk_count=len(chunks),
                success_rate=success_rate,
                processing_time=processing_time,
                quality_score=final_quality
            )
            
        except Exception as e:
            logger.exception("❌ Map-reduce summarization failed")
            return SummaryResult(
                summary=f"Summarization failed: {str(e)}",
                strategy_used=config.strategy.value,
                model_used=config.model.value,
                chunk_count=0,
                success_rate=0.0,
                processing_time=time.time() - start_time
            )
    
    def _hybrid_summarize(self, text: str, config: SummaryConfig) -> SummaryResult:
        """Hybrid approach: extractive preprocessing + abstractive summarization"""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Extract key sentences
            key_sentences = self._extractive_summarize(text, config.extractive_ratio)
            extracted_text = ' '.join(key_sentences)
            
            logger.info(f"🔍 Extracted {len(key_sentences)} key sentences for hybrid summarization")
            
            # Step 2: Apply abstractive summarization to extracted content
            config_copy = SummaryConfig(
                strategy=SummaryStrategy.ABSTRACTIVE,
                model=config.model,
                max_chunk_tokens=config.max_chunk_tokens,
                max_summary_tokens=config.max_summary_tokens,
                min_summary_tokens=config.min_summary_tokens
            )
            
            if len(extracted_text.split()) < 100:
                # If extracted text is short, use it directly
                final_summary = extracted_text
                quality_score = 0.8
            else:
                # Apply abstractive summarization
                abstractive_result = self._map_reduce_summarize(extracted_text, config_copy)
                final_summary = abstractive_result.summary
                quality_score = abstractive_result.quality_score
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=final_summary,
                strategy_used=config.strategy.value,
                model_used=config.model.value,
                chunk_count=1,
                success_rate=1.0 if quality_score > 0.5 else 0.5,
                processing_time=processing_time,
                quality_score=quality_score,
                key_sentences=key_sentences
            )
            
        except Exception as e:
            logger.exception("❌ Hybrid summarization failed")
            return SummaryResult(
                summary=f"Hybrid summarization failed: {str(e)}",
                strategy_used=config.strategy.value,
                model_used=config.model.value,
                chunk_count=0,
                success_rate=0.0,
                processing_time=time.time() - start_time
            )
    
    def _extractive_only_summarize(self, text: str, config: SummaryConfig) -> SummaryResult:
        """Pure extractive summarization"""
        import time
        start_time = time.time()
        
        try:
            key_sentences = self._extractive_summarize(text, config.extractive_ratio)
            summary = ' '.join(key_sentences)
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=summary,
                strategy_used=config.strategy.value,
                model_used="extractive_tfidf",
                chunk_count=1,
                success_rate=1.0,
                processing_time=processing_time,
                quality_score=0.7,  # Extractive summaries are generally reliable
                key_sentences=key_sentences
            )
            
        except Exception as e:
            logger.exception("❌ Extractive summarization failed")
            return SummaryResult(
                summary=f"Extractive summarization failed: {str(e)}",
                strategy_used=config.strategy.value,
                model_used="extractive_tfidf",
                chunk_count=0,
                success_rate=0.0,
                processing_time=time.time() - start_time
            )
    
    def summarize(self, text: str, config: SummaryConfig = None) -> SummaryResult:
        """Main summarization method with strategy selection and fallback"""
        if not text.strip():
            logger.warning("⚠️ Empty input text for summarization")
            return SummaryResult(
                summary="No content to summarize",
                strategy_used="none",
                model_used="none",
                chunk_count=0,
                success_rate=0.0,
                processing_time=0.0
            )
        
        if config is None:
            config = SummaryConfig()
        
        logger.info(f"🚀 Starting summarization with strategy: {config.strategy.value}")
        
        # Try primary strategy
        try:
            if config.strategy == SummaryStrategy.MAP_REDUCE:
                result = self._map_reduce_summarize(text, config)
            elif config.strategy == SummaryStrategy.HYBRID:
                result = self._hybrid_summarize(text, config)
            elif config.strategy == SummaryStrategy.EXTRACTIVE:
                result = self._extractive_only_summarize(text, config)
            else:  # ABSTRACTIVE
                config.strategy = SummaryStrategy.MAP_REDUCE  # Default to map-reduce for single strategy
                result = self._map_reduce_summarize(text, config)
            
            # Check if result is acceptable
            if result.success_rate >= 0.3 and len(result.summary) > 20:
                return result
            
        except Exception as e:
            logger.exception(f"Primary strategy {config.strategy.value} failed")
        
        # Fallback to extractive summarization
        logger.warning("🔄 Falling back to extractive summarization")
        try:
            fallback_config = SummaryConfig(strategy=SummaryStrategy.EXTRACTIVE)
            return self._extractive_only_summarize(text, fallback_config)
        except Exception as e:
            logger.exception("❌ All summarization strategies failed")
            return SummaryResult(
                summary="All summarization methods failed. Please try with shorter text.",
                strategy_used="fallback_failed",
                model_used="none",
                chunk_count=0,
                success_rate=0.0,
                processing_time=0.0
            )

# Global summarizer instance
_summarizer_instance = None

def get_summarizer() -> EnhancedSummarizer:
    """Get singleton summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = EnhancedSummarizer()
    return _summarizer_instance

# Main API functions
def summarize_chunks(chunks: List[str], 
                    strategy: str = "map_reduce",
                    model: str = "sshleifer/distilbart-cnn-12-6",
                    max_tokens: int = 150) -> str:
    """
    Backward compatible function for existing code
    """
    if not chunks:
        logger.warning("⚠️ Empty chunks for summarization")
        return "No content to summarize."
    
    # Combine chunks into single text
    text = '\n'.join(chunk for chunk in chunks if chunk.strip())
    
    # Create config
    try:
        strategy_enum = SummaryStrategy(strategy)
    except ValueError:
        strategy_enum = SummaryStrategy.MAP_REDUCE
    
    try:
        model_enum = SummarizerModel(model)
    except ValueError:
        model_enum = SummarizerModel.DISTILBART
    
    config = SummaryConfig(
        strategy=strategy_enum,
        model=model_enum,
        max_summary_tokens=max_tokens
    )
    
    # Summarize
    summarizer = get_summarizer()
    result = summarizer.summarize(text, config)
    
    return result.summary

def enhanced_summarize(text: str, 
                      strategy: str = "map_reduce",
                      model: str = "sshleifer/distilbart-cnn-12-6",
                      max_tokens: int = 150,
                      extractive_ratio: float = 0.3) -> Dict:
    """
    Enhanced summarization with full result metadata
    """
    try:
        strategy_enum = SummaryStrategy(strategy)
    except ValueError:
        strategy_enum = SummaryStrategy.MAP_REDUCE
    
    try:
        model_enum = SummarizerModel(model)
    except ValueError:
        model_enum = SummarizerModel.DISTILBART
    
    config = SummaryConfig(
        strategy=strategy_enum,
        model=model_enum,
        max_summary_tokens=max_tokens,
        extractive_ratio=extractive_ratio
    )
    
    summarizer = get_summarizer()
    result = summarizer.summarize(text, config)
    
    return {
        'summary': result.summary,
        'strategy_used': result.strategy_used,
        'model_used': result.model_used,
        'chunk_count': result.chunk_count,
        'success_rate': result.success_rate,
        'processing_time': result.processing_time,
        'quality_score': result.quality_score,
        'key_sentences': result.key_sentences or []
    }

# Utility functions
def get_available_models() -> List[str]:
    """Get list of available summarization models"""
    return [model.value for model in SummarizerModel]

def get_available_strategies() -> List[str]:
    """Get list of available summarization strategies"""
    return [strategy.value for strategy in SummaryStrategy]