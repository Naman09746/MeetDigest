# modules/summarizer.py

from transformers import pipeline
from typing import List
import torch
import streamlit as st
from modules.logger import logger

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource
def load_summarizer(model_name: str = DEFAULT_MODEL):
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"🔄 Loading summarization model: {model_name} on {'GPU' if device == 0 else 'CPU'}")
    try:
        return pipeline("summarization", model=model_name, device=device)
    except Exception as e:
        logger.exception("❌ Failed to load summarization model.")
        raise RuntimeError("Could not load summarizer model.") from e


def summarize_chunks(
    chunks: List[str],
    max_input_tokens: int = 512,
    max_output_tokens: int = 120,
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Summarize a list of transcript chunks.
    Returns a single combined summary.
    """
    if not chunks:
        logger.warning("⚠️ Empty input chunks received for summarization.")
        return "No content to summarize."

    try:
        summarizer = load_summarizer(model_name)
        summaries = []

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            estimated_char_limit = max_input_tokens * 4  # Approx 4 chars/token
            truncated_chunk = chunk[:estimated_char_limit]

            logger.debug(f"📄 Summarizing chunk {idx + 1}/{len(chunks)} (chars: {len(truncated_chunk)})")

            try:
                summary = summarizer(
                    truncated_chunk,
                    max_length=max_output_tokens,
                    min_length=30,
                    do_sample=False
                )[0]["summary_text"]

                summaries.append(summary.strip())

            except Exception as chunk_error:
                logger.exception(f"❌ Failed to summarize chunk {idx + 1}")
                summaries.append("[Error summarizing this part.]")

        if summaries:
            final = "\n".join(summaries).strip()
            logger.info(f"✅ Generated summary from {len(chunks)} chunks.")
            return final
        else:
            logger.warning("⚠️ No successful summaries returned.")
            return "Summary generation failed."

    except Exception as e:
        logger.exception("❌ Overall summarization failed.")
        return "An error occurred during summarization."
