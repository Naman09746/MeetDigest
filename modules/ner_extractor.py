# modules/ner_extractor.py

import spacy
from typing import List, Dict
from modules.logger import logger
from modules.date_utils import normalize_dates
import streamlit as st

# Load spaCy model once using Streamlit cache
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Heuristic action verbs (expandable list)
ACTION_VERBS = [
    "complete", "submit", "review", "finalize", "prepare",
    "send", "follow up", "discuss", "present", "resolve", "update",
    "schedule", "assign", "plan", "organize", "check", "remind"
]


def extract_people(doc) -> List[str]:
    people = {
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "PERSON" and len(ent.text.strip()) > 1
    }

    logger.info(f"👤 People extracted: {len(people)}")
    return list(people)


from modules.date_utils import extract_all_dates_from_text

def extract_dates(doc) -> List[str]:
    parsed_dates = extract_all_dates_from_text(doc.text, output_format="human")
    logger.info(f"🕒 Dates parsed: {len(parsed_dates)}")
    return parsed_dates


def extract_action_items(doc) -> List[str]:
    action_items = set()
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        for verb in ACTION_VERBS:
            if verb in sent_lower:
                action_items.add(sent.text.strip())
                break

    logger.info(f"✅ Action items extracted: {len(action_items)}")
    return list(action_items)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract people, dates, and action items from transcript text.
    Returns a dictionary of structured entities.
    """
    if not text.strip():
        logger.warning("⚠️ Empty input text passed to NER extractor.")
        return {"people": [], "dates": [], "action_items": []}

    try:
        doc = nlp(text)

        return {
            "people": extract_people(doc),
            "dates": extract_dates(doc),
            "action_items": extract_action_items(doc)
        }

    except Exception as e:
        logger.exception("❌ Failed to extract entities.")
        return {"people": [], "dates": [], "action_items": []}
