# modules/ner_extractor.py
import spacy
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from modules.logger import logger
from modules.date_utils import normalize_dates, extract_all_dates_from_text
import streamlit as st
from modules.meeting_context import MeetingContext

# Load spaCy model once using Streamlit cache
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSPECIFIED = "unspecified"

@dataclass
class ActionItem:
    """Structured action item with metadata"""
    text: str
    priority: Priority
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    verb: Optional[str] = None
    confidence: float = 0.0
    extraction_method: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['priority'] = self.priority.value
        return result

@dataclass
class Person:
    """Structured person entity with role detection"""
    name: str
    mentions: int = 1
    roles: List[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DateEntity:
    """Structured date entity with context"""
    text: str
    normalized: str
    context: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

# Enhanced action detection patterns
ACTION_PATTERNS = {
    # Direct imperatives with common action verbs
    'imperatives': [
        "complete", "submit", "review", "finalize", "prepare", "send", 
        "follow up", "discuss", "present", "resolve", "update", "schedule", 
        "assign", "plan", "organize", "check", "remind", "investigate",
        "implement", "create", "develop", "test", "deploy", "fix",
        "coordinate", "confirm", "validate", "approve", "analyze"
    ],
    
    # Modal verbs indicating action requirements
    'modals': [
        "need to", "have to", "must", "should", "ought to", "required to",
        "supposed to", "going to", "will", "shall"
    ],
    
    # Responsibility indicators
    'assignments': [
        "responsible for", "in charge of", "assigned to", "delegated to",
        "tasked with", "accountable for", "owns", "leads"
    ]
}

# Priority detection patterns
PRIORITY_PATTERNS = {
    Priority.HIGH: [
        r'\b(asap|urgent|critical|emergency|immediate|priority|rush)\b',
        r'\b(today|by end of day|eod|this morning)\b',
        r'\b(red flag|blocker|critical path)\b'
    ],
    Priority.MEDIUM: [
        r'\b(should|need to|important|soon|this week)\b',
        r'\b(by friday|end of week|next few days)\b',
        r'\b(follow up|check in|review)\b'
    ],
    Priority.LOW: [
        r'\b(can|might|could|optional|when possible|if time permits)\b',
        r'\b(nice to have|eventually|someday|future)\b',
        r'\b(low priority|back burner)\b'
    ]
}

# Role detection patterns
ROLE_PATTERNS = {
    'manager': [r'\b(manager|director|lead|supervisor|head of|ceo|cto|cfo)\b'],
    'developer': [r'\b(developer|engineer|programmer|coder|dev)\b'],
    'designer': [r'\b(designer|ui|ux|creative)\b'],
    'analyst': [r'\b(analyst|data scientist|researcher)\b'],
    'coordinator': [r'\b(coordinator|organizer|admin|assistant)\b'],
    'consultant': [r'\b(consultant|advisor|specialist|expert)\b']
}

class EnhancedNERExtractor:
    """Enhanced NER extractor with dependency parsing and priority detection"""
    
    def __init__(self):
        self.nlp = nlp
    
    def extract_people(self, doc) -> List[Person]:
        """Extract people with role detection and mention counting"""
        people_dict = {}
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                name = ent.text.strip()
                if name in people_dict:
                    people_dict[name].mentions += 1
                else:
                    people_dict[name] = Person(
                        name=name,
                        confidence=0.9  # High confidence for NER-detected entities
                    )
        
        # Detect roles from surrounding context
        for person in people_dict.values():
            person.roles = self._detect_roles(doc, person.name)
        
        logger.info(f"👤 People extracted: {len(people_dict)} unique persons")
        return list(people_dict.values())
    
    def _detect_roles(self, doc, person_name: str) -> List[str]:
        """Detect roles for a person based on context"""
        roles = set()
        text_lower = doc.text.lower()
        person_lower = person_name.lower()
        
        # Find sentences containing the person
        for sent in doc.sents:
            if person_lower in sent.text.lower():
                sent_text = sent.text.lower()
                for role, patterns in ROLE_PATTERNS.items():
                    for pattern in patterns:
                        if re.search(pattern, sent_text):
                            roles.add(role)
        
        return list(roles)
    
    def extract_dates(self, doc) -> List[DateEntity]:
        """Extract dates with context"""
        try:
            parsed_dates = extract_all_dates_from_text(doc.text, output_format="human")
            date_entities = []
            
            for date_str in parsed_dates:
                # Find context for each date
                context = self._find_date_context(doc, date_str)
                date_entities.append(DateEntity(
                    text=date_str,
                    normalized=date_str,  # Assuming date_utils handles normalization
                    context=context,
                    confidence=0.8
                ))
            
            logger.info(f"📅 Dates extracted: {len(date_entities)}")
            return date_entities
            
        except Exception as e:
            logger.exception("Error extracting dates")
            return []
    
    def _find_date_context(self, doc, date_str: str) -> str:
        """Find context sentence for a date"""
        for sent in doc.sents:
            if date_str.lower() in sent.text.lower():
                return sent.text.strip()
        return ""
    
    def extract_action_items(self, doc) -> List[ActionItem]:
        """Extract action items using multiple methods"""
        action_items = []
        
        # Method 1: Dependency parsing for imperatives
        imperatives = self._extract_imperatives(doc)
        action_items.extend(imperatives)
        
        # Method 2: Modal verb patterns
        modal_actions = self._extract_modal_actions(doc)
        action_items.extend(modal_actions)
        
        # Method 3: Assignment patterns
        assignment_actions = self._extract_assignment_actions(doc)
        action_items.extend(assignment_actions)
        
        # Method 4: Keyword-based fallback
        keyword_actions = self._extract_keyword_actions(doc)
        action_items.extend(keyword_actions)
        
        # Remove duplicates and enhance with metadata
        unique_actions = self._deduplicate_actions(action_items)
        enhanced_actions = self._enhance_action_metadata(doc, unique_actions)
        
        logger.info(f"✅ Action items extracted: {len(enhanced_actions)}")
        return enhanced_actions
    
    def _extract_imperatives(self, doc) -> List[ActionItem]:
        """Extract imperatives using dependency parsing"""
        imperatives = []
        
        for sent in doc.sents:
            for token in sent:
                # Look for root verbs with no subject (imperatives)
                if (token.dep_ == "ROOT" and 
                    token.pos_ == "VERB" and
                    not any(child.dep_ in ["nsubj", "nsubjpass"] for child in token.children)):
                    
                    # Check if it's an action verb
                    if token.lemma_.lower() in ACTION_PATTERNS['imperatives']:
                        action = ActionItem(
                            text=sent.text.strip(),
                            priority=Priority.UNSPECIFIED,
                            verb=token.lemma_.lower(),
                            confidence=0.9,
                            extraction_method="imperative_parsing"
                        )
                        imperatives.append(action)
        
        return imperatives
    
    def _extract_modal_actions(self, doc) -> List[ActionItem]:
        """Extract actions with modal verbs"""
        modal_actions = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            for modal in ACTION_PATTERNS['modals']:
                if modal in sent_lower:
                    # Find the main action verb
                    verb = self._find_main_verb_after_modal(sent, modal)
                    action = ActionItem(
                        text=sent_text,
                        priority=Priority.UNSPECIFIED,
                        verb=verb,
                        confidence=0.7,
                        extraction_method="modal_detection"
                    )
                    modal_actions.append(action)
                    break
        
        return modal_actions
    
    def _extract_assignment_actions(self, doc) -> List[ActionItem]:
        """Extract actions with explicit assignments"""
        assignment_actions = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            for assignment in ACTION_PATTERNS['assignments']:
                if assignment in sent_lower:
                    # Extract assignee
                    assignee = self._extract_assignee(sent, assignment)
                    action = ActionItem(
                        text=sent_text,
                        priority=Priority.UNSPECIFIED,
                        assignee=assignee,
                        confidence=0.8,
                        extraction_method="assignment_detection"
                    )
                    assignment_actions.append(action)
                    break
        
        return assignment_actions
    
    def _extract_keyword_actions(self, doc) -> List[ActionItem]:
        """Fallback keyword-based extraction"""
        keyword_actions = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            for verb in ACTION_PATTERNS['imperatives']:
                if verb in sent_lower:
                    action = ActionItem(
                        text=sent_text,
                        priority=Priority.UNSPECIFIED,
                        verb=verb,
                        confidence=0.5,
                        extraction_method="keyword_matching"
                    )
                    keyword_actions.append(action)
                    break
        
        return keyword_actions
    
    def _find_main_verb_after_modal(self, sent, modal: str) -> Optional[str]:
        """Find the main action verb after a modal"""
        sent_text = sent.text.lower()
        modal_index = sent_text.find(modal)
        
        if modal_index != -1:
            # Look for verbs after the modal
            for token in sent:
                if (token.i * len(token.text) > modal_index and 
                    token.pos_ == "VERB" and 
                    token.lemma_.lower() in ACTION_PATTERNS['imperatives']):
                    return token.lemma_.lower()
        
        return None
    
    def _extract_assignee(self, sent, assignment_phrase: str) -> Optional[str]:
        """Extract assignee from assignment phrase"""
        # Look for person entities near assignment phrase
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        return None
    
    def _deduplicate_actions(self, actions: List[ActionItem]) -> List[ActionItem]:
        """Remove duplicate actions based on text similarity"""
        unique_actions = []
        seen_texts = set()
        
        for action in actions:
            # Simple deduplication based on text
            if action.text not in seen_texts:
                unique_actions.append(action)
                seen_texts.add(action.text)
        
        return unique_actions
    
    def _enhance_action_metadata(self, doc, actions: List[ActionItem]) -> List[ActionItem]:
        """Enhance actions with priority, assignee, and deadline detection"""
        enhanced_actions = []
        
        for action in actions:
            # Detect priority
            action.priority = self._detect_priority(action.text)
            
            # Extract deadline if not already set
            if not action.deadline:
                action.deadline = self._extract_deadline_from_text(action.text)
            
            # Extract assignee if not already set
            if not action.assignee:
                action.assignee = self._extract_assignee_from_text(doc, action.text)
            
            enhanced_actions.append(action)
        
        return enhanced_actions
    
    def _detect_priority(self, text: str) -> Priority:
        """Detect priority level from text"""
        text_lower = text.lower()
        
        for priority, patterns in PRIORITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return priority
        
        return Priority.UNSPECIFIED
    
    def _extract_deadline_from_text(self, text: str) -> Optional[str]:
        """Extract deadline from action text"""
        try:
            dates = extract_all_dates_from_text(text, output_format="human")
            return dates[0] if dates else None
        except:
            return None
    
    def _extract_assignee_from_text(self, doc, action_text: str) -> Optional[str]:
        """Extract assignee from action text"""
        # Find the sentence containing the action
        for sent in doc.sents:
            if action_text in sent.text:
                # Look for person entities in the same sentence
                for ent in sent.ents:
                    if ent.label_ == "PERSON":
                        return ent.text.strip()
        return None

# Initialize the enhanced extractor
_extractor_instance = None

def get_extractor():
    """Get singleton extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EnhancedNERExtractor()
    return _extractor_instance

def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract people, dates, and action items from transcript text.
    Returns a dictionary of structured entities in dictionary format for JSON compatibility.
    """
    if not text.strip():
        logger.warning("⚠️ Empty input text passed to NER extractor.")
        return {"people": [], "dates": [], "action_items": []}
    
    try:
        extractor = get_extractor()
        doc = extractor.nlp(text)
        
        # Extract entities
        people = extractor.extract_people(doc)
        dates = extractor.extract_dates(doc)
        action_items = extractor.extract_action_items(doc)
        
        # Convert to dictionaries for JSON serialization
        result = {
            "people": [person.to_dict() for person in people],
            "dates": [date.to_dict() for date in dates],
            "action_items": [action.to_dict() for action in action_items]
        }
        
        logger.info(f"✅ Entity extraction completed: "
                   f"{len(people)} people, {len(dates)} dates, {len(action_items)} action items")
        
        return result
        
    except Exception as e:
        logger.exception("❌ Failed to extract entities.")
        return {"people": [], "dates": [], "action_items": []}

def enrich_context_with_entities(context: MeetingContext) -> MeetingContext:
    """
    Pipeline-stage wrapper that extracts entities from MeetingContext
    and enriches it in-place.
    """
    if not context.raw_text.strip():
        logger.warning("⚠️ Empty transcript text in MeetingContext for NER.")
        context.metadata["entities"] = {
            "people": [],
            "dates": [],
            "action_items": []
        }
        return context

    try:
        entities = extract_entities(context.raw_text)

        # Attach structured entities to context
        context.metadata["entities"] = entities
        context.metadata["people"] = entities.get("people", [])
        context.metadata["dates"] = entities.get("dates", [])
        context.metadata["action_items"] = entities.get("action_items", [])

        logger.info(
            f"🧠 NER added to context: "
            f"{len(context.metadata['people'])} people, "
            f"{len(context.metadata['dates'])} dates, "
            f"{len(context.metadata['action_items'])} action items"
        )

        return context

    except Exception:
        logger.exception("❌ Failed to enrich MeetingContext with NER data.")
        context.metadata["entities_error"] = "NER extraction failed"
        return context

# Backward compatibility function
def extract_entities_legacy(text: str) -> Dict[str, List[str]]:
    """
    Legacy function that returns simple lists for backward compatibility.
    """
    structured_entities = extract_entities(text)
    
    return {
        "people": [person["name"] for person in structured_entities["people"]],
        "dates": [date["text"] for date in structured_entities["dates"]],
        "action_items": [action["text"] for action in structured_entities["action_items"]]
    }

# Utility functions for analysis
def get_high_priority_actions(entities: Dict[str, List[Dict]]) -> List[Dict]:
    """Get all high priority action items"""
    return [action for action in entities.get("action_items", []) 
            if action.get("priority") == Priority.HIGH.value]

def get_actions_by_assignee(entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Group actions by assignee"""
    actions_by_assignee = {}
    for action in entities.get("action_items", []):
        assignee = action.get("assignee", "Unassigned")
        if assignee not in actions_by_assignee:
            actions_by_assignee[assignee] = []
        actions_by_assignee[assignee].append(action)
    return actions_by_assignee

def get_actions_with_deadlines(entities: Dict[str, List[Dict]]) -> List[Dict]:
    """Get all actions that have deadlines"""
    return [action for action in entities.get("action_items", []) 
            if action.get("deadline")]