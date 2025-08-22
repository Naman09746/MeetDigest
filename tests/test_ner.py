# tests/test_ner.py

import pytest
from modules import preprocessor
from modules.ner_extractor import extract_entities

@pytest.mark.parametrize("text, expected_people", [
    ("Alice and Bob will attend the meeting.", ["Alice", "Bob"]),
    ("No people mentioned here.", []),
])
def test_extract_people(text, expected_people):
    entities = extract_entities(text)
    for person in expected_people:
        assert person in entities["people"]
    assert isinstance(entities["people"], list)

@pytest.mark.parametrize("text, expect_date", [
    ("The meeting is scheduled for next Friday.", True),
    ("Let’s talk about cats and dogs.", False),
])
def test_extract_dates(text, expect_date):
    entities = extract_entities(text)
    if expect_date:
        assert len(entities["dates"]) > 0
    else:
        assert len(entities["dates"]) == 0

@pytest.mark.parametrize("text, expect_action", [
    ("Please prepare the slides by Monday.", True),
    ("Nothing to do here.", False),
])
def test_extract_action_items(text, expect_action):
    entities = extract_entities(text)
    if expect_action:
        assert len(entities["action_items"]) > 0
    else:
        assert len(entities["action_items"]) == 0
