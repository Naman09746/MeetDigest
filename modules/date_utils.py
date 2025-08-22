# modules/date_utils.py

from typing import Optional, List, Union
from dateparser.search import search_dates
import dateparser
import datetime
import logging
import streamlit as st

logger = logging.getLogger("MeetingSummarizer")

OutputFormat = Union[str]  # Can refine using Literal["human", "iso", "date_only", "utc"]

DEFAULT_SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "RELATIVE_BASE": datetime.datetime.now(),
    "RETURN_AS_TIMEZONE_AWARE": True,
    "DATE_ORDER": "MDY",
}


@st.cache_resource
def get_date_settings() -> dict:
    """
    Cached default dateparser settings for Streamlit.
    """
    return DEFAULT_SETTINGS.copy()


def parse_date_string(
    date_str: str,
    settings: Optional[dict] = None,
    output_format: OutputFormat = "human"
) -> Optional[str]:
    """
    Parse a fuzzy date string into a specific format.
    """
    if not date_str.strip():
        return None

    try:
        config = get_date_settings()
        if settings:
            config.update(settings)

        parsed = dateparser.parse(date_str, settings=config)

        if not parsed:
            logger.warning(f"⚠️ Could not parse date string: '{date_str}'")
            return None

        return format_datetime(parsed, output_format)

    except Exception as e:
        logger.exception(f"❌ Failed to parse date string: '{date_str}'")
        return None


def extract_all_dates_from_text(
    text: str,
    settings: Optional[dict] = None,
    output_format: OutputFormat = "human"
) -> List[str]:
    """
    Extract fuzzy date phrases (like 'next Friday') from full text using dateparser.search.
    """
    if not text.strip():
        return []

    try:
        config = get_date_settings()
        if settings:
            config.update(settings)

        results = search_dates(text, settings=config, languages=["en"])

        if not results:
            logger.info("🔍 No dates found in input text.")
            return []

        return [
            format_datetime(parsed, output_format)
            for original, parsed in results
        ]

    except Exception:
        logger.exception("❌ Error while searching for dates in text.")
        return []


def normalize_dates(
    date_strings: List[str],
    output_format: OutputFormat = "human",
    deduplicate: bool = True,
    sort_output: bool = True,
    settings: Optional[dict] = None
) -> List[str]:
    """
    Normalize a list of individual date strings into standardized formats.
    """
    config = get_date_settings()
    if settings:
        config.update(settings)

    parsed_dates = []

    for date_str in date_strings:
        formatted = parse_date_string(
            date_str=date_str,
            settings=config,
            output_format=output_format
        )
        if formatted:
            parsed_dates.append(formatted)

    if deduplicate:
        parsed_dates = list(set(parsed_dates))

    if sort_output:
        try:
            parsed_dates.sort()
        except Exception:
            logger.warning("⚠️ Could not sort parsed dates")

    return parsed_dates


def format_datetime(dt: datetime.datetime, output_format: OutputFormat) -> str:
    """
    Format datetime object into desired output format.
    """
    if output_format == "iso":
        return dt.isoformat(sep=" ", timespec="minutes")
    elif output_format == "human":
        return dt.strftime("%A, %B %d, %Y at %I:%M %p")
    elif output_format == "date_only":
        return dt.strftime("%Y-%m-%d")
    elif output_format == "utc":
        return dt.astimezone().isoformat()
    else:
        logger.warning(f"⚠️ Unknown output_format: '{output_format}', returning ISO")
        return dt.isoformat()
