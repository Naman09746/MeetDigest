# modules/date_utils.py
from typing import Optional, List, Literal, Tuple, Dict, Any
from dateparser.search import search_dates
import dateparser
import datetime
import logging
import functools

logger = logging.getLogger("MeetingSummarizer")

# Strict output format types
OutputFormat = Literal["human", "iso", "date_only", "utc"]

# Default settings as module constant
DEFAULT_SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "RELATIVE_BASE": datetime.datetime.now(),
    "RETURN_AS_TIMEZONE_AWARE": True,
    "DATE_ORDER": "MDY",
}

@functools.lru_cache(maxsize=1)
def get_date_settings() -> Dict[str, Any]:
    """
    Cached default dateparser settings using functools for portability.
    """
    return DEFAULT_SETTINGS.copy()

def parse_date_string(
    date_str: str,
    settings: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = "human"
) -> Tuple[Optional[datetime.datetime], Optional[str]]:
    """
    Parse a fuzzy date string into a datetime object and formatted string.
    
    Returns:
        Tuple of (datetime_object, formatted_string) for safer downstream usage
    """
    if not date_str.strip():
        return None, None

    try:
        config = get_date_settings()
        if settings:
            config.update(settings)
        
        parsed = dateparser.parse(date_str, settings=config)
        if not parsed:
            logger.warning(f"⚠️ Could not parse date string: '{date_str}'")
            return None, None
        
        formatted = format_datetime(parsed, output_format)
        return parsed, formatted
        
    except Exception as e:
        logger.exception(f"❌ Failed to parse date string: '{date_str}'")
        return None, None

def extract_all_dates_from_text(
    text: str,
    settings: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = "human"
) -> List[Tuple[datetime.datetime, str]]:
    """
    Extract fuzzy date phrases from text, returning datetime objects and formatted strings.
    
    Returns:
        List of (datetime_object, formatted_string) tuples
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
        
        parsed_results = []
        for original, parsed in results:
            formatted = format_datetime(parsed, output_format)
            parsed_results.append((parsed, formatted))
        
        return parsed_results
        
    except Exception:
        logger.exception("❌ Error while searching for dates in text.")
        return []

def normalize_dates(
    date_strings: List[str],
    output_format: OutputFormat = "human",
    deduplicate: bool = True,
    sort_output: bool = True,
    settings: Optional[Dict[str, Any]] = None
) -> List[Tuple[datetime.datetime, str]]:
    """
    Normalize a list of date strings into standardized formats.
    
    Always parse → sort by datetime objects → then format for proper ordering.
    
    Returns:
        List of (datetime_object, formatted_string) tuples, sorted by datetime
    """
    config = get_date_settings()
    if settings:
        config.update(settings)
    
    parsed_dates = []
    seen_datetimes = set() if deduplicate else None
    
    for date_str in date_strings:
        dt_obj, formatted = parse_date_string(
            date_str=date_str,
            settings=config,
            output_format=output_format
        )
        
        if dt_obj and formatted:
            # Deduplicate by datetime object, not string
            if deduplicate:
                if dt_obj in seen_datetimes:
                    continue
                seen_datetimes.add(dt_obj)
            
            parsed_dates.append((dt_obj, formatted))
    
    # Sort by datetime objects, not strings
    if sort_output:
        try:
            parsed_dates.sort(key=lambda x: x[0])
        except Exception:
            logger.warning("⚠️ Could not sort parsed dates by datetime")
    
    return parsed_dates

def format_datetime(dt: datetime.datetime, output_format: OutputFormat) -> str:
    """
    Format datetime object into desired output format with strict type checking.
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
        # This should never happen with Literal types, but keeping for safety
        logger.error(f"❌ Invalid output_format: '{output_format}'")
        raise ValueError(f"Invalid output_format: '{output_format}'. Must be one of: 'human', 'iso', 'date_only', 'utc'")

# Convenience functions that maintain backward compatibility
def parse_date_string_legacy(
    date_str: str,
    settings: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = "human"
) -> Optional[str]:
    """
    Legacy function that returns only the formatted string (for backward compatibility).
    """
    dt_obj, formatted = parse_date_string(date_str, settings, output_format)
    return formatted

def extract_all_dates_from_text_legacy(
    text: str,
    settings: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = "human"
) -> List[str]:
    """
    Legacy function that returns only formatted strings (for backward compatibility).
    """
    results = extract_all_dates_from_text(text, settings, output_format)
    return [formatted for dt_obj, formatted in results]

def normalize_dates_legacy(
    date_strings: List[str],
    output_format: OutputFormat = "human",
    deduplicate: bool = True,
    sort_output: bool = True,
    settings: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Legacy function that returns only formatted strings (for backward compatibility).
    """
    results = normalize_dates(date_strings, output_format, deduplicate, sort_output, settings)
    return [formatted for dt_obj, formatted in results]

# Utility functions for common operations
def get_earliest_date(dates: List[Tuple[datetime.datetime, str]]) -> Optional[Tuple[datetime.datetime, str]]:
    """Get the earliest date from a list of (datetime, formatted_string) tuples."""
    if not dates:
        return None
    return min(dates, key=lambda x: x[0])

def get_latest_date(dates: List[Tuple[datetime.datetime, str]]) -> Optional[Tuple[datetime.datetime, str]]:
    """Get the latest date from a list of (datetime, formatted_string) tuples."""
    if not dates:
        return None
    return max(dates, key=lambda x: x[0])

def filter_dates_in_range(
    dates: List[Tuple[datetime.datetime, str]],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None
) -> List[Tuple[datetime.datetime, str]]:
    """Filter dates within a specific range."""
    filtered = dates
    
    if start_date:
        filtered = [(dt, fmt) for dt, fmt in filtered if dt >= start_date]
    
    if end_date:
        filtered = [(dt, fmt) for dt, fmt in filtered if dt <= end_date]
    
    return filtered