from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class MeetingContext:
    raw_text: str = ""
    speaker_segments: List[Any] = field(default_factory=list)
    speaker_stats: List[Any] = field(default_factory=list)

    entities: Dict[str, List[str]] = field(default_factory=dict)
    dates: List[str] = field(default_factory=list)

    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
