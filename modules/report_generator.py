# modules/report_generator.py

from typing import List, Dict, Optional
from datetime import datetime
from modules.logger import logger
import arrow

def format_section(title: str, items: List[str], empty_msg: str = "No data available.") -> str:
    section = f"\n{title}\n" + "-" * 50 + "\n"
    if items:
        section += "\n".join(f"- {item}" for item in sorted(set(items)))
    else:
        section += empty_msg
    section += "\n"
    return section


def generate_meeting_report(
    participants: List[str],
    action_items: List[str],
    deadlines: List[str],
    summary: str,
    original_transcript: str,
    title: str = "Meeting Summary Report",
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """
    Generates a full plain-text meeting report.
    """
    try:
        report_lines = []

        # Report title and timestamp
        report_lines.append(f"{title}")
        report_lines.append("=" * len(title))
        timestamp = arrow.now().format("YYYY-MM-DD HH:mm A (ZZ)")
        report_lines.append(f"Generated on: {timestamp}")

        # Optional metadata section
        if metadata:
            report_lines.append("\n📋 Metadata\n" + "-" * 50)
            for key, value in metadata.items():
                report_lines.append(f"{key}: {value}")

        # Participants
        report_lines.append(format_section("📛 Participants", participants, "No participants detected."))

        # Action items
        report_lines.append(format_section("📌 Action Items", action_items, "No action items found."))

        # Deadlines / Dates
        report_lines.append(format_section("⏰ Deadlines / Dates", deadlines, "No dates or times detected."))

        # Summary
        report_lines.append("\n🧠 Meeting Summary\n" + "-" * 50)
        report_lines.append(summary.strip() or "No summary available.")

        # Full cleaned transcript
        report_lines.append("\n📄 Full Transcript (Cleaned)\n" + "-" * 50)
        report_lines.append(original_transcript.strip() or "No transcript content available.")

        report_text = "\n".join(report_lines)
        logger.info("✅ Meeting report generated successfully.")
        return report_text

    except Exception as e:
        logger.exception("❌ Failed to generate meeting report.")
        raise
