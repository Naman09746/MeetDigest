# run_demo.py
from modules.summarizer import enhanced_summarize
import os

def demo_transcription(text_file="data/sample_transcript.txt"):
    if os.path.exists(text_file):
        with open(text_file, "r") as f:
            text = f.read().strip()
        print("\n--- Transcript Loaded ---\n")
        print(text[:500], "...\n")  # preview first 500 chars
        return text
    else:
        print("⚠️ No transcript file found, using dummy text")
        return """John: Welcome to the meeting. 
                  Mary: We need to finalize the budget. 
                  Alex: I'll prepare the report."""

def demo_summarization(text):
    summary_info = enhanced_summarize(
        text,
        strategy="map_reduce",  # or "extractive", "abstractive"
        model="sshleifer/distilbart-cnn-12-6",
        max_tokens=120
    )
    print("\n--- Summary ---\n")
    print(summary_info["summary"])
    return summary_info

if __name__ == "__main__":
    transcript = demo_transcription()
    summary = demo_summarization(transcript)
