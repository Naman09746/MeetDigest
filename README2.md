
# MeetDigest
=======
# 📝 Meeting Minutes Summarizer

An AI-powered tool to automatically generate structured meeting summaries from transcripts or audio files. It extracts action items, participants, deadlines, and produces a final report in plain text format.

---

## 🚀 Features

- 📂 **Supports multiple input formats**: `.txt`, `.vtt`, `.srt`, `.mp3`, `.wav`, `.m4a`, `.webm`
- 🧹 **Cleans raw transcripts**: removes timestamps, filler words, noise
- 🗣️ **Speaker diarization** (optional via WhisperX)
- 🤖 **Named Entity Recognition**: extracts people, dates, action items
- ✂️ **Summarization** using Transformers (`DistilBART`)
- 📄 **Auto-generated meeting reports** (downloadable)
- 🧠 **Modular, testable, production-ready code**
- ⚙️ **Whisper-based audio transcription** (local)
- 📦 **Streamlit web app interface**

---

## 📸 Demo

> *You can include a screenshot or gif here*  
> Example:
> ![App Screenshot](screenshots/demo.png)

---

## 📁 Project Structure

meeting-summarizer/
├── app.py # Main Streamlit app
├── requirements.txt
├── README.md
├── modules/
│ ├── input_handler.py # Text file and subtitle parser
│ ├── transcriber.py # Whisper-based audio transcription
│ ├── diarization.py # WhisperX speaker diarization
│ ├── preprocessor.py # Cleaner, chunker, speaker segmenter
│ ├── summarizer.py # HuggingFace summarizer
│ ├── ner_extractor.py # Named entity and action item extraction
│ ├── report_generator.py # Final report generation
│ ├── date_utils.py # Fuzzy date parsing
│ ├── file_utils.py # File extension/type utilities
│ └── logger.py # Centralized logging
├── tests/
│ ├── test_ner.py
│ ├── test_preprocess.py
│ └── test_transcriber.py

---

## 📦 Installation

### 🔧 Prerequisites

- Python 3.8 or higher
- [ffmpeg](https://ffmpeg.org/download.html) (required by Whisper & WhisperX)
- (Optional) CUDA GPU for faster processing

### 📥 Clone and Install

```bash
git clone https://github.com/your-username/meeting-minutes-summarizer.git
cd meeting-minutes-summarizer
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
🧠 Run the App
streamlit run app.py
Then open http://localhost:8501 in your browser.
🎙️ Supported Input Files
Format	Description
.txt	Plain text transcript
.srt	Subtitle file (SubRip format)
.vtt	WebVTT subtitle file
.mp3, .wav, .m4a, .webm	Audio files (transcribed via Whisper)
🧪 Running Tests
pytest tests/
📌 Models Used
🤖 Summarizer: sshleifer/distilbart-cnn-12-6
🔊 Transcriber: OpenAI Whisper
🗣️ Diarizer: WhisperX
🧠 NER: spaCy en_core_web_sm
✅ Future Enhancements
📄 PDF / Markdown report exports
🧠 Fine-tuned summarization models
🔁 Real-time transcription & diarization
🌐 Hugging Face Space or Docker deployment
📃 License
MIT License
🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
✨ Credits
Built with ❤️ using Streamlit, HuggingFace Transformers, spaCy, and Whisper
📫 Contact
GitHub: @your-username
Email: your.email@example.com
