# 📝 Meeting Minutes Summariser (AI-Powered)

An end-to-end **AI-powered Meeting Minutes Summariser** that transforms raw meeting audio or transcripts into **structured insights**, including:
- Clean transcripts
- Speaker diarization
- Named Entity Recognition (NER)
- Action items & key dates
- Executive summaries
- Downloadable professional reports

Built with **modern NLP and Speech AI pipelines**, this project is designed to be **robust, modular, and production-ready**.

---

## 🚀 Features

- 🎙️ **Audio Transcription** using Whisper / Faster-Whisper  
- 🗣️ **Speaker Diarization** using WhisperX + PyAnnote  
- 🧠 **Named Entity Recognition (NER)** (people, dates, action items)  
- ✍️ **Automatic Meeting Summaries**  
- 📄 **Professional Report Generation** (TXT / PDF-ready)  
- 🖥️ **Interactive Streamlit UI** with multi-page navigation  
- ⚙️ **Clean Pipeline Architecture** using a shared `MeetingContext`  

---

## 🏗️ Architecture Overview

```text
Upload File
   ↓
Transcription (Audio/Text)
   ↓
(Optional) Speaker Diarization
   ↓
Entity Extraction (NER)
   ↓
Summarization
   ↓
Report Generation
   ↓
Streamlit UI Output
```

All stages share a single immutable data object: **`MeetingContext`**, ensuring clean data flow and easy extensibility.

---

## 📂 Project Structure

```text
meeting_minutes_summariser/
│
├── app.py                     # Streamlit application (UI + orchestration)
├── requirements.txt           # Project dependencies
│
├── modules/
│   ├── meeting_context.py     # Central pipeline data structure
│   ├── transcriber.py         # Audio transcription logic
│   ├── diarisation.py         # Speaker diarization
│   ├── ner_extractor.py       # Named Entity Recognition
│   ├── summariser.py          # Text summarization
│   ├── report_generator.py    # Report creation
│   ├── input_handler.py       # TXT / SRT / VTT parsing
│   ├── date_utils.py          # Date parsing utilities
│   └── logger.py              # Logging setup
│
└── venv/ (optional)           # Virtual environment
```

---

## 🧪 Supported Input Formats

- **Audio**: `mp3`, `wav`, `m4a`, `webm`
- **Text**: `txt`, `srt`, `vtt`

---

## ⚙️ Installation & Setup

### 1️⃣ Prerequisites
- **Python 3.10 or 3.11** (Python 3.12 is NOT supported)
- `ffmpeg` installed on your system

```bash
# macOS
brew install ffmpeg
```

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/meeting_minutes_summariser.git
cd meeting_minutes_summariser
```

---

### 3️⃣ Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

---

## 🧪 Testing Checklist

- ✅ Upload `.txt` file → transcript, entities & summary appear
- ✅ Upload short `.mp3` audio → transcription works
- ✅ Enable diarization → speaker segments visible
- ✅ Download report successfully
- ✅ App handles unsupported files gracefully

---

## 📊 Example Use Cases

- Corporate meeting summarization  
- Academic seminar transcription  
- Interview & discussion analysis  
- Project review documentation  

---

## 🛠️ Tech Stack

- **Speech AI**: Whisper, Faster-Whisper, WhisperX  
- **Diarization**: PyAnnote, SpeechBrain  
- **NLP**: spaCy, Transformers, NLTK  
- **UI**: Streamlit  
- **Reports**: ReportLab, Python-Docx  
- **Language**: Python  

---

## 🧠 Key Engineering Highlights

- Context-driven pipeline (`MeetingContext`)
- Clear separation of UI, services, and utilities
- Graceful error handling & logging
- Production-style architecture suitable for real-world deployment

---

## 📌 Future Enhancements

- 📊 Visual analytics (speaker talk-time charts)
- 🌐 Deployment on Streamlit Cloud / HuggingFace Spaces
- 📁 Export to PDF & DOCX
- 🔐 Authentication & user sessions

---

## 👨‍💻 Author

**Naman Joshi**  
B.Tech CSE (AI & ML)  
GitHub: https://github.com/Naman09746  
LinkedIn: https://www.linkedin.com/in/naman-joshi0313/

---

## ⭐ If you like this project

Please consider giving it a **star ⭐ on GitHub**!
