# app.py

import streamlit as st
from typing import Optional
from modules.logger import logger
from modules.meeting_context import MeetingContext

from modules import (
    file_utils,
    input_handler,
    transcriber,
    diarization,
    ner_extractor,
    summarizer,
    report_generator
)

# ============================================================
# UI CONFIG + THEME
# ============================================================

st.set_page_config(
    page_title="📝 Meeting Minutes AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stMetric { background-color: #0e1117; padding: 14px; border-radius: 12px; }
textarea { border-radius: 12px !important; }
.stDownloadButton button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SERVICE LAYER
# ============================================================

class MeetingSummarizerService:
    """Runs the full MeetingContext pipeline"""

    def __init__(self):
        self.diarizer = None

    def run_pipeline(
        self,
        uploaded_file,
        file_ext: str,
        use_diarization: bool = False,
        model_choice: str = "base"
    ):
        try:
            # -------- 1. TRANSCRIPTION / TEXT INGESTION --------
            if file_utils.is_text_file(uploaded_file):
                if file_ext == "txt":
                    raw_text = input_handler.read_text_file(uploaded_file)
                elif file_ext == "srt":
                    raw_text = input_handler.parse_srt(uploaded_file).plain_text
                elif file_ext == "vtt":
                    raw_text = input_handler.parse_vtt(uploaded_file).plain_text
                else:
                    raise ValueError("Unsupported text format")

                context = MeetingContext(raw_text=raw_text)
            else:
                context = transcriber.transcribe_audio(
                    uploaded_file,
                    extension=file_ext,
                    model_name=model_choice
                )

            # -------- 2. OPTIONAL DIARIZATION --------
            if use_diarization and file_utils.is_audio_file(uploaded_file):
                if not self.diarizer:
                    self.diarizer = diarization.DiarizationService()

                context = self.diarizer.transcribe_and_diarize(
                    uploaded_file,
                    extension=file_ext,
                    context=context
                )

            # -------- 3. NER --------
            context = ner_extractor.enrich_context_with_entities(context)

            # -------- 4. SUMMARY --------
            context = summarizer.enrich_context_with_summary(context)

            # -------- 5. REPORT --------
            report = report_generator.generate_report_from_context(context)

            return True, context, report, ""

        except Exception as e:
            logger.exception("Pipeline execution failed")
            return False, None, None, str(e)


# ============================================================
# UI LAYER
# ============================================================

class MeetingSummarizerUI:

    def __init__(self):
        self.service = MeetingSummarizerService()
        self._init_session_state()

    # ---------------- SESSION STATE ----------------

    def _init_session_state(self):
        defaults = {
            "context": None,
            "report": None,
            "processing_stage": "upload",
            "errors": [],
            "current_file": None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _add_error(self, msg: str):
        st.session_state.errors.append(msg)
        logger.error(msg)

    def _clear_errors(self):
        st.session_state.errors = []

    # ---------------- NAVIGATION ----------------

    def sidebar_navigation(self):
        st.sidebar.title("🧭 Navigation")
        return st.sidebar.radio(
            "Go to",
            ["🏠 Home", "🎧 Process Meeting", "📊 Insights", "📄 Report"]
        )

    # ---------------- FILE UPLOAD ----------------

    def render_file_upload(self):
        st.markdown("### 📂 Upload Meeting File")
        st.markdown("Supported: **TXT, SRT, VTT, MP3, WAV, M4A, WEBM**")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=file_utils.SUPPORTED_ALL
        )

        if uploaded_file:
            if st.session_state.current_file != uploaded_file.name:
                self._clear_errors()
                st.session_state.processing_stage = "upload"
                st.session_state.context = None
                st.session_state.report = None
                st.session_state.current_file = uploaded_file.name

        return uploaded_file

    # ---------------- PROCESS FILE ----------------

    def process_file(self, uploaded_file):
        file_ext = file_utils.get_file_extension(uploaded_file)

        if not file_utils.is_supported_file(uploaded_file):
            self._add_error(f"Unsupported file format: .{file_ext}")
            return

        if file_utils.is_audio_file(uploaded_file):
            self._process_audio_file(uploaded_file, file_ext)
        else:
            self._process_text_file(uploaded_file, file_ext)

    def _process_text_file(self, uploaded_file, file_ext: str):
        with st.spinner("🧾 Processing transcript..."):
            success, context, report, error = self.service.run_pipeline(
                uploaded_file,
                file_ext,
                use_diarization=False
            )

            if success:
                st.session_state.context = context
                st.session_state.report = report
                st.session_state.processing_stage = "done"
                st.rerun()
            else:
                self._add_error(error)

    def _process_audio_file(self, uploaded_file, file_ext: str):
        st.subheader("🔊 Audio Options")

        col1, col2 = st.columns(2)
        with col1:
            use_diarization = st.checkbox("🗣️ Enable Speaker Diarization", value=False)
        with col2:
            model_choice = st.selectbox(
                "🎯 Whisper Model",
                ["base", "small", "medium", "large"],
                index=0
            )

        if st.button("🚀 Run Pipeline"):
            with st.spinner("🎧 Running full AI pipeline..."):
                success, context, report, error = self.service.run_pipeline(
                    uploaded_file,
                    file_ext,
                    use_diarization=use_diarization,
                    model_choice=model_choice
                )

                if success:
                    st.session_state.context = context
                    st.session_state.report = report
                    st.session_state.processing_stage = "done"
                    st.rerun()
                else:
                    self._add_error(error)

    # ---------------- PAGES ----------------

    def render_home(self):
        st.markdown("## 📝 Meeting Minutes Summarizer")
        st.markdown("""
        **AI-powered system to convert meetings into structured insights.**

        ### ✨ Features
        - 🎙️ Audio transcription (Whisper)
        - 🗣️ Speaker diarization (WhisperX)
        - 🧠 Entity & action extraction
        - 📄 Automatic meeting minutes
        - 📥 Downloadable reports

        ### 🚀 How it works
        1. Upload meeting audio or transcript  
        2. Run the AI pipeline  
        3. Review insights & download report  
        """)

    def render_process_page(self):
        uploaded_file = self.render_file_upload()
        if uploaded_file and st.session_state.processing_stage == "upload":
            self.process_file(uploaded_file)

    def render_insights(self):
        context: Optional[MeetingContext] = st.session_state.context
        if not context:
            st.info("Upload and process a meeting first.")
            return

        st.markdown("## 📊 Meeting Insights")

        col1, col2, col3 = st.columns(3)
        col1.metric("🗣️ Speakers", len(context.speaker_stats))
        col2.metric("📄 Words", context.metadata.get("total_words", 0))
        col3.metric("⏱️ Duration (sec)", int(context.metadata.get("duration", 0)))

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📄 Transcript", "🗣️ Speakers", "🧠 Entities"])

        with tab1:
            st.text_area("Transcript", context.raw_text, height=300)

        with tab2:
            if context.speaker_segments:
                for speaker, text in context.speaker_segments:
                    with st.expander(f"🎤 {speaker}"):
                        st.write(text)
            else:
                st.info("No diarization data available.")

        with tab3:
            st.json(context.metadata.get("entities", {}))

    def render_report_page(self):
        context = st.session_state.context
        report = st.session_state.report

        if not context:
            st.info("No report generated yet.")
            return

        st.markdown("## 📄 Final Report")

        st.subheader("🧠 Executive Summary")
        st.text_area(
            "Summary",
            context.metadata.get("summary", ""),
            height=220
        )

        if report:
            st.download_button(
                "📥 Download Meeting Report",
                data=str(report),
                file_name="meeting_report.txt"
            )

    def render_error_summary(self):
        if st.session_state.errors:
            with st.expander("⚠️ Errors", expanded=True):
                for err in st.session_state.errors:
                    st.error(err)
                if st.button("Clear Errors"):
                    self._clear_errors()
                    st.rerun()

    # ---------------- MAIN RUN ----------------

    def run(self):
        page = self.sidebar_navigation()

        if page == "🏠 Home":
            self.render_home()
        elif page == "🎧 Process Meeting":
            self.render_process_page()
        elif page == "📊 Insights":
            self.render_insights()
        elif page == "📄 Report":
            self.render_report_page()

        self.render_error_summary()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    try:
        app = MeetingSummarizerUI()
        app.run()
    except Exception as e:
        logger.exception("Critical application error")
        st.error("🚨 Critical error occurred. Please check logs.")
        st.error(str(e))


if __name__ == "__main__":
    main()
