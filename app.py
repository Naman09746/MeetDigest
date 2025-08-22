# app.py

import streamlit as st
from modules.logger import logger
from modules import (
    file_utils,
    input_handler,
    transcriber,
    preprocessor,
    diarization,
    ner_extractor,
    summarizer,
    report_generator
)

# App setup
st.set_page_config(page_title="📝 Meeting Summarizer", layout="wide")
st.title("📝 AI-Powered Meeting Summarizer")

st.markdown("""
Upload your meeting transcript or audio file (supported: `.txt`, `.srt`, `.vtt`, `.mp3`, `.wav`, `.m4a`, `.webm`).  
The app will clean it, identify action items, participants, and deadlines, and generate a downloadable meeting report.
""")

# Upload section
uploaded_file = st.file_uploader("📂 Upload File", type=file_utils.SUPPORTED_ALL)

if uploaded_file:
    file_ext = file_utils.get_file_extension(uploaded_file)

    if not file_utils.is_supported_file(uploaded_file):
        st.error(f"❌ Unsupported file format: .{file_ext}")
        st.stop()

    raw_text = None
    speaker_segments = []

    try:
        # Process audio or text
        if file_utils.is_text_file(uploaded_file):
            with st.spinner("🧾 Reading transcript..."):
                if file_ext == 'txt':
                    raw_text = input_handler.read_text_file(uploaded_file)
                elif file_ext == 'srt':
                    raw_text = input_handler.parse_srt(uploaded_file)
                elif file_ext == 'vtt':
                    raw_text = input_handler.parse_vtt(uploaded_file)

        elif file_utils.is_audio_file(uploaded_file):
            st.markdown("🔊 **Audio Processing Options**")
            use_diarization = st.checkbox("🗣️ Use Speaker Diarization (WhisperX)", value=False)
            model_choice = st.selectbox("🎯 Whisper Model", ["base", "small", "medium", "large"], index=0)

            with st.spinner("Transcribing audio..."):
                if use_diarization:
                    diarizer = diarization.DiarizationService()
                    speaker_segments = diarizer.transcribe_and_diarize(uploaded_file, extension=file_ext)
                    raw_text = " ".join([text for _, text in speaker_segments])
                else:
                    raw_text = transcriber.transcribe_audio(uploaded_file, extension=file_ext, model_name=model_choice)

        # Display raw transcript
        if raw_text:
            st.subheader("📄 Raw Transcript")
            st.text_area("Raw Transcript", raw_text, height=300)

            # Clean transcript
            with st.spinner("🧹 Cleaning transcript..."):
                cleaned_text = preprocessor.clean_transcript(raw_text)

            st.subheader("🧹 Cleaned Transcript")
            st.text_area("Cleaned", cleaned_text, height=250)

            # Segment by speaker (if not using WhisperX already)
            if not speaker_segments:
                speaker_segments = preprocessor.segment_by_speaker(cleaned_text)

            st.subheader("🗣️ Speaker Segments")
            for speaker, segment in speaker_segments:
                st.markdown(f"**{speaker}:** {segment}")

            # Chunk for summarizer
            with st.spinner("🧩 Chunking for summarizer..."):
                chunks = preprocessor.chunk_text(cleaned_text)

            st.subheader(f"🧠 Prepared {len(chunks)} Chunks")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i + 1}"):
                    st.write(chunk)

            # Named entity extraction
            st.subheader("🔍 Entity Extraction")
            with st.spinner("Extracting people, dates, action items..."):
                entities = ner_extractor.extract_entities(cleaned_text)

            st.markdown("#### 👤 People")
            if entities["people"]:
                for person in entities["people"]:
                    st.markdown(f"- {person}")
            else:
                st.info("No participants identified.")

            st.markdown("#### 📅 Deadlines / Dates")
            if entities["dates"]:
                for date in entities["dates"]:
                    st.markdown(f"- {date}")
            else:
                st.info("No dates found.")

            st.markdown("#### ✅ Action Items")
            if entities["action_items"]:
                for action in entities["action_items"]:
                    st.markdown(f"- {action}")
            else:
                st.info("No action items found.")

            # Summary generation
            st.subheader("🧠 Generate Summary")
            if st.button("🚀 Summarize"):
                with st.spinner("Summarizing..."):
                    summary = summarizer.summarize_chunks(chunks)
                    st.success("✅ Summary generated.")
                    st.text_area("Meeting Summary", summary, height=300)

                    # Generate final report
                    report = report_generator.generate_meeting_report(
                        participants=entities["people"],
                        action_items=entities["action_items"],
                        deadlines=entities["dates"],
                        summary=summary,
                        original_transcript=cleaned_text
                    )

                    st.subheader("📄 Download Report")
                    st.download_button(
                        label="📥 Download .txt Report",
                        data=report,
                        file_name="meeting_summary_report.txt",
                        mime="text/plain"
                    )

                    with st.expander("📖 View Full Report"):
                        st.text_area("Full Report", report, height=400)

    except Exception as e:
        logger.exception("❌ Unexpected error in app.")
        st.error(f"An error occurred: {e}")
