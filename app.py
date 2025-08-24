# app.py

import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
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

# Business Logic Layer
class MeetingSummarizerService:
    """Service layer that handles all business logic without UI dependencies"""
    
    def __init__(self):
        self.diarizer = None
    
    def process_text_file(self, uploaded_file, file_ext: str) -> Tuple[bool, str, str]:
        """
        Process text files (txt, srt, vtt)
        Returns: (success, raw_text, error_message)
        """
        try:
            if file_ext == 'txt':
                raw_text = input_handler.read_text_file(uploaded_file)
            elif file_ext == 'srt':
                raw_text = input_handler.parse_srt(uploaded_file)
            elif file_ext == 'vtt':
                raw_text = input_handler.parse_vtt(uploaded_file)
            else:
                return False, "", f"Unsupported text format: {file_ext}"
            
            return True, raw_text, ""
        except Exception as e:
            logger.exception(f"Error processing text file: {e}")
            return False, "", f"Failed to process text file: {str(e)}"
    
    def process_audio_file(self, uploaded_file, file_ext: str, 
                          use_diarization: bool = False, 
                          model_choice: str = "base") -> Tuple[bool, str, List[Tuple[str, str]], str]:
        """
        Process audio files with optional diarization
        Returns: (success, raw_text, speaker_segments, error_message)
        """
        try:
            speaker_segments = []
            
            if use_diarization:
                if not self.diarizer:
                    self.diarizer = diarization.DiarizationService()
                speaker_segments = self.diarizer.transcribe_and_diarize(uploaded_file, extension=file_ext)
                raw_text = " ".join([text for _, text in speaker_segments])
            else:
                raw_text = transcriber.transcribe_audio(uploaded_file, extension=file_ext, model_name=model_choice)
            
            return True, raw_text, speaker_segments, ""
        except Exception as e:
            logger.exception(f"Error processing audio file: {e}")
            return False, "", [], f"Failed to process audio file: {str(e)}"
    
    def clean_and_prepare_transcript(self, raw_text: str) -> Tuple[bool, str, List[str], str]:
        """
        Clean transcript and prepare chunks
        Returns: (success, cleaned_text, chunks, error_message)
        """
        try:
            cleaned_text = preprocessor.clean_transcript(raw_text)
            chunks = preprocessor.chunk_text(cleaned_text)
            return True, cleaned_text, chunks, ""
        except Exception as e:
            logger.exception(f"Error cleaning transcript: {e}")
            return False, "", [], f"Failed to clean transcript: {str(e)}"
    
    def get_speaker_segments(self, cleaned_text: str, existing_segments: List[Tuple[str, str]] = None) -> Tuple[bool, List[Tuple[str, str]], str]:
        """
        Get speaker segments (use existing if available)
        Returns: (success, speaker_segments, error_message)
        """
        try:
            if existing_segments:
                return True, existing_segments, ""
            
            speaker_segments = preprocessor.segment_by_speaker(cleaned_text)
            return True, speaker_segments, ""
        except Exception as e:
            logger.exception(f"Error segmenting speakers: {e}")
            return False, [], f"Failed to segment speakers: {str(e)}"
    
    def extract_entities(self, cleaned_text: str) -> Tuple[bool, Dict[str, List[str]], str]:
        """
        Extract named entities
        Returns: (success, entities_dict, error_message)
        """
        try:
            entities = ner_extractor.extract_entities(cleaned_text)
            return True, entities, ""
        except Exception as e:
            logger.exception(f"Error extracting entities: {e}")
            return False, {}, f"Failed to extract entities: {str(e)}"
    
    def generate_summary(self, chunks: List[str]) -> Tuple[bool, str, str]:
        """
        Generate summary from chunks
        Returns: (success, summary, error_message)
        """
        try:
            summary = summarizer.summarize_chunks(chunks)
            return True, summary, ""
        except Exception as e:
            logger.exception(f"Error generating summary: {e}")
            return False, "", f"Failed to generate summary: {str(e)}"
    
    def generate_report(self, participants: List[str], action_items: List[str], 
                       deadlines: List[str], summary: str, transcript: str) -> Tuple[bool, str, str]:
        """
        Generate final report
        Returns: (success, report, error_message)
        """
        try:
            report = report_generator.generate_meeting_report(
                participants=participants,
                action_items=action_items,
                deadlines=deadlines,
                summary=summary,
                original_transcript=transcript
            )
            return True, report, ""
        except Exception as e:
            logger.exception(f"Error generating report: {e}")
            return False, "", f"Failed to generate report: {str(e)}"

# UI Layer
class MeetingSummarizerUI:
    """UI layer that handles all Streamlit interactions"""
    
    def __init__(self):
        self.service = MeetingSummarizerService()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'raw_text': '',
            'cleaned_text': '',
            'chunks': [],
            'speaker_segments': [],
            'entities': {},
            'summary': '',
            'report': '',
            'processing_stage': 'upload',
            'errors': []
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _add_error(self, error_message: str):
        """Add error to session state and display"""
        st.session_state.errors.append(error_message)
        st.error(f"❌ {error_message}")
        logger.error(error_message)
    
    def _clear_errors(self):
        """Clear all errors from session state"""
        st.session_state.errors = []
    
    def _reset_processing_state(self):
        """Reset processing state for new file"""
        keys_to_reset = ['raw_text', 'cleaned_text', 'chunks', 'speaker_segments', 
                        'entities', 'summary', 'report']
        for key in keys_to_reset:
            st.session_state[key] = [] if key in ['chunks', 'speaker_segments'] else {} if key == 'entities' else ''
        st.session_state.processing_stage = 'upload'
        self._clear_errors()
    
    def render_file_upload(self):
        """Render file upload section"""
        st.markdown("""
        Upload your meeting transcript or audio file (supported: `.txt`, `.srt`, `.vtt`, `.mp3`, `.wav`, `.m4a`, `.webm`).  
        The app will clean it, identify action items, participants, and deadlines, and generate a downloadable meeting report.
        """)
        
        uploaded_file = st.file_uploader("📂 Upload File", type=file_utils.SUPPORTED_ALL)
        
        if uploaded_file:
            # Reset state for new file
            if st.session_state.get('current_file') != uploaded_file.name:
                self._reset_processing_state()
                st.session_state.current_file = uploaded_file.name
            
            return uploaded_file
        return None
    
    def process_file(self, uploaded_file):
        """Process uploaded file based on type"""
        file_ext = file_utils.get_file_extension(uploaded_file)
        
        if not file_utils.is_supported_file(uploaded_file):
            self._add_error(f"Unsupported file format: .{file_ext}")
            return False
        
        # Process based on file type
        if file_utils.is_text_file(uploaded_file):
            return self._process_text_file(uploaded_file, file_ext)
        elif file_utils.is_audio_file(uploaded_file):
            return self._process_audio_file(uploaded_file, file_ext)
        else:
            self._add_error(f"Unknown file type: {file_ext}")
            return False
    
    def _process_text_file(self, uploaded_file, file_ext: str) -> bool:
        """Process text file with error handling"""
        with st.spinner("🧾 Reading transcript..."):
            success, raw_text, error = self.service.process_text_file(uploaded_file, file_ext)
            
            if success:
                st.session_state.raw_text = raw_text
                st.session_state.speaker_segments = []  # No speaker segments from text files
                st.session_state.processing_stage = 'transcribed'
                return True
            else:
                self._add_error(error)
                return False
    
    def _process_audio_file(self, uploaded_file, file_ext: str) -> bool:
        """Process audio file with error handling"""
        # Audio processing options
        st.markdown("🔊 **Audio Processing Options**")
        use_diarization = st.checkbox("🗣️ Use Speaker Diarization (WhisperX)", value=False)
        model_choice = st.selectbox("🎯 Whisper Model", ["base", "small", "medium", "large"], index=0)
        
        if st.button("🎵 Process Audio"):
            with st.spinner("Transcribing audio..."):
                success, raw_text, speaker_segments, error = self.service.process_audio_file(
                    uploaded_file, file_ext, use_diarization, model_choice
                )
                
                if success:
                    st.session_state.raw_text = raw_text
                    st.session_state.speaker_segments = speaker_segments
                    st.session_state.processing_stage = 'transcribed'
                    st.rerun()
                    return True
                else:
                    self._add_error(error)
                    return False
        return False
    
    def render_transcript_section(self):
        """Render transcript display and cleaning"""
        if st.session_state.processing_stage == 'transcribed' and st.session_state.raw_text:
            # Display raw transcript
            st.subheader("📄 Raw Transcript")
            st.text_area("Raw Transcript", st.session_state.raw_text, height=300, key="raw_display")
            
            # Clean transcript
            if st.button("🧹 Clean Transcript") or st.session_state.processing_stage == 'transcribed':
                with st.spinner("🧹 Cleaning transcript..."):
                    success, cleaned_text, chunks, error = self.service.clean_and_prepare_transcript(st.session_state.raw_text)
                    
                    if success:
                        st.session_state.cleaned_text = cleaned_text
                        st.session_state.chunks = chunks
                        st.session_state.processing_stage = 'cleaned'
                        st.rerun()
                    else:
                        self._add_error(error)
    
    def render_cleaned_transcript_section(self):
        """Render cleaned transcript and speaker segments"""
        if st.session_state.processing_stage in ['cleaned', 'analyzed'] and st.session_state.cleaned_text:
            st.subheader("🧹 Cleaned Transcript")
            st.text_area("Cleaned", st.session_state.cleaned_text, height=250, key="cleaned_display")
            
            # Get speaker segments
            success, speaker_segments, error = self.service.get_speaker_segments(
                st.session_state.cleaned_text, st.session_state.speaker_segments
            )
            
            if success:
                st.session_state.speaker_segments = speaker_segments
                
                st.subheader("🗣️ Speaker Segments")
                if speaker_segments:
                    for speaker, segment in speaker_segments:
                        st.markdown(f"**{speaker}:** {segment}")
                else:
                    st.info("No speaker segments identified")
            else:
                self._add_error(error)
            
            # Show chunks
            if st.session_state.chunks:
                st.subheader(f"🧩 Prepared {len(st.session_state.chunks)} Chunks")
                for i, chunk in enumerate(st.session_state.chunks):
                    with st.expander(f"Chunk {i + 1}"):
                        st.write(chunk)
    
    def render_entity_extraction_section(self):
        """Render entity extraction section"""
        if st.session_state.processing_stage in ['cleaned', 'analyzed'] and st.session_state.cleaned_text:
            st.subheader("🔍 Entity Extraction")
            
            if st.button("🔍 Extract Entities") or st.session_state.processing_stage == 'cleaned':
                with st.spinner("Extracting people, dates, action items..."):
                    success, entities, error = self.service.extract_entities(st.session_state.cleaned_text)
                    
                    if success:
                        st.session_state.entities = entities
                        st.session_state.processing_stage = 'analyzed'
                        st.rerun()
                    else:
                        self._add_error(error)
                        return
            
            # Display entities
            if st.session_state.entities:
                entities = st.session_state.entities
                
                st.markdown("#### 👤 People")
                if entities.get("people"):
                    for person in entities["people"]:
                        st.markdown(f"- {person}")
                else:
                    st.info("No participants identified.")
                
                st.markdown("#### 📅 Deadlines / Dates")
                if entities.get("dates"):
                    for date in entities["dates"]:
                        st.markdown(f"- {date}")
                else:
                    st.info("No dates found.")
                
                st.markdown("#### ✅ Action Items")
                if entities.get("action_items"):
                    for action in entities["action_items"]:
                        st.markdown(f"- {action}")
                else:
                    st.info("No action items found.")
    
    def render_summary_section(self):
        """Render summary generation section"""
        if st.session_state.processing_stage == 'analyzed' and st.session_state.chunks:
            st.subheader("🧠 Generate Summary")
            
            if st.button("🚀 Summarize"):
                with st.spinner("Summarizing..."):
                    success, summary, error = self.service.generate_summary(st.session_state.chunks)
                    
                    if success:
                        st.session_state.summary = summary
                        st.success("✅ Summary generated.")
                        st.rerun()
                    else:
                        self._add_error(error)
                        return
            
            # Display summary
            if st.session_state.summary:
                st.text_area("Meeting Summary", st.session_state.summary, height=300, key="summary_display")
                
                # Generate report
                self._generate_and_display_report()
    
    def _generate_and_display_report(self):
        """Generate and display the final report"""
        if not st.session_state.report:
            entities = st.session_state.entities
            success, report, error = self.service.generate_report(
                participants=entities.get("people", []),
                action_items=entities.get("action_items", []),
                deadlines=entities.get("dates", []),
                summary=st.session_state.summary,
                transcript=st.session_state.cleaned_text
            )
            
            if success:
                st.session_state.report = report
            else:
                self._add_error(error)
                return
        
        # Display download options
        if st.session_state.report:
            st.subheader("📄 Download Report")
            st.download_button(
                label="📥 Download .txt Report",
                data=st.session_state.report,
                file_name="meeting_summary_report.txt",
                mime="text/plain"
            )
            
            with st.expander("📖 View Full Report"):
                st.text_area("Full Report", st.session_state.report, height=400, key="report_display")
    
    def render_error_summary(self):
        """Render error summary if there are errors"""
        if st.session_state.errors:
            with st.expander("⚠️ Error Summary", expanded=len(st.session_state.errors) > 0):
                for i, error in enumerate(st.session_state.errors, 1):
                    st.error(f"{i}. {error}")
                
                if st.button("🧹 Clear Errors"):
                    self._clear_errors()
                    st.rerun()
    
    def run(self):
        """Main application runner"""
        # App setup
        st.set_page_config(page_title="📝 Meeting Summarizer", layout="wide")
        st.title("📝 AI-Powered Meeting Summarizer")
        
        # File upload
        uploaded_file = self.render_file_upload()
        
        if uploaded_file:
            # Process file
            if st.session_state.processing_stage == 'upload':
                self.process_file(uploaded_file)
            
            # Render sections based on processing stage
            self.render_transcript_section()
            self.render_cleaned_transcript_section()
            self.render_entity_extraction_section()
            self.render_summary_section()
        
        # Error summary
        self.render_error_summary()

# Application Entry Point
def main():
    """Main entry point"""
    try:
        app = MeetingSummarizerUI()
        app.run()
    except Exception as e:
        logger.exception("Critical application error")
        st.error("🚨 Critical application error occurred. Please check logs and try again.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()