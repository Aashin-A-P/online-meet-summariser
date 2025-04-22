import streamlit as st
import os
import torch
from utils.video_to_audio import extract_audio_from_video
from utils.transcriber import split_audio, transcribe_chunks
from utils.summarizer import get_summarizer, summarize_text
from utils.preprocess_transcript import clean_transcript  # Import the new preprocessing function
from utils.refine_summary import refine_summary  # Import the new refine_summary function
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

st.set_page_config(page_title="Online Meet Summarizer", layout="wide")

st.title("🎥 Online Meet Summarizer")

# Upload video
video = st.file_uploader("Upload .mp4 recording file of your meet", type=["mp4"])
if video:
    with open("temp.mp4", "wb") as f:
        f.write(video.read())

    st.info("Extracting audio...")
    audio_path = extract_audio_from_video("temp.mp4")

    st.info("Loading Whisper model...")
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to("cuda" if torch.cuda.is_available() else "cpu")

    st.info("Splitting audio...")
    chunks = split_audio(audio_path)

    st.info("Transcribing...")
    transcript = transcribe_chunks(chunks, processor, model, device="cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the transcript
    st.info("Preprocessing transcript...")
    cleaned_transcript = clean_transcript(transcript)  # Apply the cleaning

    # Calculate the word count of the transcript
    transcript_word_count = len(cleaned_transcript.split())

    st.subheader("📝 Transcript")
    st.text_area("Transcript", cleaned_transcript, height=300)
    st.info(f"Transcript word count: {transcript_word_count}")  # Display the word count of the transcript

    st.info("Summarizing...")
    summarizer = get_summarizer()
    summary = summarize_text(summarizer, cleaned_transcript)  # Summarize the cleaned transcript

    # Calculate the word count of the summary
    summary_word_count = len(summary.split())

    st.subheader("📄 Summary")
    st.success(summary)
    st.info(f"Summary word count: {summary_word_count}")  # Display the word count of the summary

    # Cleanup
    for chunk in chunks + ["temp.mp4", audio_path]:
        if os.path.exists(chunk):
            os.remove(chunk)
    st.info("Cleanup complete.")
