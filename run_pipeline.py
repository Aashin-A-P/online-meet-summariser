import subprocess
import os

def run_video_to_audio():
    print("\n🎬 Step 1: Extracting audio from video...")
    subprocess.run(["python", "video_to_audio_conversion.py"], check=True)

def run_transcription():
    print("\n📝 Step 2: Transcribing audio...")
    subprocess.run(["python", "transcript.py"], check=True)

def run_summarization():
    print("\n🧠 Step 3: Summarizing transcript...")
    subprocess.run(["python", "summarize.py"], check=True)

def main():
    print("🚀 Starting Full Pipeline...\n")
    run_video_to_audio()
    run_transcription()
    run_summarization()
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
