import subprocess
import tkinter as tk
from tkinter import filedialog
import sys  # ✅ Add this for exiting with error codes

def extract_audio_from_video(input_video_path, output_audio_path):
    """
    Extracts audio from a video file and converts it to mono-channel WAV at 16kHz.
    Requires ffmpeg to be installed.
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_video_path,
            '-ar', '16000',      # Set sample rate to 16kHz
            '-ac', '1',          # Set to mono audio
            '-c:a', 'pcm_s16le', # Set codec to PCM 16-bit little-endian
            output_audio_path
        ]

        print("🚀 Extracting audio from video...")
        subprocess.run(command, check=True)
        print(f"✅ Audio successfully extracted to {output_audio_path}")

    except subprocess.CalledProcessError as e:
        print("❌ Error during audio extraction:", e)
        sys.exit(1)

def upload_file():
    """
    Open a file dialog to select the video file for conversion.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(
        title="Select Video File", 
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    
    if video_path:
        print(f"📂 Selected video file: {video_path}")
        output_audio = "audio.wav"  # Set output audio file name
        extract_audio_from_video(video_path, output_audio)
    else:
        print("❌ No file selected. Exiting with error.")
        sys.exit(1)  # ✅ Exit with error code if no file is selected

# Run the file upload and extraction
upload_file()
