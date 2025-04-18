import subprocess

def extract_audio_from_video(input_video_path, output_audio_path="audio.wav"):
    command = [
        "ffmpeg",
        "-i", input_video_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_audio_path
    ]
    subprocess.run(command, check=True)
    return output_audio_path
