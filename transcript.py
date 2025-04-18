from pydub import AudioSegment
import math
import torch
import torchaudio
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def split_audio_into_chunks(audio_path, chunk_length_ms=30000, overlap_ms=2000):
    """
    Splits an audio file into smaller chunks for processing.
    Args:
        audio_path (str): Path to the audio file.
        chunk_length_ms (int): Length of each chunk in milliseconds (default: 30 seconds).
        overlap_ms (int): Overlap between chunks in milliseconds (default: 2 seconds).
    Returns:
        list: A list of file paths to audio chunks.
    """
    audio = AudioSegment.from_wav(audio_path)
    num_chunks = math.ceil(len(audio) / (chunk_length_ms - overlap_ms))
    print(f"🔪 Splitting into {num_chunks} chunks...")

    chunks = []
    for i in range(num_chunks):
        start = i * (chunk_length_ms - overlap_ms)
        end = min(len(audio), start + chunk_length_ms)
        chunk = audio[start:end]
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def transcribe_chunks(chunks, device, processor, model):
    """
    Transcribes audio chunks into text.
    Args:
        chunks (list): List of chunk file paths.
        device (str): Device to run the model on ("cuda" or "cpu").
        processor: Processor to handle audio preprocessing.
        model: Pre-trained model for transcription.
    Returns:
        str: Final combined transcript from all chunks.
    """
    full_transcript = []
    
    for i, chunk in enumerate(chunks):
        waveform, sample_rate = torchaudio.load(chunk)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(device)

        # Generate with safe token limit
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"], max_new_tokens=444)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"✅ Chunk {i+1}/{len(chunks)} done.")
        full_transcript.append(transcription)

    return "\n".join(full_transcript)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model and processor
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to(device)
    
    audio_path = "audio.wav"  # Path to your audio file
    
    # Step 1: Split the audio into chunks
    chunks = split_audio_into_chunks(audio_path)
    
    # Step 2: Transcribe each chunk
    full_transcript = transcribe_chunks(chunks, device, processor, model)
    
    print("📝 Final Transcript:\n")
    print(full_transcript[:3000])  # Just print first 3000 chars to keep it short

    # Optional: Save to file
    with open("transcript.txt", "w") as f:
        f.write(full_transcript)
    
    for chunk in chunks:
        os.remove(chunk)
        
if __name__ == "__main__":
    main()
