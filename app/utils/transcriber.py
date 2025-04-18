from pydub import AudioSegment
import math, torchaudio, torch, os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def split_audio(audio_path, chunk_length_ms=30000, overlap_ms=2000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    num_chunks = math.ceil(len(audio) / (chunk_length_ms - overlap_ms))
    print(f"🔪 Splitting audio into {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * (chunk_length_ms - overlap_ms)
        end = min(len(audio), start + chunk_length_ms)
        chunk = audio[start:end]
        fname = f"chunk_{i}.wav"
        chunk.export(fname, format="wav")
        chunks.append(fname)
        print(f"✅ Created chunk: {fname} (from {start} ms to {end} ms)")

    return chunks

def transcribe_chunks(chunks, processor, model, device):
    transcript = []
    print(f"\n🧠 Transcribing {len(chunks)} chunks...\n")

    for idx, chunk in enumerate(chunks):
        print(f"🎧 Processing chunk {idx + 1}/{len(chunks)}: {chunk}")
        waveform, sr = torchaudio.load(chunk)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(inputs["input_features"], max_new_tokens=444)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        transcript.append(text)
        print(f"📝 Transcription for chunk {idx + 1} done.")

    print("\n🧹 Cleaning up temporary chunk files...")
    for file in chunks:
        os.remove(file)
        print(f"🗑️ Deleted {file}")

    return "\n".join(transcript)
