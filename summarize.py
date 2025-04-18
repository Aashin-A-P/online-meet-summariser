from transformers import pipeline
import textwrap

def load_transcript(file_path="transcript.txt"):
    """Load the full transcript from a text file."""
    with open(file_path, "r") as f:
        return f.read()

def chunk_text(text, max_chunk_size=1000):
    """Split the text into manageable chunks for summarization."""
    return textwrap.wrap(text, width=max_chunk_size, break_long_words=False, replace_whitespace=False)

def summarize_chunks(chunks, model_name="facebook/bart-large-cnn"):
    """Summarize each text chunk using the summarization pipeline."""
    summarizer = pipeline("summarization", model=model_name, device=-1)  # CPU
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"⏱️ Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=350, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return "\n".join(summaries)

def main():
    transcript = load_transcript()
    chunks = chunk_text(transcript, max_chunk_size=1000)
    final_summary = summarize_chunks(chunks)

    print("\n📋 Final Summary:\n", final_summary)

    # Optional: Save the summary
    with open("summary.txt", "w") as f:
        f.write(final_summary)
    print("\n✅ Summary saved to summary.txt")

if __name__ == "__main__":
    main()
