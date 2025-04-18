from transformers import pipeline
import textwrap

def chunk_text(text, max_chunk_size=1000):
    """Split the text into manageable chunks for summarization."""
    return textwrap.wrap(text, width=max_chunk_size, break_long_words=False, replace_whitespace=False)

def get_summarizer(model_name="facebook/bart-large-cnn", device=-1):
    """Load the summarization model pipeline."""
    return pipeline("summarization", model=model_name, device=device)

def summarize_text(summarizer, text, max_chunk_size=1000):
    """Split the text and summarize each chunk, then join the results."""
    chunks = chunk_text(text, max_chunk_size)
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"⏱️ Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=350, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return "\n".join(summaries)

# Optional CLI usage
def main():
    from pathlib import Path

    transcript_path = Path("transcript.txt")
    if not transcript_path.exists():
        print("❌ transcript.txt not found!")
        return

    text = transcript_path.read_text()
    summarizer = get_summarizer()
    summary = summarize_text(summarizer, text)

    print("\n📋 Final Summary:\n", summary)

    with open("summary.txt", "w") as f:
        f.write(summary)
    print("\n✅ Summary saved to summary.txt")

if __name__ == "__main__":
    main()
