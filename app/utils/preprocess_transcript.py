import re

def clean_transcript(text):
    # Lowercase for consistency
    text = text.lower()

    # Remove common filler words
    fillers = [
        r"\b(uh+|um+|ah+|like|you know|i mean|so+|okay+|alright+|hmm+|yeah+|mm+)\b",
        r"\b(laughs|coughs|sighs|chuckles|groans|sniffs)\b"
    ]
    for pattern in fillers:
        text = re.sub(pattern, '', text)

    # Remove timestamps (e.g., [00:01:23] or (00:01:23))
    text = re.sub(r"[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]? ?", '', text)

    # Remove speaker labels (e.g., "John:", "Speaker 1:")
    text = re.sub(r"(speaker\s?\d+:|[a-zA-Z]+:)", '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common sentence fragmentation
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)

    # Remove duplicates/repetitions (naive version)
    sentences = text.split('. ')
    seen = set()
    filtered = []
    for s in sentences:
        if s.strip() not in seen:
            filtered.append(s.strip())
            seen.add(s.strip())

    cleaned_text = '. '.join(filtered)
    return cleaned_text.strip()
