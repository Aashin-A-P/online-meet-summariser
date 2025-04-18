from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

def refine_summary(summary_text: str, sentence_count: int = 2) -> str:
    parser = PlaintextParser.from_string(summary_text, None)
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    refined = " ".join(str(sentence) for sentence in summary_sentences)
    return refined
