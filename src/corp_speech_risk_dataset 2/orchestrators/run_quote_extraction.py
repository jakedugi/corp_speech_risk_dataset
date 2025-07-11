from .quote_extraction_pipeline import QuoteExtractionPipeline
import json

def main():
    pipe = QuoteExtractionPipeline()
    with open("extracted_quotes.jsonl", "w", encoding="utf8") as out:
        for doc_id, quotes in pipe.run():
            rec = {
                "doc_id": doc_id,
                "quotes": [
                    {"text": q.quote, "speaker": q.speaker, "score": q.score, "urls": q.urls}
                    for q in quotes
                ]
            }
            out.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main() 