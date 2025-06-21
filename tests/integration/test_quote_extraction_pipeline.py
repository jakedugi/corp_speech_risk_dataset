
import json
from pathlib import Path
from corp_speech_risk_dataset.orchestrators import quote_extraction_config as config
from corp_speech_risk_dataset.extractors.quote_extractor import QuoteExtractor
from corp_speech_risk_dataset.extractors.attribution import Attributor

class QuoteExtractionPipeline:
    def __init__(self):
        # Dynamically read config values (supports monkeypatching)
        self.json_dir = Path(config.JSON_DIR)
        self.txt_dir = Path(config.TXT_DIR)
        self.keywords = config.KEYWORDS
        self.aliases = config.COMPANY_ALIASES
        self.seed_quotes = config.SEED_QUOTES
        self.threshold = config.THRESHOLD

        # Debug loader paths
        print(f"[PRINT Loader] runtime JSON_DIR={self.json_dir}, TXT_DIR={self.txt_dir}")

        if not self.json_dir.exists() or not self.txt_dir.exists():
            raise FileNotFoundError(
                f"Configured pipeline directories do not exist: {self.json_dir}, {self.txt_dir}"
            )

    def run(self):
        """
        Iterate over document pairs in JSON_DIR and TXT_DIR, extract quotes,
        and yield (doc_id, quotes) tuples.
        """
        for json_path in sorted(self.json_dir.glob("*.json")):
            stem = json_path.stem
            txt_path = self.txt_dir / f"{stem}.txt"
            if not txt_path.exists():
                continue

            # Initialize NLP and coreference attributor
            _ = Attributor(self.aliases)

            # Extract quotes using configured parameters
            extractor = QuoteExtractor(
                json_path=json_path,
                txt_path=txt_path,
                keywords=self.keywords,
                aliases=self.aliases,
                seed_quotes=self.seed_quotes,
                threshold=self.threshold
            )
            quotes = extractor.extract()

            if quotes:
                yield stem, quotes

    def save_results(self, results, output_file="extracted_quotes.jsonl"):
        """
        Write the extracted results to a JSONL file, one record per line:
        {
            "doc_id": str,
            "quotes": [
                {"text", "speaker", "score", "urls"}, ...
            ]
        }
        """
        with open(output_file, "w", encoding="utf8") as out:
            for doc_id, quotes in results:
                rec = {
                    "doc_id": doc_id,
                    "quotes": [
                        {
                            "text": q.quote,
                            "speaker": q.speaker,
                            "score": q.score,
                            "urls": q.urls,
                        }
                        for q in quotes
                    ],
                }
                out.write(json.dumps(rec) + "\n")
