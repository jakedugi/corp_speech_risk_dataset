from features_case_agnostic import *
from utils_case_agnostic import *
import time


def process_data(
    input_file,
    output_file,
    features_to_extract="all",
    corpus_quotes=None,
    corpus_contexts=None,
):
    data = load_data(input_file)
    if not corpus_quotes:
        corpus_quotes = [item["text"] for item in data]
        corpus_contexts = [item["context"] for item in data]

    timings = {
        "sentiment": [],
        "deontic_lexicons": [],
        "pos_lexicons": [],
        "ner": [],
        "dependency_linguistics": [],
        "wl_kernel": [],
        "local_keywords": [],
        "speaker_embedding": [],
        "section_headers": [],
        "quote_size": [],
        "omission_commission": [],
        "guilt_detection": [],
        "lying_detection": [],
        "evidential_lexicons": [],
        "causal_lexicons": [],
        "conditional_lexicons": [],
        "temporal_lexicons": [],
        "certainty_lexicons": [],
        "discourse_markers": [],
        "liability_lexicons": [],
    }
    token_cache = {}  # Cache for tokenized lengths

    for item in data:
        features = {}
        text = item["text"]
        context = item["context"]
        speaker = item["speaker"]

        # Group 1
        if features_to_extract == "all" or "sentiment" in features_to_extract:
            start = time.time()
            features.update(extract_sentiment(text, context))
            timings["sentiment"].append(time.time() - start)

        if features_to_extract == "all" or "deontic_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_deontic_lexicons(text, context, token_cache))
            timings["deontic_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "pos_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_pos_lexicons(text, context))
            timings["pos_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "ner" in features_to_extract:
            start = time.time()
            features.update(extract_ner(text, context, token_cache))
            timings["ner"].append(time.time() - start)

        if (
            features_to_extract == "all"
            or "dependency_linguistics" in features_to_extract
        ):
            start = time.time()
            features.update(extract_dependency_linguistics(text, context))
            timings["dependency_linguistics"].append(time.time() - start)

        if features_to_extract == "all" or "wl_kernel" in features_to_extract:
            start = time.time()
            features.update(extract_wl_kernel(text, context))
            timings["wl_kernel"].append(time.time() - start)

        # Group 2
        if features_to_extract == "all" or "local_keywords" in features_to_extract:
            start = time.time()
            features.update(extract_local_keywords_pair(text, context, k=5))
            timings["local_keywords"].append(time.time() - start)

        if features_to_extract == "all" or "speaker_embedding" in features_to_extract:
            start = time.time()
            features.update(extract_speaker_embedding(speaker))
            timings["speaker_embedding"].append(time.time() - start)

        if features_to_extract == "all" or "section_headers" in features_to_extract:
            start = time.time()
            features.update(extract_section_headers(context))
            timings["section_headers"].append(time.time() - start)

        # Group 3
        if features_to_extract == "all" or "quote_size" in features_to_extract:
            start = time.time()
            result, token_cache = extract_quote_size(text, context, token_cache)
            features.update(result)
            timings["quote_size"].append(time.time() - start)

        if features_to_extract == "all" or "omission_commission" in features_to_extract:
            start = time.time()
            features.update(extract_omission_commission(text, context))
            timings["omission_commission"].append(time.time() - start)

        if features_to_extract == "all" or "guilt_detection" in features_to_extract:
            start = time.time()
            features.update(extract_guilt_detection(text, context, token_cache))
            timings["guilt_detection"].append(time.time() - start)

        if features_to_extract == "all" or "lying_detection" in features_to_extract:
            start = time.time()
            features.update(extract_lying_detection(text, context, token_cache))
            timings["lying_detection"].append(time.time() - start)

        if features_to_extract == "all" or "evidential_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_evidential_lexicons(text, context, token_cache))
            timings["evidential_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "causal_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_causal_lexicons(text, context, token_cache))
            timings["causal_lexicons"].append(time.time() - start)

        if (
            features_to_extract == "all"
            or "conditional_lexicons" in features_to_extract
        ):
            start = time.time()
            features.update(extract_conditional_lexicons(text, context, token_cache))
            timings["conditional_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "temporal_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_temporal_lexicons(text, context, token_cache))
            timings["temporal_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "certainty_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_certainty_lexicons(text, context, token_cache))
            timings["certainty_lexicons"].append(time.time() - start)

        if features_to_extract == "all" or "discourse_markers" in features_to_extract:
            start = time.time()
            features.update(extract_discourse_markers(text, context, token_cache))
            timings["discourse_markers"].append(time.time() - start)

        if features_to_extract == "all" or "liability_lexicons" in features_to_extract:
            start = time.time()
            features.update(extract_liability_lexicons(text, context, token_cache))
            timings["liability_lexicons"].append(time.time() - start)

        item["raw_features"] = features

    save_data(data, output_file)
    return timings
