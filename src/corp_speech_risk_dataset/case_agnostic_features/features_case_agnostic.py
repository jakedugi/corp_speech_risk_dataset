import re
import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Device selection: CUDA > MPS > CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

SPACY_CONFIGS = {
    "pos": {"disable": ["ner", "parser"]},  # POS needs tagger only
    "ner": {"disable": ["tagger", "parser"]},  # NER standalone
    "deps": {"disable": ["ner"]},  # Deps need parser + tagger
    "wl": {"disable": ["ner"]},  # WL uses deps, so same as above
}

# Load models
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])  # Disable unused components
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(device)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Precompile regex patterns
evidential_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "alleged",
        "according to",
        "reported",
        "testified",
        "declared",
        "purportedly",
        "asserted",
        "claimed",
        "said",
        "stated",
    ]
]
causal_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "because",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "due to",
        "as a result",
        "since",
        "so that",
        "thereby",
    ]
]
conditional_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "if",
        "unless",
        "provided that",
        "in the event",
        "should",
        "assuming",
        "contingent upon",
        "insofar as",
    ]
]
temporal_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "before",
        "after",
        "during",
        "subsequently",
        "prior to",
        "hitherto",
        "thereafter",
        "at once",
        "immediately",
        "ultimately",
    ]
]
certainty_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "certainly",
        "definitely",
        "undoubtedly",
        "possibly",
        "probably",
        "clearly",
        "evidently",
        "apparently",
        "presumably",
        "arguably",
        "likely",
        "assuredly",
        "conceivably",
        "manifestly",  # Legal certainty terms
    ]
]
discourse_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "whereas",
        "however",
        "moreover",
        "furthermore",
        "nevertheless",
        "on the other hand",
        "accordingly",
        "in contrast",
        "notwithstanding",
        "in addition",
        "conversely",
        "henceforth",
        "thereupon",  # Legal discourse markers
    ]
]
liability_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "damage",
        "injury",
        "loss",
        "compensation",
        "liability",
        "negligence",
        "fault",
        "remedy",
        "indemnify",
        "penalty",
    ]
]
deontic_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "must",
        "shall",
        "may",
        "ought",
        "should",
        "required",
        "prohibited",
        "permitted",
        "obligated",
        "forbidden",
        "mandated",
        "compelled",
        "authorized",
        "banned",
        "duty",
    ]
]
omission_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in ["failed to", "neglected", "omitted", "did not", "refrained from"]
]
commission_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in ["did", "performed", "committed", "executed", "caused"]
]
guilt_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "regret",
        "sorry",
        "ashamed",
        "blame",
        "fault",
        "responsible",
        "remorse",
        "guilty",
        "culpable",
        "accountable",
        "apologize",
        "admit",
        "confess",
        "admit",
        "confession",
        "shame",
        "guilt",
    ]
]
lying_patterns = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for term in [
        "perhaps",
        "maybe",
        "sort of",
        "not",
        "no",
        "never",
        "I think",
        "possibly",
    ]
]
HEADER_PATTERNS = re.compile(
    r"(?:^[A-Z\s]{5,}$|"  # ALL CAPS headers (lowered threshold from 10 to capture shorter like "SYLLABUS")
    r"^[\d\s]*[A-Z][A-Z\s]+$|"  # Mostly caps with numbers
    r"^\s*(?:SECTION|PART|CHAPTER|ARTICLE)\s+[IVXLCDM\d]+(?:\([A-Z]\))?|"  # Section headers, added optional (X) for subsections
    r"^\s*[A-Z]\.\s+[A-Z][A-Z\s]+|"  # Lettered section headers
    r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z\s]+|"  # Numbered section headers, added sub-levels like 1.1
    r"^\s*Held:|"  # Common in opinions
    r"^\s*Syllabus)",  # Specific common header
    re.MULTILINE | re.IGNORECASE,  # Added IGNORECASE for mixed-case docs
)

SECTION_BOUNDARIES = re.compile(
    r"(?:WHEREAS|NOW THEREFORE|IT IS HEREBY|ORDERED AND ADJUDGED|"
    r"FOR THE FOREGOING REASONS|IN CONCLUSION|ACCORDINGLY|"
    r"BACKGROUND|PROCEDURAL HISTORY|FACTUAL BACKGROUND|DISCUSSION|ANALYSIS|"
    r"CONCLUSION|RELIEF|DAMAGES|SETTLEMENT TERMS|FINAL JUDGMENT|"
    r"MONETARY JUDGMENT|INJUNCTIVE RELIEF|ATTORNEY FEES|COSTS)",
    re.MULTILINE | re.IGNORECASE,
)


def extract_evidential_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in evidential_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in evidential_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_evidential_count": quote_count / quote_len,
        "context_evidential_count": context_count / context_len,
    }


def extract_causal_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in causal_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(len(pattern.findall(text.lower())) for pattern in causal_patterns)
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in causal_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_causal_count": quote_count / quote_len,
        "context_causal_count": context_count / context_len,
    }


def extract_conditional_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in conditional_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in conditional_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in conditional_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_conditional_count": quote_count / quote_len,
        "context_conditional_count": context_count / context_len,
    }


def extract_temporal_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in temporal_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in temporal_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in temporal_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_temporal_count": quote_count / quote_len,
        "context_temporal_count": context_count / context_len,
    }


def extract_certainty_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in certainty_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in certainty_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in certainty_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_certainty_count": quote_count / quote_len,
        "context_certainty_count": context_count / context_len,
    }


def extract_discourse_markers(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in discourse_markers (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in discourse_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in discourse_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_discourse_count": quote_count / quote_len,
        "context_discourse_count": context_count / context_len,
    }


def extract_liability_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in liability_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_count = sum(
        len(pattern.findall(text.lower())) for pattern in liability_patterns
    )
    context_count = sum(
        len(pattern.findall(context.lower())) for pattern in liability_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_liability_count": quote_count / quote_len,
        "context_liability_count": context_count / context_len,
    }


def extract_quote_size(text, context, cache=None):
    if cache is None:
        cache = {}
    if text not in cache:
        cache[text] = len(tokenizer.tokenize(text, max_length=1024, truncation=True))
    if context not in cache:
        cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )
    quote_len = cache[text]
    context_len = cache[context]
    max_len = max(quote_len, context_len) or 1
    return {
        "quote_size": quote_len / max_len,
        "context_size": context_len / max_len,
    }, cache


def extract_sentiment(text, context):
    quote_sent = sentiment_analyzer.polarity_scores(text)
    context_sent = sentiment_analyzer.polarity_scores(context)
    return {
        "quote_sentiment": [quote_sent["pos"], quote_sent["neg"], quote_sent["neu"]],
        "context_sentiment": [
            context_sent["pos"],
            context_sent["neg"],
            context_sent["neu"],
        ],
    }


def extract_deontic_lexicons(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in deontic_lexicons (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_counts = sum(
        len(pattern.findall(text.lower())) for pattern in deontic_patterns
    )
    context_counts = sum(
        len(pattern.findall(context.lower())) for pattern in deontic_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_deontic_count": quote_counts / quote_len,
        "context_deontic_count": context_counts / context_len,
    }


def extract_pos_lexicons(text, context):
    docs = list(nlp.pipe([text, context], **SPACY_CONFIGS.get("pos", {})))
    quote_doc, context_doc = docs
    pos_tags = [
        "NOUN",
        "PROPN",
        "VERB",
        "AUX",
        "ADJ",
        "ADV",
        "DET",
        "PRON",
        "ADP",
        "CCONJ",
        "SCONJ",
    ]
    quote_pos = Counter([token.pos_ for token in quote_doc])
    context_pos = Counter([token.pos_ for token in context_doc])
    total_quote = len(quote_doc)
    total_context = len(context_doc)
    return {
        "quote_pos": [
            quote_pos[tag] / total_quote if total_quote else 0 for tag in pos_tags
        ],
        "context_pos": [
            context_pos[tag] / total_context if total_context else 0 for tag in pos_tags
        ],
    }


def extract_local_keywords(text, k=5):
    tokens = [
        w.lower()
        for w in re.findall(r"\w+", text)
        if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2
    ]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(k)]


def extract_local_keywords_pair(text, context, k=5):
    return {
        "quote_top_keywords": extract_local_keywords(text, k),
        "context_top_keywords": extract_local_keywords(context, k),
    }


def extract_ner(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in NER (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    docs = list(nlp.pipe([text, context], **SPACY_CONFIGS.get("ner", {})))
    quote_doc, context_doc = docs
    ner_types = ["PERSON", "ORG", "LAW", "GPE", "EVENT", "MONEY", "QUANTITY"]
    quote_ner = Counter([ent.label_ for ent in quote_doc.ents])
    context_ner = Counter([ent.label_ for ent in context_doc.ents])

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_ner": [quote_ner.get(t, 0) / quote_len for t in ner_types],
        "context_ner": [context_ner.get(t, 0) / context_len for t in ner_types],
    }


def embed_text(text, model=legal_bert, tokenizer=tokenizer, device=device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()


def extract_speaker_embedding(speaker):
    return {"speaker_raw": speaker}


def extract_section_headers(context):
    headers = set(
        HEADER_PATTERNS.findall(context) + SECTION_BOUNDARIES.findall(context)
    )  # Union for broader capture, set to dedup
    headers = list(headers) if headers else ["UNKNOWN"]  # Fallback
    return {"section_headers": headers}


def token_depth(token):
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


def extract_dependency_linguistics(text, context):
    docs = list(nlp.pipe([text, context], disable=["ner", "lemmatizer"]))
    quote_doc, context_doc = docs
    deps = [
        "nsubj",
        "nsubjpass",
        "dobj",
        "iobj",
        "prep",
        "pobj",
        "aux",
        "auxpass",
        "neg",
        "advmod",
        "ccomp",
        "xcomp",
        "csubj",
        "csubjpass",
        "advcl",
        "amod",
        "compound",
        "appos",
        "conj",
        "cc",
        "mark",
        "relcl",
    ]
    quote_deps = Counter([token.dep_ for token in quote_doc])
    context_deps = Counter([token.dep_ for token in context_doc])
    quote_depth = max([token_depth(token) for token in quote_doc]) if quote_doc else 0
    context_depth = (
        max([token_depth(token) for token in context_doc]) if context_doc else 0
    )
    return {
        "quote_deps": [quote_deps.get(d, 0) for d in deps] + [quote_depth],
        "context_deps": [context_deps.get(d, 0) for d in deps] + [context_depth],
    }


def extract_wl_kernel(text, context):
    docs = list(nlp.pipe([text, context], disable=["ner", "lemmatizer"]))
    quote_doc, context_doc = docs
    G_quote = nx.Graph()
    for token in quote_doc:
        G_quote.add_node(token.i, label=token.dep_)
        if token.head != token:
            G_quote.add_edge(token.i, token.head.i)
    G_context = nx.Graph()
    for token in context_doc:
        G_context.add_node(token.i, label=token.dep_)
        if token.head != token:
            G_context.add_edge(token.i, token.head.i)

    def wl_labels(G, iterations=2):
        labels = {n: data["label"] for n, data in G.nodes(data=True)}
        for _ in range(iterations):
            new_labels = {}
            for n in G:
                neigh = sorted([labels[m] for m in G.neighbors(n)])
                new_labels[n] = hash((labels[n], tuple(neigh)))
            labels = new_labels
        return len(set(labels.values()))

    return {"quote_wl": wl_labels(G_quote), "context_wl": wl_labels(G_context)}


def extract_omission_commission(text, context):
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in omission_commission (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_om = sum(len(pattern.findall(text.lower())) for pattern in omission_patterns)
    quote_com = sum(
        len(pattern.findall(text.lower())) for pattern in commission_patterns
    )
    context_om = sum(
        len(pattern.findall(context.lower())) for pattern in omission_patterns
    )
    context_com = sum(
        len(pattern.findall(context.lower())) for pattern in commission_patterns
    )
    quote_len = len(tokenizer.tokenize(text)) or 1
    context_len = len(tokenizer.tokenize(context)) or 1
    return {
        "quote_omission": quote_om / quote_len,
        "quote_commission": quote_com / quote_len,
        "context_omission": context_om / context_len,
        "context_commission": context_com / context_len,
    }


def extract_guilt_detection(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in guilt_detection (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_guilt = sum(len(pattern.findall(text.lower())) for pattern in guilt_patterns)
    context_guilt = sum(
        len(pattern.findall(context.lower())) for pattern in guilt_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_guilt": quote_guilt / quote_len,
        "context_guilt": context_guilt / context_len,
    }


def extract_lying_detection(text, context, token_cache=None):
    if token_cache is None:
        token_cache = {}
    if not text or not context:
        print(
            f"Warning: Empty text or context detected in lying_detection (text: {len(text) if text else 0}, context: {len(context) if context else 0})"
        )
    quote_lying = sum(len(pattern.findall(text.lower())) for pattern in lying_patterns)
    context_lying = sum(
        len(pattern.findall(context.lower())) for pattern in lying_patterns
    )

    # Use cached token lengths for consistent normalization
    if text not in token_cache:
        token_cache[text] = len(
            tokenizer.tokenize(text, max_length=1024, truncation=True)
        )
    if context not in token_cache:
        token_cache[context] = len(
            tokenizer.tokenize(context, max_length=1024, truncation=True)
        )

    quote_len = token_cache[text] or 1
    context_len = token_cache[context] or 1

    return {
        "quote_lying": quote_lying / quote_len,
        "context_lying": context_lying / context_len,
    }


def fuse_features(existing_fused, new_features):
    flat_new = []
    for v in new_features.values():
        if isinstance(v, list):
            flat_new.extend(v)
        else:
            flat_new.append(v)
    return np.concatenate([existing_fused, flat_new]).tolist()


# def extract_tfidf(corpus_quotes, corpus_contexts, text, context, pca_dim=50):
#     vectorizer = TfidfVectorizer(max_features=100)
#     vectorizer.fit(corpus_quotes + corpus_contexts)
#     quote_tfidf = vectorizer.transform([text]).toarray()[0]
#     context_tfidf = vectorizer.transform([context]).toarray()[0]
#     if pca_dim:
#         pca = PCA(n_components=pca_dim)
#         quote_tfidf = pca.fit_transform([quote_tfidf])[0]
#         context_tfidf = pca.transform([context_tfidf])[0]
#     return {
#         "quote_tfidf": quote_tfidf.tolist(),
#         "context_tfidf": context_tfidf.tolist()
#     }
#     # Dim: 100-200, reduce with PCA. Fusion: High, complements BERT, concatenate, not directly in GraphSAGE.


# Decisions:
# - Modular: Separated extraction functions for extendability.
# - CLI: Supports individual (e.g., --features=quote_size,sentiment) or all.
# - Optimization: CPU for Spacy/VADER/regex (fast on M1), MPS/GPU for BERT embeddings.
# - TF-IDF: Fit on whole corpus for IDF, PCA optional (set to 50 for dim control).
# - WL: Approximated with label propagation count for speed (O(nodes), not O(n^2)).
# - Detections: Lexicon-based for speed/meaningful signal, no ML to avoid training.
# - Fusion: Simple concat; user can PCA later if dim high. Comments inline on benefits.
# - Section headers: Regex extraction, assume in context.
# - Input/Output: JSONL for large data.
# - All implemented without skipping; two-step: extract raw, then fuse.
# - Works one-shot: Install deps (torch, transformers, spacy, vaderSentiment, sklearn, networkx), download spacy model.
