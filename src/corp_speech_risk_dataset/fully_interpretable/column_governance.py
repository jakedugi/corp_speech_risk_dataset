"""Column governance for POLAR interpretable models (minimal, leak-proof).

This module enforces a strict feature policy:
  • Allow only simple, auditable scalars and pre-derived `interpretable_*` features.
  • Block raw text, embeddings, sentiments, keywords, POS/NER, speaker_raw, section headers,
    predictions/targets, and any CV metadata.
  • Provide a tiny, explicit numeric whitelist for stable counts/sizes.
  • Optionally derive *only* simple rates/flags from raw count features (no opaque transforms).
"""

from typing import List, Dict, Any
import re
import numpy as np
from loguru import logger

# ---------------- Strict allow/block policy ----------------
# Minimal numeric whitelist (stable, auditable scalars)
NUMERIC_WHITELIST = {
    "quote_size",
    "context_size",
    "quote_omission",
    "quote_commission",
    "context_omission",
    "context_commission",
    "quote_guilt",
    "context_guilt",
    "quote_lying",
    "context_lying",
    "quote_evidential_count",
    "context_evidential_count",
    "quote_causal_count",
    "context_causal_count",
    "quote_conditional_count",
    "context_conditional_count",
    "quote_temporal_count",
    "context_temporal_count",
    "quote_certainty_count",
    "context_certainty_count",
    "quote_discourse_count",
    "context_discourse_count",
    "quote_liability_count",
    "context_liability_count",
    "quote_deontic_count",
    "context_deontic_count",
}

# Metadata always permitted (not used as features directly)
META_KEYS = {
    "case_id",
    "quote_id",
    "doc_id",
    "_src",
    "timestamp",
    "case_time",
    "final_judgement_real",
}

# Hard blocklist: anything here must NEVER reach the model
BLOCKLIST_PATTERNS = [
    # Embeddings & opaque vectors
    r".*emb.*",
    r"st_.*",
    r"fused_.*",
    r"gph_.*",
    # Token/ID level or graph structures
    r"sp_ids",
    r"wl_indices",
    r"wl_counts",
    r"(^|_)deps($|_)",
    r"^raw_deps$",
    # Raw text or high-cardinality/leaky text-ish inputs
    r"^text$",
    r"^context$",
    r"^section_headers$",
    r"^speaker_raw$",
    r"^speaker$",
    # Raw arrays we explicitly do NOT pass
    r"^quote_sentiment$",
    r"^context_sentiment$",
    r"^quote_top_keywords$",
    r"^context_top_keywords$",
    r"^quote_pos$",
    r"^context_pos$",
    r"^quote_ner$",
    r"^context_ner$",
    r"^quote_wl$",
    r"^context_wl$",
    r"^text_len$",
    r"^context_len$",
    # Labels / outcomes / predictions / CV metadata
    r"bucket",
    r"coral_.*",
    r"polar_.*",
    r"final_.*",
    r"outcome_.*",
    r"y_.*",
    r"label.*",
    r"target.*",
    r".*_pred.*",
    r".*_prob.*",
    r".*score.*",
    r".*confidence.*",
    r"^fold($|_)",
    r"^split($|_)",
    # CRITICAL: Court/venue features and derived proxies (prevents court leakage)
    r".*court.*",
    r".*venue.*",
    r".*district.*",
    r".*jurisdiction.*",
    r".*circuit.*",
    r".*division.*",
    r".*state.*",
    r"court_code_length",
    r"venue_length",
    r"district_length",
    # Path-based signals that could encode venue
    r".*src_path.*",
    r".*file_path.*",
    r".*path_length.*",
    # Speaker features (already blocked but being explicit)
    r".*speaker.*",
    r"speaker_length",
    r"has_speaker",
    # IDs and hashes
    r".*_id$",
    r".*_hash$",
    r"case_id.*",
    r"doc_id.*",
    # Timestamps that could leak
    r".*timestamp.*",
    r".*filing.*",
    r".*date.*",
    # BLOCK SPECIFIC INTERPRETABLE FEATURES (as requested by user)
    r"^interpretable_context_length$",
    r"^interpretable_feature_count$",
    r"^interpretable_ling_number_count$",
    r"^interpretable_ling_time_ref_count$",
    r"^interpretable_struct_avg_sent_len$",
    r"^interpretable_struct_caps_ratio$",
    r"^interpretable_struct_char_count$",
    r"^interpretable_struct_exclamations$",
    r"^interpretable_struct_parentheses$",
    r"^interpretable_struct_punct_density$",
    r"^interpretable_struct_questions$",
    r"^interpretable_struct_sentence_count$",
    r"^interpretable_struct_std_sent_len$",
    r"^interpretable_struct_word_count$",
    r"^interpretable_text_length$",
    # DROPPED FEATURES FROM ANALYSIS (automatically generated)
    r"^interpretable\_lex\_deception\_quote\_count$",
    r"^interpretable\_lex\_deception\_quote\_norm$",
    r"^interpretable\_lex\_guarantee\_quote\_count$",
    r"^interpretable\_lex\_guarantee\_quote\_norm$",
    r"^interpretable\_lex\_pricing\_claims\_quote\_count$",
    r"^interpretable\_lex\_pricing\_claims\_quote\_norm$",
    r"^interpretable\_lex\_scienter\_count$",
    r"^interpretable\_lex\_scienter\_norm$",
    r"^interpretable\_lex\_scienter\_present$",
    r"^interpretable\_lex\_scienter\_quote\_count$",
    r"^interpretable\_lex\_scienter\_quote\_norm$",
    r"^interpretable\_lex\_disclaimers\_count$",
    r"^interpretable\_lex\_disclaimers\_norm$",
    r"^interpretable\_lex\_disclaimers\_present$",
    r"^interpretable\_lex\_disclaimers\_quote\_count$",
    r"^interpretable\_lex\_disclaimers\_quote\_norm$",
    r"^interpretable\_seq\_trans\_neutral\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_superlatives\_to\_neutral$",
    r"^interpretable\_seq\_risk\_mean\_pos$",
    r"^interpretable\_seq\_risk\_std\_pos$",
    r"^interpretable\_seq\_trans\_neutral\_to\_hedges$",
    r"^interpretable\_seq\_trans\_hedges\_to\_neutral$",
    r"^interpretable\_seq\_trans\_neutral\_to\_reliability$",
    r"^interpretable\_seq\_trans\_neutral\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_guarantee\_to\_neutral$",
    r"^interpretable\_seq\_trans\_hedges\_to\_hedges$",
    r"^interpretable\_seq\_trans\_neutral\_to\_scienter$",
    r"^interpretable\_seq\_trans\_scienter\_to\_neutral$",
    r"^interpretable\_seq\_trans\_neutral\_to\_pricing\_claims$",
    r"^interpretable\_seq\_trans\_pricing\_claims\_to\_neutral$",
    r"^interpretable\_seq\_trans\_guarantee\_to\_hedges$",
    r"^interpretable\_seq\_trans\_neutral\_to\_deception$",
    r"^interpretable\_seq\_trans\_deception\_to\_neutral$",
    r"^interpretable\_seq\_trans\_neutral\_to\_disclaimers$",
    r"^interpretable\_seq\_trans\_disclaimers\_to\_neutral$",
    r"^interpretable\_seq\_trans\_superlatives\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_guarantee\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_guarantee\_to\_pricing\_claims$",
    r"^interpretable\_seq\_trans\_superlatives\_to\_hedges$",
    r"^interpretable\_seq\_trans\_hedges\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_pricing\_claims\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_deception\_to\_deception$",
    r"^interpretable\_seq\_trans\_deception\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_pricing\_claims\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_guarantee\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_scienter\_to\_hedges$",
    r"^interpretable\_seq\_trans\_superlatives\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_hedges\_to\_pricing\_claims$",
    r"^interpretable\_seq\_trans\_disclaimers\_to\_superlatives$",
    r"^interpretable\_seq\_trans\_superlatives\_to\_pricing\_claims$",
    r"^interpretable\_seq\_trans\_hedges\_to\_guarantee$",
    r"^interpretable\_seq\_trans\_deception\_to\_scienter$",
    r"^interpretable\_seq\_trans\_deception\_to\_disclaimers$",
    r"^interpretable\_seq\_trans\_disclaimers\_to\_hedges$",
    r"^interpretable\_seq\_trans\_pricing\_claims\_to\_hedges$",
    # FINAL PRUNING (redundancy removal)
    r"^interpretable\_lex\_deception\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_guarantee\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_hedges\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_hedges\_quote\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_hedges\_quote\_norm$",  # redundant_normalization
    r"^interpretable\_lex\_pricing\_claims\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_superlatives\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_superlatives\_quote\_count$",  # redundant_raw_count
    r"^interpretable\_lex\_superlatives\_quote\_norm$",  # redundant_normalization
    r"^interpretable\_ling\_low\_certainty$",  # redundant_in_concept
    r"^interpretable\_ling\_negation\_count$",  # redundant_raw_count
    r"^interpretable\_seq\_discourse\_causal$",  # redundant_in_concept
    r"^interpretable\_seq\_discourse\_conditional$",  # redundant_in_concept
    r"^interpretable\_seq\_discourse\_contrast$",  # redundant_in_concept
    r"^interpretable\_seq\_discourse\_temporal$",  # redundant_in_concept
    r"^interpretable\_lex\_pricing\_claims\_norm$",  # weak_separation
    r"^interpretable\_lex\_superlatives\_norm$",  # weak_separation
    r"^interpretable\_ling\_negation\_norm$",  # weak_separation
    # ============================================================
    # AUTOMATIC BLACKLIST - Features that failed comprehensive validation
    # Generated by auto_blacklist_failed_features.py
    # SIZE BIAS - 11 features
    r"^interpretable\_seq_trans_ambiguity_to_hedges$",  # size_bias
    r"^interpretable\_seq_trans_compliance_to_superlatives$",  # size_bias
    r"^interpretable\_seq_trans_deception_to_scope_limiters$",  # size_bias
    r"^interpretable\_seq_trans_guarantee_to_deception$",  # size_bias
    r"^interpretable\_seq_trans_guarantee_to_pricing_claims$",  # size_bias
    r"^interpretable\_seq_trans_guarantee_to_superlatives$",  # size_bias
    r"^interpretable\_seq_trans_hedges_to_hedges$",  # size_bias
    r"^interpretable\_seq_trans_pricing_claims_to_superlatives$",  # size_bias
    r"^interpretable\_seq_trans_scope_limiters_to_ambiguity$",  # size_bias
    r"^interpretable\_seq_trans_superlatives_to_pricing_claims$",  # size_bias
    r"^interpretable\_seq_trans_superlatives_to_scope_limiters$",  # size_bias
    # LEAKAGE - 1 features
    r"^interpretable\_seq_trans_superlatives_to_pricing_claims$",  # leakage
    # EXTREMELY SPARSE - 29 features
    r"^interpretable\_lex_disclaimers_quote_count$",  # extremely_sparse
    r"^interpretable\_lex_disclaimers_quote_norm$",  # extremely_sparse
    r"^interpretable\_lex_positive_qualifiers_count$",  # extremely_sparse
    r"^interpretable\_lex_positive_qualifiers_norm$",  # extremely_sparse
    r"^interpretable\_lex_positive_qualifiers_present$",  # extremely_sparse
    r"^interpretable\_lex_positive_qualifiers_quote_count$",  # extremely_sparse
    r"^interpretable\_lex_positive_qualifiers_quote_norm$",  # extremely_sparse
    r"^interpretable\_lex_reliability_quote_count$",  # extremely_sparse
    r"^interpretable\_lex_reliability_quote_norm$",  # extremely_sparse
    r"^interpretable\_lex_scienter_quote_count$",  # extremely_sparse
    r"^interpretable\_lex_scienter_quote_norm$",  # extremely_sparse
    r"^interpretable\_ling_money_count$",  # extremely_sparse
    r"^interpretable\_lr_disclaimer_endpos_present$",  # extremely_sparse
    r"^interpretable\_lr_documented_fact_sentence_count$",  # extremely_sparse
    r"^interpretable\_lr_documented_fact_sentence_share$",  # extremely_sparse
    r"^interpretable\_lr_evidential_count$",  # extremely_sparse
    r"^interpretable\_lr_hedge_near_guarantee_count$",  # extremely_sparse
    r"^interpretable\_lr_hedge_near_guarantee_norm$",  # extremely_sparse
    r"^interpretable\_lr_safe_harbor_count$",  # extremely_sparse
    r"^interpretable\_lr_safe_harbor_norm$",  # extremely_sparse
    r"^interpretable\_lr_safe_harbor_present$",  # extremely_sparse
    r"^interpretable\_lr_scope_near_guarantee_count$",  # extremely_sparse
    r"^interpretable\_lr_scope_near_guarantee_norm$",  # extremely_sparse
    r"^interpretable\_lr_sec_ref_count$",  # extremely_sparse
    r"^interpretable\_lr_url_count$",  # extremely_sparse
    r"^interpretable\_struct_exclamations$",  # extremely_sparse
    r"^interpretable\_struct_hashtag_count$",  # extremely_sparse
    r"^interpretable\_struct_link_presence$",  # extremely_sparse
    r"^interpretable\_struct_mention_count$",  # extremely_sparse
    # WEAK DISCRIMINATION - 63 features
    r"^interpretable\_lex_hedges_quote_count$",  # weak_discrimination
    r"^interpretable\_lex_hedges_quote_norm$",  # weak_discrimination
    r"^interpretable\_lex_pricing_claims_quote_count$",  # weak_discrimination
    r"^interpretable\_lex_pricing_claims_quote_norm$",  # weak_discrimination
    r"^interpretable\_lex_superlatives_quote_count$",  # weak_discrimination
    r"^interpretable\_lex_superlatives_quote_norm$",  # weak_discrimination
    r"^interpretable\_ling_conditional_count$",  # weak_discrimination
    r"^interpretable\_ling_negation_norm$",  # weak_discrimination
    r"^interpretable\_ling_passive_voice_count$",  # weak_discrimination
    r"^interpretable\_ling_passive_voice_norm$",  # weak_discrimination
    r"^interpretable\_lr_disclaimer_cluster$",  # weak_discrimination
    r"^interpretable\_lr_evidential_count$",  # weak_discrimination
    r"^interpretable\_lr_evidential_norm$",  # weak_discrimination
    r"^interpretable\_lr_evidential_present$",  # weak_discrimination
    r"^interpretable\_lr_evidential_strength$",  # weak_discrimination
    r"^interpretable\_lr_evidential_strength_norm$",  # weak_discrimination
    r"^interpretable\_lr_hedge_near_guarantee_count$",  # weak_discrimination
    r"^interpretable\_lr_hedge_near_guarantee_norm$",  # weak_discrimination
    r"^interpretable\_lr_negated_guarantee_count$",  # weak_discrimination
    r"^interpretable\_lr_negated_guarantee_norm$",  # weak_discrimination
    r"^interpretable\_lr_quote_presence$",  # weak_discrimination
    r"^interpretable\_lr_safe_harbor_count$",  # weak_discrimination
    r"^interpretable\_lr_safe_harbor_norm$",  # weak_discrimination
    r"^interpretable\_lr_safe_harbor_present$",  # weak_discrimination
    r"^interpretable\_lr_sec_ref_count$",  # weak_discrimination
    r"^interpretable\_lr_url_count$",  # weak_discrimination
    r"^interpretable\_seq_risk_mean_pos$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_deception$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_disclaimers$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_hedges$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_pricing_claims$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_scope_limiters$",  # weak_discrimination
    r"^interpretable\_seq_trans_ambiguity_to_superlatives$",  # weak_discrimination
    r"^interpretable\_seq_trans_compliance_to_ambiguity$",  # weak_discrimination
    r"^interpretable\_seq_trans_compliance_to_compliance$",  # weak_discrimination
    r"^interpretable\_seq_trans_compliance_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_compliance_to_reliability$",  # weak_discrimination
    r"^interpretable\_seq_trans_compliance_to_scope_limiters$",  # weak_discrimination
    r"^interpretable\_seq_trans_deception_to_disclaimers$",  # weak_discrimination
    r"^interpretable\_seq_trans_deception_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_disclaimers_to_hedges$",  # weak_discrimination
    r"^interpretable\_seq_trans_disclaimers_to_superlatives$",  # weak_discrimination
    r"^interpretable\_seq_trans_guarantee_to_deception$",  # weak_discrimination
    r"^interpretable\_seq_trans_guarantee_to_scope_limiters$",  # weak_discrimination
    r"^interpretable\_seq_trans_hedges_to_ambiguity$",  # weak_discrimination
    r"^interpretable\_seq_trans_hedges_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_hedges_to_superlatives$",  # weak_discrimination
    r"^interpretable\_seq_trans_pricing_claims_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_pricing_claims_to_hedges$",  # weak_discrimination
    r"^interpretable\_seq_trans_reliability_to_compliance$",  # weak_discrimination
    r"^interpretable\_seq_trans_reliability_to_superlatives$",  # weak_discrimination
    r"^interpretable\_seq_trans_scienter_to_hedges$",  # weak_discrimination
    r"^interpretable\_seq_trans_scope_limiters_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_scope_limiters_to_pricing_claims$",  # weak_discrimination
    r"^interpretable\_seq_trans_scope_limiters_to_scope_limiters$",  # weak_discrimination
    r"^interpretable\_seq_trans_superlatives_to_compliance$",  # weak_discrimination
    r"^interpretable\_seq_trans_superlatives_to_guarantee$",  # weak_discrimination
    r"^interpretable\_seq_trans_superlatives_to_pricing_claims$",  # weak_discrimination
    r"^interpretable\_seq_trans_superlatives_to_superlatives$",  # weak_discrimination
    r"^interpretable\_struct_hashtag_count$",  # weak_discrimination
    r"^interpretable\_struct_link_presence$",  # weak_discrimination
    r"^interpretable\_struct_mention_count$",  # weak_discrimination
    r"^interpretable\_struct_word_count$",  # weak_discrimination
    # SEQUENCE TRANSITIONS - 87 features
    r"^interpretable\_seq_trans_ambiguity_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_deception$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_disclaimers$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_reliability$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_ambiguity_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_reliability$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_compliance_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_deception$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_disclaimers$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_scienter$",  # sequence_transitions
    r"^interpretable\_seq_trans_deception_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_disclaimers_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_disclaimers_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_disclaimers_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_deception$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_guarantee_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_hedges_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_deception$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_disclaimers$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_positive_qualifiers$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_reliability$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_scienter$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_neutral_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_positive_qualifiers_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_pricing_claims_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_pricing_claims_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_pricing_claims_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_pricing_claims_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_pricing_claims_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_reliability_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_reliability_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_reliability_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_reliability_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_scienter_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_scienter_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_scope_limiters_to_superlatives$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_ambiguity$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_compliance$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_guarantee$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_hedges$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_neutral$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_pricing_claims$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_reliability$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_scope_limiters$",  # sequence_transitions
    r"^interpretable\_seq_trans_superlatives_to_superlatives$",  # sequence_transitions
]

# ---------------- Utilities ----------------


def _is_blocked(col: str) -> bool:
    # Check ALL patterns including specific interpretable features to block
    for p in BLOCKLIST_PATTERNS:
        if re.search(p, col):
            return True
    return False


def validate_columns(
    df_columns: List[str], allow_extra: bool = False
) -> Dict[str, Any]:
    """Validate/whitelist columns.

    Allowed feature columns are ONLY:
      • any column starting with `interpretable_` (assumed pre-derived and auditable),
      • any column starting with `int_` (derived interpretable features), and
      • members of NUMERIC_WHITELIST.
    Everything else is ignored or blocked per BLOCKLIST_PATTERNS.
    """
    # Hard blocks
    blocked = [c for c in df_columns if (c not in META_KEYS) and _is_blocked(c)]
    if blocked:
        logger.error(f"Blocked features detected: {blocked}")
        raise ValueError(f"Blocked features: {blocked}")

    # Allowed
    interpretable = [c for c in df_columns if c.startswith("interpretable_")]
    derived_interpretable = [
        c for c in df_columns if c.startswith("int_")
    ]  # New derived features
    numerics = [c for c in df_columns if c in NUMERIC_WHITELIST]
    allowed = numerics + interpretable + derived_interpretable

    # Extras (not blocked, not explicitly allowed)
    extras = [
        c
        for c in df_columns
        if (c not in allowed) and (c not in META_KEYS) and not _is_blocked(c)
    ]
    if extras and not allow_extra:
        logger.warning(f"Ignoring extra non-interpretable columns: {extras}")

    logger.info(
        f"Interpretable kept: {len(interpretable)} | Derived interpretable kept: {len(derived_interpretable)} | Numeric kept: {len(numerics)} | Total: {len(allowed)}"
    )
    return {
        "valid": True,
        "interpretable_features": allowed,
        "blocked_found": blocked,
        "extra_columns": extras,
        "feature_count": len(allowed),
    }


# ---------------- Minimal derivation (optional) ----------------
# We deliberately avoid complex transforms. If you still pass a raw record with
# raw count features, this helper exposes *simple* rates/flags only.


def _rate(count: float, denom: float) -> float:
    if denom <= 0:
        return float(count)
    return float(count) / float(denom)


def derive_interpretable(record: Dict[str, Any]) -> Dict[str, float]:
    """Derive **only** simple, auditable features.

    From record["raw_features"], we pull count fields and produce:
      • interpretable_{scope}_{field}_rate  (rate per tokens proxy)
      • interpretable_{scope}_{field}_any   (binary presence flag)

    No sentiments, keywords, POS/NER, or free-text features are produced here.
    """
    out: Dict[str, float] = {}
    raw = record.get("raw_features", {}) or {}

    # Token proxies (optional denominators for rates)
    quote_tokens = float(record.get("quote_size", 0) or 0) * 1000.0
    context_tokens = float(record.get("context_size", 0) or 0) * 1000.0
    token_guess = {"quote": max(1.0, quote_tokens), "context": max(1.0, context_tokens)}

    count_fields = [
        "deontic_count",
        "guilt",
        "lying",
        "evidential_count",
        "causal_count",
        "conditional_count",
        "temporal_count",
        "certainty_count",
        "discourse_count",
        "liability_count",
        # sizes themselves come from NUMERIC_WHITELIST; we don't re-derive them here
    ]

    for scope in ["quote", "context"]:
        for f in count_fields:
            k = f"{scope}_{f}"
            if k in raw:
                val = float(raw[k])
                out[f"interpretable_{k}_rate"] = _rate(val, token_guess[scope])
                out[f"interpretable_{k}_any"] = 1.0 if val > 0 else 0.0

    return out


# ---------------- Safe JSON helper ----------------


def pyify(o: Any) -> Any:
    """Convert numpy / non-JSON types to plain Python for json.dump."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)
            out[k] = pyify(v)
        return out
    if isinstance(o, (list, tuple, set)):
        return [pyify(x) for x in o]

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o
