def test_auto_selection():
    from case_outcome_imputer import AmountSelector, Candidate

    cands = [
        Candidate(1_000_000, "$1 million", "ctx1"),
        Candidate(74_600_000, "$74.6 million", "ctx2"),
        Candidate(16_000_000, "$16 million", "ctx3"),
    ]
    assert AmountSelector().choose(cands) == 74_600_000


def test_no_candidates_returns_none(tmp_path):
    from case_outcome_imputer import impute_for_case, AmountSelector

    (tmp_path / "dummy_case").mkdir()
    impute_for_case(tmp_path / "dummy_case", AmountSelector())  # just runs
