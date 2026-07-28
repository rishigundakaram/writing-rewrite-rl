import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "environments" / "writing_rewrite"))

import writing_rewrite as wr


def test_aggregate_ranks_uniform_weights() -> None:
    # two dims over 3 candidates; dim1 order 0<1<2, dim2 reversed
    rank_lists = [[0, 1, 2], [2, 1, 0]]
    agg = wr.aggregate_ranks(rank_lists, [1.0, 1.0], 3)
    assert agg == [0.5, 0.5, 0.5]  # perfectly disagreeing judges cancel


def test_aggregate_ranks_weighted() -> None:
    rank_lists = [[0, 1, 2], [2, 1, 0]]
    agg = wr.aggregate_ranks(rank_lists, [3.0, 1.0], 3)
    assert agg[0] > agg[1] > agg[2]  # heavier dim dominates
    assert all(0.0 <= a <= 1.0 for a in agg)


def test_aggregate_ranks_single_candidate() -> None:
    assert wr.aggregate_ranks([[0]], [1.0], 1) == [1.0]


def test_emdash_penalty_source_conditional() -> None:
    assert wr.emdash_penalty("plain source", "text with — a dash") == 0.5
    assert wr.emdash_penalty("source — has one", "text with — a dash") == 1.0
    assert wr.emdash_penalty("plain source", "no dashes here") == 1.0
    assert wr.emdash_penalty("plain", "en dash – too") == 0.5


def test_rank_dimensions_has_variety_plus_pillars() -> None:
    dims = wr._rank_dimensions()
    assert "sentence_variety" in dims
    assert len(dims) == 6
    assert "anti_llm_slop" in dims


def test_dim_weights_uniform_and_override(monkeypatch) -> None:
    dims = list(wr._rank_dimensions())
    assert set(wr._dim_weights(dims).values()) == {1.0}
    monkeypatch.setenv("JUDGE_WEIGHTS", '{"anti_llm_slop": 2.5}')
    w = wr._dim_weights(dims)
    assert w["anti_llm_slop"] == 2.5 and w["sentence_variety"] == 1.0
