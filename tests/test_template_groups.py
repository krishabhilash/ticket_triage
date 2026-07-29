"""Tests for evaluation-only template grouping."""

import pytest

from ticket_triage.template_groups import normalize_template


def test_superficial_template_variants_share_group() -> None:
    first = "Hello team, I bought $4,500 of USDT but the price is wrong. Thanks."
    second = "URGENT: I bought $25 of ETH, but the price is wrong! Please advise."
    assert normalize_template(first) == normalize_template(second)


def test_different_intents_remain_separate() -> None:
    fraud = "Someone withdrew 2 BTC without my permission."
    dispute = "I was charged twice for the same 2 BTC purchase."
    assert normalize_template(fraud) != normalize_template(dispute)


@pytest.mark.parametrize("message", ["", "   "])
def test_template_normalization_rejects_empty_message(message: str) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        normalize_template(message)
