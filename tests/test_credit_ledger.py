import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_ledger import MONTHLY, TOPUP, Layers, Lot, drop_expired_topups, split

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def days(n):
    return NOW + timedelta(days=n)


def topup(amount, expires_in_days):
    return Lot(amount, days(expires_in_days), days(expires_in_days - 30), TOPUP)


def monthly(amount, period_ends_in_days):
    return Lot(amount, days(period_ends_in_days), days(period_ends_in_days - 30), MONTHLY)


def amounts(layers):
    """Remaining amount per lot, in spend order (soonest expiry first)."""
    return [(lot.kind, lot.amount) for lot in reversed(layers.lots)]


def test_soonest_expiring_credits_are_spent_first():
    # Period renews in 17 days; the top-up lasts 30. 200 of 1,100 credits spent.
    layers = split(900, 100, [topup(500, 30), monthly(500, 17)])

    assert amounts(layers) == [(MONTHLY, 300), (TOPUP, 500)]


def test_after_renewal_an_older_topup_is_spent_before_fresh_monthly_credits():
    # Top-up bought 16 days ago (14 left); fresh allowance lasts the full month.
    layers = split(1800, 100, [monthly(1000, 30), topup(1000, 14)])

    assert amounts(layers) == [(TOPUP, 700), (MONTHLY, 1000)]


def test_two_topups_spend_the_older_one_first():
    layers = split(1100, 0, [topup(1000, 30), topup(1000, 28)])

    assert amounts(layers) == [(TOPUP, 100), (TOPUP, 1000)]


def test_permanent_credits_are_spent_last():
    layers = split(40, 100, [monthly(1000, 10), topup(500, 20)])

    assert (layers.permanent, layers.lots, layers.unassigned) == (40, [], 0)


def test_refund_refills_the_lowest_layer_first():
    # Spent down to 40, then a 30-credit refund: permanent credits come back first.
    layers = split(70, 100, [topup(500, 20)])

    assert (layers.permanent, layers.topups) == (70, 0)


def test_balance_above_every_layer_is_unassigned():
    # Existing users have no stored layers until their first plan period.
    assert split(200, 0, []) == Layers(0, [], 200)


def test_expired_topups_are_dropped_but_monthly_credits_wait_for_stripe():
    layers = split(1600, 100, [monthly(500, -1), topup(400, 5), topup(600, -1)])
    kept = drop_expired_topups(layers, NOW)

    assert amounts(kept) == [(MONTHLY, 500), (TOPUP, 400)]
    assert kept.total == 1000
