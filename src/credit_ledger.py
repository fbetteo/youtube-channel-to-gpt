"""
Credit layers for subscriptions: what expires when, and what is spent first.

`user_credits.credits` is the single spendable balance; reserve/deduct/refund
code only adds to and subtracts from it. The balance is split into layers:

    lots       `credit_lots` rows, each with an `expires_at`:
                 monthly  the plan allowance, expiring at the period end; replaced
                          at renewal and removed when the plan ends (never on a
                          timer, so a late renewal payment does not wipe it)
                 topup    30 days after purchase, or when the plan ends,
                          whichever comes first
    permanent  `user_credits.permanent_credits`: welcome credits, legacy packs and
               the balance held when first subscribing; never expires
    unassigned the rest: a non-subscriber's balance, or refunds of credits whose
               lot is gone

Spending order is unassigned first, then lots by soonest expiry, then
permanent. Stored lot sizes are upper bounds: spending only lowers `credits`, so
the real size of each layer is derived from the total by `split`, filling
permanent first and then lots from the latest expiry. Every write locks the
user's row, normalizes the stored sizes to the real ones, edits the layers, and
stores the new total.
"""

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

TOPUP_LIFETIME = timedelta(days=30)
# past_due keeps credits while Stripe retries the card.
ACTIVE_SUBSCRIPTION_STATUSES = ("active", "trialing", "past_due")
MONTHLY, TOPUP = "monthly", "topup"


@dataclass(frozen=True)
class Lot:
    amount: int
    expires_at: datetime
    created_at: datetime
    kind: str = TOPUP
    source_id: Optional[str] = None


@dataclass(frozen=True)
class Layers:
    permanent: int
    lots: List[Lot] = field(default_factory=list)  # latest expiry first: spent last
    unassigned: int = 0

    @property
    def monthly(self) -> int:
        return sum(lot.amount for lot in self.lots if lot.kind == MONTHLY)

    @property
    def topup_lots(self) -> List[Lot]:
        return [lot for lot in self.lots if lot.kind == TOPUP]

    @property
    def topups(self) -> int:
        return sum(lot.amount for lot in self.topup_lots)

    @property
    def total(self) -> int:
        return self.permanent + sum(lot.amount for lot in self.lots) + self.unassigned


def split(total: int, permanent: int, lots: List[Lot]) -> Layers:
    """Real layer sizes for a balance, given stored upper bounds."""
    remaining = max(total, 0)
    kept_permanent = min(permanent, remaining)
    remaining -= kept_permanent
    kept_lots = []
    # Fill from the latest expiry, so the soonest-expiring lot is spent first.
    for lot in sorted(lots, key=lambda l: (l.expires_at, l.created_at), reverse=True):
        amount = min(lot.amount, remaining)
        remaining -= amount
        if amount:
            kept_lots.append(replace(lot, amount=amount))
    return Layers(kept_permanent, kept_lots, remaining)


def drop_expired_topups(layers: Layers, now: datetime) -> Layers:
    return replace(
        layers,
        lots=[l for l in layers.lots if l.kind != TOPUP or l.expires_at > now],
    )


async def _claim(tx, key: str, user_id: str, credits: int) -> bool:
    """Record a Stripe session/invoice once; False when it was already credited."""
    return (
        await tx.fetchval(
            """
            INSERT INTO stripe_checkout_credits (session_id, user_id, credits)
            VALUES ($1, $2, $3)
            ON CONFLICT (session_id) DO NOTHING
            RETURNING session_id
            """,
            key,
            user_id,
            credits,
        )
        is not None
    )


async def _load(conn, user_id: str, lock: bool) -> Optional[Layers]:
    row = await conn.fetchrow(
        "SELECT credits, permanent_credits FROM user_credits WHERE user_id = $1"
        + (" FOR UPDATE" if lock else ""),
        user_id,
    )
    if row is None:
        return None
    lots = [
        Lot(r["amount"], r["expires_at"], r["created_at"], r["kind"], r["source_id"])
        for r in await conn.fetch(
            "SELECT amount, expires_at, created_at, kind, source_id FROM credit_lots WHERE user_id = $1",
            user_id,
        )
    ]
    return split(row["credits"], row["permanent_credits"], lots)


async def _save(tx, user_id: str, layers: Layers) -> None:
    await tx.execute(
        "UPDATE user_credits SET credits = $2, permanent_credits = $3 WHERE user_id = $1",
        user_id,
        layers.total,
        layers.permanent,
    )
    await tx.execute("DELETE FROM credit_lots WHERE user_id = $1", user_id)
    await tx.executemany(
        """
        INSERT INTO credit_lots (user_id, amount, expires_at, created_at, kind, source_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            (user_id, lot.amount, lot.expires_at, lot.created_at, lot.kind, lot.source_id)
            for lot in layers.lots
        ],
    )


async def _update(
    user_id: str,
    change: Callable[[Layers], Layers],
    claim: Optional[tuple] = None,
) -> bool:
    """Lock, normalize, apply `change`, save. False when `claim` was already used."""
    from db_youtube_transcripts.database import get_db_transaction

    async with get_db_transaction() as tx:
        if claim and not await _claim(tx, *claim):
            return False
        await tx.execute(
            "INSERT INTO user_credits (user_id, credits) VALUES ($1, 0) "
            "ON CONFLICT (user_id) DO NOTHING",
            user_id,
        )
        layers = await _load(tx, user_id, lock=True)
        await _save(tx, user_id, change(layers))
    return True


async def grant_topup(session_id: str, user_id: str, credits: int) -> bool:
    """Add a top-up lot once per Checkout session; it expires in 30 days."""
    now = datetime.now(timezone.utc)
    lot = Lot(credits, now + TOPUP_LIFETIME, now, TOPUP, session_id)

    def change(layers: Layers) -> Layers:
        layers = drop_expired_topups(layers, now)
        return replace(layers, lots=[*layers.lots, lot])

    return await _update(user_id, change, claim=(session_id, user_id, credits))


async def grant_permanent(session_id: str, user_id: str, credits: int) -> bool:
    """Legacy one-time packs: credits that never expire, once per Checkout session."""
    return await _update(
        user_id,
        lambda layers: replace(layers, permanent=layers.permanent + credits),
        claim=(session_id, user_id, credits),
    )


async def start_period(
    invoice_id: str,
    user_id: str,
    credits: int,
    first_period: bool,
    period_end: datetime,
) -> bool:
    """
    Start a paid plan period once per invoice: replace the monthly lot with this
    period's allowance, expiring at `period_end`. Unassigned credits are dropped
    like the old allowance, except on the first period, when they are what the
    user held before subscribing and become permanent.
    """
    now = datetime.now(timezone.utc)

    def change(layers: Layers) -> Layers:
        layers = drop_expired_topups(layers, now)
        kept = [lot for lot in layers.lots if lot.kind != MONTHLY]
        monthly = Lot(credits, period_end, now, MONTHLY, invoice_id)
        permanent = layers.permanent + (layers.unassigned if first_period else 0)
        return Layers(permanent, [*kept, monthly])

    return await _update(user_id, change, claim=(invoice_id, user_id, credits))


async def end_subscription(user_id: str) -> None:
    """Remove monthly, top-up and unassigned credits unless another plan of the user is active."""
    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        active = await conn.fetchval(
            "SELECT 1 FROM user_subscriptions WHERE user_id = $1 AND status = ANY($2::text[])",
            user_id,
            list(ACTIVE_SUBSCRIPTION_STATUSES),
        )
    if not active:
        await _update(user_id, lambda layers: Layers(layers.permanent))


async def expire_topups(user_id: str) -> None:
    now = datetime.now(timezone.utc)
    await _update(user_id, lambda layers: drop_expired_topups(layers, now))


_DUE_TOPUPS = f"kind = '{TOPUP}' AND expires_at <= NOW()"


async def summary(user_id: str) -> Layers:
    """Current layers, expiring due top-ups first."""
    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        due = await conn.fetchval(
            f"SELECT 1 FROM credit_lots WHERE user_id = $1 AND {_DUE_TOPUPS} LIMIT 1",
            user_id,
        )
    if due:
        await expire_topups(user_id)
    async with get_db_connection() as conn:
        return await _load(conn, user_id, lock=False) or Layers(0)


async def run_topup_expiry_sweeper(interval_seconds: int = 300) -> None:
    """Expire top-ups past their date. Spending in between can use them for up to one interval."""
    from db_youtube_transcripts.database import get_db_connection

    while True:
        try:
            async with get_db_connection() as conn:
                users = await conn.fetch(
                    f"SELECT DISTINCT user_id FROM credit_lots WHERE {_DUE_TOPUPS}"
                )
            for row in users:
                await expire_topups(row["user_id"])
            if users:
                logger.info(f"Expired top-ups for {len(users)} users")
        except Exception as e:
            logger.error(f"Top-up expiry sweep failed: {e}")
        await asyncio.sleep(interval_seconds)
