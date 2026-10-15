"""
Create or update the Stripe objects for the subscription + top-up pricing.

Edit PLANS / TOPUPS below and re-run; the script makes Stripe match them:
- Products have fixed IDs, so they are created once and renamed in place.
- Prices are found by lookup_key (what the backend/frontend reference, never
  price_... IDs). Stripe prices are immutable: when amount or credits change,
  a new price takes over the lookup_key and the old one is archived. Existing
  subscribers stay on the old price until moved; new checkouts use the new one.
- Each price carries metadata.credits, which the webhook grants.
- A dedicated Customer Portal configuration (cancel, update card, invoices).
  Set STRIPE_PORTAL_CONFIGURATION_ID in .env to the id it prints.
- With --webhook-url, adds (never removes) the subscription events to that
  existing webhook endpoint. The endpoint is shared with other projects.

Dry run by default: prints what it would do. Pass --apply to write.
Uses STRIPE_SECRET_KEY_LIVE, or STRIPE_SECRET_KEY_TEST_TRANSCRIPTS with --test.

    poetry run python scripts/stripe_setup.py --test
    poetry run python scripts/stripe_setup.py --test --apply
    poetry run python scripts/stripe_setup.py --apply --webhook-url https://<api>/payments/webhook
"""

import argparse
import os

import stripe
from dotenv import load_dotenv

PROJECT = "transcript-api"
CURRENCY = "usd"

# Placeholder price points: change freely and re-run.
PLANS = [
    {
        "lookup_key": "transcript_monthly_1000",
        "amount_cents": 700,
        "credits": 1000,
    },
]
# Sold in any quantity: N units = N x credits.
TOPUPS = [
    {"lookup_key": "transcript_topup_1000", "amount_cents": 300, "credits": 1000},
]

PRODUCTS = {
    "plan": {"id": "transcript_api_subscription", "name": "Transcript API subscription"},
    "topup": {"id": "transcript_api_topup", "name": "Transcript API credit top-up"},
}

WEBHOOK_EVENTS = [
    "checkout.session.completed",
    "invoice.paid",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
]


def ensure_product(kind, apply):
    spec = PRODUCTS[kind]
    try:
        product = stripe.Product.retrieve(spec["id"])
    except stripe.error.InvalidRequestError:
        print(f"[product] create {spec['id']} ({spec['name']})")
        if apply:
            stripe.Product.create(
                id=spec["id"], name=spec["name"], metadata={"project": PROJECT}
            )
        return
    if product.name != spec["name"] or not product.active:
        print(f"[product] update {spec['id']} -> name={spec['name']!r}, active")
        if apply:
            stripe.Product.modify(spec["id"], name=spec["name"], active=True)
    else:
        print(f"[product] ok {spec['id']}")


def ensure_price(kind, spec, apply):
    """Make the price behind spec['lookup_key'] match spec."""
    lookup_key = spec["lookup_key"]
    recurring = {"interval": "month"} if kind == "plan" else None
    existing = stripe.Price.list(lookup_keys=[lookup_key], active=True).data
    current = existing[0] if existing else None

    if current is not None and (
        current.unit_amount == spec["amount_cents"]
        and current.currency == CURRENCY
        and current.product == PRODUCTS[kind]["id"]
        and (current.metadata or {}).get("credits") == str(spec["credits"])
        and (current.recurring.interval if current.recurring else None)
        == (recurring["interval"] if recurring else None)
    ):
        print(f"[price] ok {lookup_key} = {current.id}")
        return

    action = "replace" if current else "create"
    print(
        f"[price] {action} {lookup_key}: ${spec['amount_cents'] / 100:.2f}"
        f"{'/month' if recurring else ''}, {spec['credits']} credits"
        + (f" (archives {current.id})" if current else "")
    )
    if not apply:
        return

    params = {
        "product": PRODUCTS[kind]["id"],
        "currency": CURRENCY,
        "unit_amount": spec["amount_cents"],
        "lookup_key": lookup_key,
        "transfer_lookup_key": True,
        "metadata": {"project": PROJECT, "kind": kind, "credits": str(spec["credits"])},
    }
    if recurring:
        params["recurring"] = recurring
    new_price = stripe.Price.create(**params)
    print(f"        new price {new_price.id}")
    if current:
        stripe.Price.modify(current.id, active=False)


def archive_unlisted_prices(apply):
    """Archive active prices on our products whose lookup_key is no longer configured."""
    wanted = {spec["lookup_key"] for spec in PLANS + TOPUPS}
    for kind, product in PRODUCTS.items():
        for price in stripe.Price.list(product=product["id"], active=True, limit=100).auto_paging_iter():
            if price.lookup_key not in wanted:
                print(f"[price] archive {price.lookup_key or price.id} ({kind}, no longer configured)")
                if apply:
                    stripe.Price.modify(price.id, active=False)


def ensure_portal(apply):
    features = {
        "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
        "payment_method_update": {"enabled": True},
        "invoice_history": {"enabled": True},
        "customer_update": {"enabled": True, "allowed_updates": ["email", "address"]},
    }
    ours = [
        c
        for c in stripe.billing_portal.Configuration.list(limit=100).auto_paging_iter()
        if (c.metadata or {}).get("project") == PROJECT
    ]
    if ours:
        print(f"[portal] update {ours[0].id}")
        if apply:
            stripe.billing_portal.Configuration.modify(ours[0].id, features=features)
        print(f"        STRIPE_PORTAL_CONFIGURATION_ID={ours[0].id}")
    else:
        print("[portal] create configuration")
        if apply:
            config = stripe.billing_portal.Configuration.create(
                features=features, metadata={"project": PROJECT}
            )
            print(f"        STRIPE_PORTAL_CONFIGURATION_ID={config.id}")


def ensure_webhook_events(url, apply):
    endpoints = [
        e
        for e in stripe.WebhookEndpoint.list(limit=100).auto_paging_iter()
        if e.url == url
    ]
    if not endpoints:
        print(f"[webhook] no endpoint with url {url}; not creating one (new secret)")
        return
    endpoint = endpoints[0]
    enabled = list(endpoint.enabled_events)
    missing = [e for e in WEBHOOK_EVENTS if e not in enabled]
    if "*" in enabled or not missing:
        print(f"[webhook] ok {endpoint.id}")
        return
    print(f"[webhook] add to {endpoint.id}: {', '.join(missing)}")
    if apply:
        stripe.WebhookEndpoint.modify(endpoint.id, enabled_events=enabled + missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write to Stripe")
    parser.add_argument("--test", action="store_true", help="use STRIPE_SECRET_KEY_TEST_TRANSCRIPTS")
    parser.add_argument("--webhook-url", help="existing endpoint to add events to")
    args = parser.parse_args()

    load_dotenv()
    key_env = "STRIPE_SECRET_KEY_TEST_TRANSCRIPTS" if args.test else "STRIPE_SECRET_KEY_LIVE"
    stripe.api_key = os.getenv(key_env)
    if not stripe.api_key:
        raise SystemExit(f"{key_env} is not set")
    mode = "LIVE" if "_live_" in stripe.api_key else "test"
    print(f"Stripe {mode} mode ({key_env}), {'APPLY' if args.apply else 'dry run'}\n")

    for kind in PRODUCTS:
        ensure_product(kind, args.apply)
    for spec in PLANS:
        ensure_price("plan", spec, args.apply)
    for spec in TOPUPS:
        ensure_price("topup", spec, args.apply)
    archive_unlisted_prices(args.apply)
    ensure_portal(args.apply)
    if args.webhook_url:
        ensure_webhook_events(args.webhook_url, args.apply)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
