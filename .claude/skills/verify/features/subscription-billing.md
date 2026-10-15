# Subscription billing

Users subscribe to a monthly plan that grants credits each billing period. Unused monthly credits are dropped at renewal. Only subscribers can buy top-ups, which last 30 days or until the plan ends, whichever comes first. When the plan ends, monthly and top-up credits are removed; only permanent credits (welcome credits, legacy packs, the balance held when first subscribing) remain. Subscribers manage the plan in the Stripe Customer Portal.

## Sub-features

- `sub-checkout` creates a Stripe Checkout session in subscription mode for a plan lookup key.
- `sub-first-period` grants the plan's credits on the first paid invoice and turns the existing balance into permanent credits.
- `sub-renewal` drops unused monthly credits and adds a fresh allowance on each renewal invoice.
- `sub-duplicate` ignores redelivered invoices (no double grant).
- `sub-cancel` removes remaining monthly and top-up credits when the subscription ends and keeps permanent ones.
- `topup-expiry` removes a top-up's remaining credits 30 days after purchase (sweeper every 5 minutes, and on `GET /user/credits`).
- `topup-gate` refuses top-up checkout without an active subscription (403) and allows it with one.
- `sub-already` refuses a second subscription checkout (409).
- `portal` returns a Stripe Customer Portal URL for subscribers (404 without a Stripe customer).
- `legacy-pack` still accepts old one-time pack `price_id`s.

## How to get to it (user POV)

- Website pricing: the plan card calls `POST /payments/create-subscription-checkout {lookup_key}`; top-up cards call `POST /payments/create-checkout-session {lookup_key}`.
- Website dashboard: balance and plan from `GET /user/credits`; Manage Subscription calls `POST /payments/portal`.
- Stripe calls `POST /payments/webhook` with `invoice.paid` and `customer.subscription.*` events.

## Driving it with verify.py

Preconditions:

- `$V doctor` exits 0, including `Stripe key is test mode`.
- Stripe test mode has the prices from `scripts/stripe_setup.py` (`transcript_monthly_1000`, `transcript_topup_1000`). If not, run `poetry run python scripts/stripe_setup.py --test --apply`.
- A fresh UUID with its welcome credits: run `MSYS_NO_PATHCONV=1 $V call GET /user/credits --user <uuid> --run billing` (100 credits).

- **Top-up refused before subscribing.** Run `MSYS_NO_PATHCONV=1 $V call POST /payments/create-checkout-session --user <uuid> --json '{"lookup_key":"transcript_topup_1000"}' --run billing`. Status 403, `Top-ups require an active subscription`.
- **Plan checkout.** Run `MSYS_NO_PATHCONV=1 $V call POST /payments/create-subscription-checkout --user <uuid> --json '{"lookup_key":"transcript_monthly_1000"}' --run billing`. Status 200 with a `checkout.stripe.com` URL and `monthly_credits: 1000`. Completing it needs a browser, so continue with the next step.
- **Subscribe.** Run `$V stripe-subscribe --user <uuid> --run billing`. The printed webhook results include `invoice.paid subscription_create -> 200 {'status': 'success'}`. Then `GET /user/credits`: `credits: 1100`, `subscription_credits: 1000`, `permanent_credits: 100`, `subscription.status: active`.
- **Second subscription refused.** Run the plan checkout command again. Status 409.
- **Top-up allowed.** Run the top-up checkout command again. Status 200 with a checkout URL and `credits: 1000`.
- **Spend.** Run `MSYS_NO_PATHCONV=1 $V call POST /download/transcript/raw --user <uuid> --json '{"youtube_url":"jNQXAC9IVRw"}' --run billing`, then `GET /user/credits`. `credits: 1099`, `subscription_credits: 999` (the soonest-expiring credits are spent first; here the monthly ones).
- **Renewal.** Run `$V stripe-advance --days 32 --run billing`. The printed results include `invoice.paid subscription_cycle -> 200 {'status': 'success'}`. Then `GET /user/credits`: `credits: 1100` (100 permanent + a fresh 1000; the 999 leftover is gone) and a later `subscription.current_period_end`.
- **Idempotency.** Run `$V sql "SELECT session_id, credits FROM stripe_checkout_credits WHERE user_id = '<uuid>'" --run billing`. One row per paid invoice (2), no duplicates.
- **Portal.** Run `MSYS_NO_PATHCONV=1 $V call POST /payments/portal --user <uuid> --run billing`. Status 200 with a `billing.stripe.com` URL.
- **Cancel.** Run `$V stripe-cancel --run billing`. The results include `customer.subscription.deleted canceled -> 200`. Then `GET /user/credits`: `credits: 100`, `subscription: null`.
- **Stored state.** Run `$V sql "SELECT status, monthly_credits FROM user_subscriptions WHERE user_id = '<uuid>'" --run billing`. One row, `status: canceled`.

## Gotchas

- Run each Stripe step once per subscription. `stripe-subscribe` stores one test clock in `instance.json`; a second call replaces the tracked customer, so start a new user and instance for a second lifecycle.
- `stripe-advance` waits for the renewal invoice to be paid before forwarding events. If it prints no `subscription_cycle` line, Stripe was slow: run `$V stripe-forward --run billing` to deliver the remaining events (each event is delivered once).
- Events are forwarded from the Stripe test account, which other projects share; the helper only forwards events whose `customer` is the one it created.
- The portal uses `STRIPE_PORTAL_CONFIGURATION_ID` from `.env` when set, otherwise the account's default test portal. If the test account has no default portal, Stripe returns an error (400 from the API).
- Old pack `price_id`s exist only in live mode, so `legacy-pack` cannot be driven against the test key. Report it as skipped.
- Buying a top-up needs a completed hosted Checkout (a browser), so the helper cannot create one. The top-up checkout step proves the gate and that the session carries `metadata.kind=topup`; `topup-expiry` and top-ups ending with the plan are not drivable here. Report them as skipped, with that reason.
