# Credits and account

A signed-in website user sees a credit balance. A user's first balance check creates the account with 100 welcome credits. Subscribers also see how much of the balance is monthly (resets at renewal), top-up (expires after 30 days or with the plan) and permanent (never expires).

## Sub-features

- `credits-welcome` creates a new user with 100 credits on the first balance read.
- `credits-read` returns the balance without changing it on later reads.
- `credits-split` reports `subscription_credits`, `topup_credits`, `topups`, `permanent_credits` and `subscription` (null without a plan).
- `profile-read` returns the user ID, credits and email from the token.
- `credits-auth` rejects a missing token (401) and an invalid token (403).

## How to get to it (user POV)

- Open the website dashboard: it calls `GET /user/credits` on load and on the refresh button.
- The website profile view calls `GET /user/profile`.

## Driving it with verify.py

Preconditions:

- `$V doctor` exits 0.
- Pick an unused UUID, for example `22222222-2222-4222-8222-222222222222`.

- **First visit.** Run `MSYS_NO_PATHCONV=1 $V call GET /user/credits --user <uuid> --run credits`. Status 200 with `credits: 100`, `subscription_credits: 0`, `topup_credits: 0`, `permanent_credits: 100`, `subscription: null`.
- **Stored.** Run `$V sql "SELECT credits, permanent_credits FROM user_credits WHERE user_id = '<uuid>'" --run credits`. One row, `credits: 100`.
- **Repeat read.** Run the first command again. Still 100; no second row.
- **Profile.** Run `MSYS_NO_PATHCONV=1 $V call GET /user/profile --user <uuid> --run credits`. 200 with the same `user_id` and `credits: 100`.
- **No token.** Run `MSYS_NO_PATHCONV=1 $V call GET /user/credits --run credits`. Status 401 (403 means a token was sent and rejected).

## Gotchas

- The welcome grant happens on read, so any route that needs a `user_credits` row fails for a user who never read their credits. `POST /user/api-keys` returns 500 (foreign key on `api_keys.user_id`) for such a user. This is a product bug outside the website's normal dashboard-first flow.
- `permanent_credits` in the table stays 0 until a user's first plan period. The API reports the whole balance as permanent for non-subscribers, so assert on the API response, not the raw column.
