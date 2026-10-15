# Transcript API verification map

This directory is the maintained source for verifying the user-facing behavior of the transcript API. Read the index before driving, then use the matching feature file as the recipe.

## Baseline preconditions

- An instance started by `verify.py up` is healthy: `verify.py doctor` exits 0. Never drive an instance this tool did not start, and never the production API.
- The database starts empty. Users exist only after their first `GET /user/credits`, which grants 100 welcome credits (the website dashboard makes that call on load).
- Use a fresh user UUID per journey so balances start at 100.
- In Git Bash, prefix commands that pass URL paths with `MSYS_NO_PATHCONV=1`.

## Driving conventions

- `V="poetry run python .claude/skills/verify/verify.py"`, run from the repo root.
- Website calls: `$V call <METHOD> <path> --user <uuid>`. Developer calls: `--api-key yt_live_...`. Hosted MCP: `--bearer yt_live_...`.
- Pass `--run <feature-id>` on every command so evidence lands in `scripts/e2e_results/verify/<feature-id>/evidence.jsonl`.
- Change state only through the API and the Stripe helpers. `$V sql` is read-only and is for observing.

## Proof and skip reporting

- Capture the action and the resulting state: the request and response, then a second read (`GET /user/credits`, `$V sql`) showing the change.
- Record credits before and after anything that charges or refunds.
- Report an unreachable path with the attempted command and the unmet precondition. Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior, followed by exactly four H2 sections: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify.py`, `Gotchas`.

## Features

- [Credits and account](./credits-and-account.md): welcome credits, balance and profile.
- [Subscription billing](./subscription-billing.md): monthly plan, renewal reset, cancellation, top-up gate, portal, legacy packs.
- [Developer API and CLI](./developer-api.md): API keys, `/api/v1` credits and single transcripts, the `ytx` CLI.
- [Hosted MCP](./hosted-mcp.md): `/mcp` JSON-RPC tool listing and `get_transcript`.
- [Website single video](./website-single-video.md): `/download/transcript/raw` signed in and anonymous; `/summaries/single` availability.

## Not drivable in isolation

- Channel and playlist bulk jobs (`/channel/*`, `/playlist/*`, `/api/v1/transcripts/channel|playlist`, MCP `start_*_job`) need the production Lambda, S3 and SQS. The instance's AWS credentials are deliberately invalid, so dispatch fails. Their completion path is not proven by this harness; report it as skipped with that reason.
- Single-video summaries need `SUMMARY_OPENAI_API_KEY`, which the instance leaves unset.
