# Frontend integration

Frontend: `../../youtube-transcript-v0-chakra` from the backend root, locally `C:/Users/franb/projects/youtube-transcript-v0-chakra`. It is a separate Git repo. Read its [AGENTS.md](../../../youtube-transcript-v0-chakra/AGENTS.md) before editing; review changes in each repo independently.

It uses Next.js App Router, TypeScript, Chakra UI, and Supabase SSR. Frontend/backend run separately. Server-side proxies use `TRANSCRIPT_API_URL`, defaulting to `http://127.0.0.1:8000`.

## Where to look there

| Topic | Frontend source |
| --- | --- |
| UI and marketing/blog conventions | `docs/frontend.md`, `components/try-it-out.tsx` |
| API proxies and polling | `docs/api-backend.md`, `app/api/`, `lib/transcript-service.ts` |
| Sessions and tokens | `docs/auth.md`, `utils/supabase/`, `middleware.ts` |
| Checkout and referrals | `docs/payments.md`, `app/api/payments/checkout/route.ts` |
| Local configuration | `docs/ops-env.md`, `.env.local.example` |
| Analytics | `docs/analytics.md` |

## Shared contracts

- Single-video summaries use the dedicated frontend `/api/summaries/single` proxy
  to backend `/summaries/single`, with required Supabase bearer auth and SSE
  progress/result events. `components/single-video-summary.tsx` and
  `lib/summary-service.ts` implement controls, result tabs, source links,
  cancellation, and Markdown export. Keep the transcript after generation fails.

- Website proxies mainly call `/channel`, `/playlist`, `/download`, `/user`, `/payments`. Developer integrations use `/api/v1` and MCP; confirm which surface a feature calls.
- Preserve bearer forwarding for authenticated backend calls. Next.js session cookies are not backend JWT credentials.
- Return real backend job IDs and poll backend state; avoid independent job stores in frontend routes. Preserve interval cleanup/adaptive polling.
- Check request fields, error mapping (`detail` to `error`), statuses, download readiness, formatting/language options, and cancellation on both sides.
- Playlist batch example types live in `docs/frontend/playlistBatchApi.contract.ts` in this repo; update when the contract changes.

Shared features may require edits in both repos. Keep backend details here and UI/session details in frontend topic docs.
