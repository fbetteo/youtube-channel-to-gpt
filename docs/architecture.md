# Architecture

## Active components

| Component | Entry point / responsibility |
| --- | --- |
| FastAPI | `src/transcript_api.py`: website endpoints, models, JWT auth, credits, Stripe, startup/shutdown |
| Developer API | `src/routers/developer_api.py`: `/api/v1` contracts and API-key access |
| Hosted MCP / public docs | `src/routers/mcp.py`, `src/routers/agent_docs.py` |
| YouTube service | `src/youtube_service.py`: discovery, metadata, formatting, Lambda dispatch, timeouts, ZIP retrieval |
| Durable state | `db_youtube_transcripts/job_manager.py`, `discovery_job_manager.py`, `database.py`; `src/hybrid_job_manager.py` wraps transcript jobs |
| Worker | `lambda-transcript-processor/lambda_function.py`: fetch, format, upload to S3, report results |
| Result handling | `src/job_result_processor.py`: shared HTTP/SQS logic; `src/sqs_result_consumer.py` runs separately |
| CLI / local MCP | `cli/src/cli.ts`, built by root `package.json` |
| Web frontend | Separate Next.js repo; see [frontend.md](frontend.md) |

## Main flows

- Single video: auth/credit or anonymous limit -> extraction -> text response.
- Channel/playlist: discover -> normalize selection -> persist job/reserve credits -> background metadata/dispatch -> Lambda -> S3/result message -> database progress/finalization -> polling/ZIP download.
- Discovery jobs have separate persisted state from transcript jobs. Playlist batches create child transcript jobs; inspect their handlers before assuming identical persistence or response shapes.

`src/fastapi_main.py`, `fastapi_assistant.py`, `fastapi_retrieve.py`, `db/`, and `fork_src/` contain other/older flows. Start with the transcript service for current product work; inspect those files when the task touches them.

Business logic still lives partly in large API/service modules. Extend nearby patterns for focused changes; moving everything into routers is not a prerequisite.
