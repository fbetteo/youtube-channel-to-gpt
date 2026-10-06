# Working in this repo

This repo contains the Python transcript backend, AWS worker, and TypeScript CLI/MCP client. Read only the topic docs relevant to your task.

## How we work

- Keep changes small, explicit, and focused. Reuse existing patterns before adding abstractions or services.
- Inspect implementation before changing behavior. Document inconsistencies rather than assuming every endpoint follows the same rules.
- Preserve Poetry for Python and npm for the CLI. Prefer cmd on Windows; Linux deployment commands may use bash.
- Validate affected behavior and report what was checked and any remaining limitations.
- Keep credentials out of code, docs, and logs. Explain database migrations and production changes before running them.
- Update the relevant topic doc when behavior changes. Keep this file a map and avoid duplicating rules across agent instruction files.

## Where to look

| Task | Read |
| --- | --- |
| Service boundaries and entry points | [docs/architecture.md](docs/architecture.md) |
| API contracts, errors, CLI, and MCP | [docs/api.md](docs/api.md) |
| Authentication and ownership | [docs/auth.md](docs/auth.md) |
| Jobs, workers, credits, and payments | [docs/jobs-and-credits.md](docs/jobs-and-credits.md) |
| Local setup, checks, and configuration | [docs/development.md](docs/development.md) |
| Frontend changes and shared contracts | [docs/frontend.md](docs/frontend.md) |

Frontend repo: `../../youtube-transcript-v0-chakra` (from this repo), locally `C:/Users/franb/projects/youtube-transcript-v0-chakra`. Read its [AGENTS.md](../../youtube-transcript-v0-chakra/AGENTS.md) before editing it; cross-project tasks may require changes in both repos.

Topic docs are the maintained contributor guidance. [docs/agent.md](docs/agent.md) is the public guide for agents **using** the product. Root setup guides and prompt files are historical context; verify them against code before following them.
