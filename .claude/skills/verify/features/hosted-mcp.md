# Hosted MCP

AI agents connect to `/mcp` with an API key as a bearer token, list the transcript tools, and call them. Tool calls reuse the developer API, so they charge the same credits.

## Sub-features

- `mcp-initialize` answers the JSON-RPC `initialize` handshake.
- `mcp-tools` lists `get_transcript`, `get_channel_info`, `list_channel_videos`, `start_channel_job`, `start_playlist_job`, `get_job_status`, `get_download_url` and `cancel_job`.
- `mcp-transcript` returns a transcript through `tools/call` `get_transcript` and charges 1 credit.
- `mcp-auth` rejects tool calls without a valid key.

## How to get to it (user POV)

- An MCP client configured with URL `http://<host>/mcp` and header `Authorization: Bearer <api_key>`.
- The local CLI also offers the same tools over stdio (`ytx mcp`); that path is separate and not covered here.

## Driving it with verify.py

Preconditions:

- `$V doctor` exits 0.
- An API key from the [developer API](./developer-api.md) create-key step, for a user with known credits.

- **Handshake.** Run `MSYS_NO_PATHCONV=1 $V call POST /mcp --bearer <key> --json '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' --run mcp`. 200 with `result.serverInfo.name: youtube-transcript-agent`.
- **List tools.** Run `MSYS_NO_PATHCONV=1 $V call POST /mcp --bearer <key> --json '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' --run mcp`. `result.tools` has the 8 names above.
- **Call get_transcript.** Run `MSYS_NO_PATHCONV=1 $V call POST /mcp --bearer <key> --json '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_transcript","arguments":{"video_url":"jNQXAC9IVRw"}}}' --run mcp`. `result.content` holds the transcript.
- **Charged.** Run `MSYS_NO_PATHCONV=1 $V call GET /api/v1/account/credits --api-key <key> --run mcp`. One credit fewer than before the tool call.
- **No key.** Run the tool call without `--bearer`. A JSON-RPC error or 401, and no credit change.

## Gotchas

- MCP takes the key as `Authorization: Bearer`, not `X-API-Key`. `--api-key` does not authenticate `/mcp`.
- JSON-RPC errors can come back with HTTP 200. Assert on `result` versus `error` in the body, not only on the status.
- Job tools (`start_channel_job`, `start_playlist_job`) dispatch to Lambda, which is disabled in this instance. See the README's "Not drivable in isolation".
