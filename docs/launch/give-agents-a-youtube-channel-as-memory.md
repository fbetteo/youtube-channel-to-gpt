# Give Claude/Codex Any YouTube Channel As Memory

Most useful expertise is trapped in long videos. Agents can reason over it, but only after the transcript is easy to fetch, store, and refresh.

YouTube Transcript Agent turns a channel into a ZIP of clean transcript files:

```bash
npx youtube-transcript-agent channel download @channel --max 100 --wait --output channel-memory.zip
```

Use it for:

- Founder interviews and market research
- Course and tutorial channels
- Podcasts with recurring expert guests
- Competitive intelligence from product demos
- Building custom GPT or long-context knowledge packs

Agents work better when they can cite the actual source material. Instead of asking an agent to guess what a creator said, give it the transcript corpus.

## Demo Flow

1. Set an API key:

```bash
npx youtube-transcript-agent auth set-key <api_key>
```

2. Download a channel:

```bash
npx youtube-transcript-agent channel download @mkbhd --max 25 --wait --output mkbhd.zip
```

3. Ask your agent to extract themes, product mentions, audience objections, or a searchable brief from the ZIP.

## Agent Integration

Local agents can use:

```bash
npx youtube-transcript-agent mcp
```

Hosted agents can use:

```text
POST /mcp
Authorization: Bearer <api_key>
```

The key idea: make video knowledge callable, repeatable, and agent-native.
