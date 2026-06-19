# X Launch Playbook

## Positioning

Primary line:

```text
Turn any YouTube video, playlist, or channel into agent-ready context.
```

## Post Angles

1. Agent memory

```text
I wanted Claude/Codex to reason over an entire YouTube channel.

So I made it one command:

npx youtube-transcript-agent channel download @channel --wait --output memory.zip

Now agents can work with the actual transcript corpus instead of guessing from titles.
```

2. Playlist to custom GPT

```text
YouTube playlists are already mini-courses.

The missing piece is turning them into clean context:

npx youtube-transcript-agent playlist download <playlist> --wait

Upload the transcripts into a custom GPT, retrieval store, or long-context agent.
```

3. MCP launch

```text
Launched MCP for YouTube transcripts.

Agents can now:
- fetch a video transcript
- inspect a channel
- start a channel/playlist job
- poll until the ZIP is ready

Hosted MCP: /mcp
Local MCP: npx youtube-transcript-agent mcp
```

## Distribution Notes

- Time posts around agent releases, coding-agent launches, and custom GPT workflows.
- Use short demo videos showing an agent calling `get_transcript` or downloading a channel.
- Repost the same article with different clips and angles rather than identical copy.
- If paying creators, use clear paid-promotion labels.
