# Build A Custom GPT From A YouTube Playlist In One Command

Playlists are underrated knowledge products. A course, podcast arc, conference track, or launch series can become a reusable agent knowledge base if the transcript extraction is simple.

With YouTube Transcript Agent:

```bash
npx youtube-transcript-agent playlist download <playlist_url_or_id> --wait --output playlist-transcripts.zip
```

Then upload the ZIP contents into your custom GPT, retrieval store, or long-context workspace.

## Why This Works

- Playlists already group related knowledge.
- Transcripts preserve the original phrasing.
- Agents can summarize, compare, classify, and extract reusable frameworks.
- Async jobs handle larger playlists without keeping the agent blocked.

## Workflow

```bash
npx youtube-transcript-agent auth set-key <api_key>
npx youtube-transcript-agent playlist download PLxxxx --wait --output course.zip
```

For a first pass:

```bash
npx youtube-transcript-agent playlist download PLxxxx --max 10 --wait --output sample.zip
```

After the job finishes, ask the agent to:

- Write a course outline
- Extract definitions and examples
- Create flashcards
- Identify contradictions between videos
- Build a source-grounded FAQ

The result is a custom GPT or agent workspace grounded in real video content.
