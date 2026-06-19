#!/usr/bin/env node
import { createInterface } from "node:readline";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
const CONFIG_DIR = join(homedir(), ".youtube-transcript-agent");
const CONFIG_PATH = join(CONFIG_DIR, "config.json");
const DEFAULT_BASE_URL = "https://api.youtubetranscripts.fbetteo.com";
const TOOL_NAMES = [
    "get_transcript",
    "get_channel_info",
    "list_channel_videos",
    "start_channel_job",
    "start_playlist_job",
    "get_job_status",
    "get_download_url",
];
function printHelp() {
    console.log(`youtube-transcript-agent (ytx)

Usage:
  ytx auth set-key <api_key> [--base-url <url>]
  ytx credits
  ytx transcript <video_url_or_id> [--timestamps] [--json]
  ytx channel info <channel> [--json]
  ytx channel videos <channel> [--json]
  ytx channel download <channel> [--max <n>] [--timestamps] [--concat] [--wait] [--output <zip>]
  ytx playlist download <playlist> [--max <n>] [--timestamps] [--concat] [--wait] [--output <zip>]
  ytx jobs status <job_id> [--json]
  ytx jobs download <job_id> [--output <zip>]
  ytx mcp

Environment:
  YOUTUBE_TRANSCRIPT_API_KEY       API key, preferred over local config
  YOUTUBE_TRANSCRIPT_API_BASE_URL  Optional local/staging API override, default ${DEFAULT_BASE_URL}`);
}
function getFlag(args, name, fallback) {
    const index = args.indexOf(name);
    if (index === -1)
        return fallback;
    return args[index + 1] ?? fallback;
}
function hasFlag(args, name) {
    return args.includes(name);
}
function requirePositional(value, usage) {
    if (!value || value.startsWith("-")) {
        throw new Error(`Usage: ${usage}`);
    }
    return value;
}
async function readConfig() {
    try {
        return JSON.parse(await readFile(CONFIG_PATH, "utf8"));
    }
    catch {
        return {};
    }
}
async function writeConfig(config) {
    await mkdir(CONFIG_DIR, { recursive: true });
    await writeFile(CONFIG_PATH, JSON.stringify(config, null, 2), "utf8");
}
async function getBaseUrl() {
    const config = await readConfig();
    return (process.env.YOUTUBE_TRANSCRIPT_API_BASE_URL ||
        process.env.API_BASE_URL ||
        config.apiBaseUrl ||
        DEFAULT_BASE_URL).replace(/\/+$/, "");
}
async function getApiKey() {
    const config = await readConfig();
    const apiKey = process.env.YOUTUBE_TRANSCRIPT_API_KEY || config.apiKey;
    if (!apiKey) {
        throw new Error("Missing API key. Set YOUTUBE_TRANSCRIPT_API_KEY or run: ytx auth set-key <api_key>");
    }
    return apiKey;
}
async function apiFetch(path, init = {}) {
    const baseUrl = await getBaseUrl();
    const apiKey = await getApiKey();
    const headers = new Headers(init.headers);
    headers.set("X-API-Key", apiKey);
    if (init.body && !headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
    }
    return fetch(`${baseUrl}${path}`, { ...init, headers });
}
async function parseResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json"))
        return response.json();
    return response.text();
}
async function requestJson(path, init = {}) {
    const response = await apiFetch(path, init);
    const body = await parseResponse(response);
    if (!response.ok) {
        const message = typeof body === "object" && body !== null && "detail" in body
            ? String(body.detail)
            : `HTTP ${response.status}`;
        throw new Error(message);
    }
    return body;
}
function printPayload(payload, asJson) {
    if (asJson || typeof payload !== "object" || payload === null) {
        console.log(typeof payload === "string" ? payload : JSON.stringify(payload, null, 2));
        return;
    }
    if ("transcript" in payload) {
        const p = payload;
        console.log(`# ${p.title ?? p.video_id ?? "Transcript"}\n`);
        console.log(String(p.transcript ?? ""));
        return;
    }
    console.log(JSON.stringify(payload, null, 2));
}
function optionsFromArgs(args) {
    return {
        include_timestamps: hasFlag(args, "--timestamps"),
        concatenate_all: hasFlag(args, "--concat"),
        include_video_title: true,
        include_video_id: true,
        include_video_url: true,
        include_view_count: false,
    };
}
async function waitForJob(jobId) {
    const started = Date.now();
    const timeoutMs = 30 * 60 * 1000;
    while (Date.now() - started < timeoutMs) {
        const status = (await requestJson(`/api/v1/jobs/${jobId}`));
        console.error(`status=${status.status} processed=${status.processed_count ?? 0}/${status.total_videos ?? 0}`);
        if (status.download_ready || ["failed"].includes(String(status.status))) {
            return status;
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
    }
    throw new Error(`Timed out waiting for job ${jobId}`);
}
async function downloadJob(jobId, output) {
    const response = await apiFetch(`/api/v1/jobs/${jobId}/download`);
    if (!response.ok || !response.body) {
        const body = await parseResponse(response);
        throw new Error(typeof body === "string" ? body : JSON.stringify(body));
    }
    await mkdir(dirname(output), { recursive: true });
    const writer = createWriteStream(output);
    const reader = response.body.getReader();
    for (;;) {
        const { value, done } = await reader.read();
        if (done)
            break;
        writer.write(Buffer.from(value));
    }
    await new Promise((resolve, reject) => {
        writer.end(resolve);
        writer.on("error", reject);
    });
    console.log(output);
}
async function command(args) {
    const [area, action, value] = args;
    if (!area || area === "--help" || area === "-h") {
        printHelp();
        return;
    }
    if (area === "auth" && action === "set-key") {
        if (!value)
            throw new Error("Usage: ytx auth set-key <api_key>");
        const config = await readConfig();
        config.apiKey = value;
        const baseUrl = getFlag(args, "--base-url");
        if (baseUrl)
            config.apiBaseUrl = baseUrl.replace(/\/+$/, "");
        await writeConfig(config);
        console.log("API key saved.");
        return;
    }
    if (area === "credits") {
        printPayload(await requestJson("/api/v1/account/credits"), hasFlag(args, "--json"));
        return;
    }
    if (area === "transcript") {
        const video = requirePositional(action, "ytx transcript <video_url_or_id>");
        const payload = await requestJson("/api/v1/transcripts/single", {
            method: "POST",
            body: JSON.stringify({
                video_url: video,
                include_timestamps: hasFlag(args, "--timestamps"),
            }),
        });
        printPayload(payload, hasFlag(args, "--json"));
        return;
    }
    if (area === "channel" && action === "info") {
        const channel = requirePositional(value, "ytx channel info <channel>");
        printPayload(await requestJson(`/api/v1/channels/${encodeURIComponent(channel)}/info`), hasFlag(args, "--json"));
        return;
    }
    if (area === "channel" && action === "videos") {
        const channel = requirePositional(value, "ytx channel videos <channel>");
        printPayload(await requestJson(`/api/v1/channels/${encodeURIComponent(channel)}/videos`), hasFlag(args, "--json"));
        return;
    }
    if (area === "channel" && action === "download") {
        const channel = requirePositional(value, "ytx channel download <channel>");
        const job = (await requestJson("/api/v1/transcripts/channel", {
            method: "POST",
            body: JSON.stringify({
                channel,
                max_videos: getFlag(args, "--max") ? Number(getFlag(args, "--max")) : undefined,
                options: optionsFromArgs(args),
            }),
        }));
        printPayload(job, true);
        if (hasFlag(args, "--wait")) {
            const status = await waitForJob(String(job.job_id));
            if (status.download_ready) {
                await downloadJob(String(job.job_id), getFlag(args, "--output", "transcripts.zip"));
            }
        }
        return;
    }
    if (area === "playlist" && action === "download") {
        const playlist = requirePositional(value, "ytx playlist download <playlist>");
        const job = (await requestJson("/api/v1/transcripts/playlist", {
            method: "POST",
            body: JSON.stringify({
                playlist,
                max_videos: getFlag(args, "--max") ? Number(getFlag(args, "--max")) : undefined,
                options: optionsFromArgs(args),
            }),
        }));
        printPayload(job, true);
        if (hasFlag(args, "--wait")) {
            const status = await waitForJob(String(job.job_id));
            if (status.download_ready) {
                await downloadJob(String(job.job_id), getFlag(args, "--output", "transcripts.zip"));
            }
        }
        return;
    }
    if (area === "jobs" && action === "status") {
        const jobId = requirePositional(value, "ytx jobs status <job_id>");
        printPayload(await requestJson(`/api/v1/jobs/${jobId}`), hasFlag(args, "--json"));
        return;
    }
    if (area === "jobs" && action === "download") {
        const jobId = requirePositional(value, "ytx jobs download <job_id>");
        await downloadJob(jobId, getFlag(args, "--output", "transcripts.zip"));
        return;
    }
    if (area === "mcp") {
        await runMcp();
        return;
    }
    throw new Error(`Unknown command: ${args.join(" ")}`);
}
async function callTool(name, args) {
    if (!TOOL_NAMES.includes(name))
        throw new Error(`Unknown tool: ${name}`);
    if (name === "get_transcript") {
        return (await requestJson("/api/v1/transcripts/single", {
            method: "POST",
            body: JSON.stringify({
                video_url: args.video_url,
                include_timestamps: Boolean(args.include_timestamps),
            }),
        }));
    }
    if (name === "get_channel_info") {
        return (await requestJson(`/api/v1/channels/${encodeURIComponent(String(args.channel))}/info`));
    }
    if (name === "list_channel_videos") {
        return (await requestJson(`/api/v1/channels/${encodeURIComponent(String(args.channel))}/videos`));
    }
    if (name === "start_channel_job") {
        return (await requestJson("/api/v1/transcripts/channel", {
            method: "POST",
            body: JSON.stringify({
                channel: args.channel,
                max_videos: args.max_videos,
                options: {
                    include_timestamps: Boolean(args.include_timestamps),
                    concatenate_all: Boolean(args.concatenate_all),
                },
            }),
        }));
    }
    if (name === "start_playlist_job") {
        return (await requestJson("/api/v1/transcripts/playlist", {
            method: "POST",
            body: JSON.stringify({
                playlist: args.playlist,
                max_videos: args.max_videos,
                options: {
                    include_timestamps: Boolean(args.include_timestamps),
                    concatenate_all: Boolean(args.concatenate_all),
                },
            }),
        }));
    }
    const status = (await requestJson(`/api/v1/jobs/${String(args.job_id)}`));
    if (status.download_url) {
        status.download_url = `${await getBaseUrl()}${status.download_url}`;
    }
    if (name === "get_download_url" && !status.download_ready) {
        throw new Error(`Job is not ready for download. Status: ${status.status}`);
    }
    return status;
}
const MCP_TOOLS = TOOL_NAMES.map((name) => ({
    name,
    description: `YouTube Transcript Agent tool: ${name}`,
    inputSchema: { type: "object", additionalProperties: true },
}));
async function runMcp() {
    const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of rl) {
        if (!line.trim())
            continue;
        let request;
        try {
            request = JSON.parse(line);
            const id = request.id;
            const method = String(request.method);
            let result;
            if (method === "initialize") {
                result = {
                    protocolVersion: "2025-06-18",
                    capabilities: { tools: {} },
                    serverInfo: { name: "youtube-transcript-agent", version: "0.1.0" },
                };
            }
            else if (method === "tools/list") {
                result = { tools: MCP_TOOLS };
            }
            else if (method === "tools/call") {
                const params = (request.params || {});
                const payload = await callTool(String(params.name), (params.arguments || {}));
                result = {
                    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
                };
            }
            else if (method === "notifications/initialized") {
                result = {};
            }
            else {
                throw new Error(`Method not found: ${method}`);
            }
            process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, result }) + "\n");
        }
        catch (error) {
            const id = request && typeof request === "object" ? request.id : null;
            process.stdout.write(JSON.stringify({
                jsonrpc: "2.0",
                id,
                error: { code: -32000, message: error instanceof Error ? error.message : String(error) },
            }) + "\n");
        }
    }
}
command(process.argv.slice(2)).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
});
