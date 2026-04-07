/**
 * Frontend contract for playlist batch child-job creation endpoints.
 *
 * Endpoints:
 * - POST /playlist/download/batch-selected
 * - GET  /playlist/download/batch-status/{batch_job_id}
 */

export interface BatchPlaylistSelection {
  playlist_id: string;
  title?: string;
}

export interface BatchPlaylistDownloadRequest {
  playlists: BatchPlaylistSelection[];
  channel_name?: string;
  max_concurrent_jobs?: number; // 1..10, defaults to 3

  // Formatting options applied to every child playlist job
  include_timestamps?: boolean;
  include_video_title?: boolean;
  include_video_id?: boolean;
  include_video_url?: boolean;
  include_view_count?: boolean;
  concatenate_all?: boolean;
}

export interface StartBatchPlaylistDownloadResponse {
  batch_job_id: string;
  status: "processing";
  total_playlists: number;
  max_concurrent_jobs: number;
  message: string;
}

export interface BatchCreatedChildJob {
  playlist_id: string;
  playlist_title: string;
  job_id: string;
  total_videos: number;
  credits_reserved: number;
}

export interface BatchFailedPlaylist {
  playlist_id?: string;
  playlist_title?: string;
  error: string;
}

export interface BatchPlaylistDownloadStatusResponse {
  batch_job_id: string;
  status: "processing" | "completed" | "completed_with_errors" | "failed" | string;
  stage: string;
  channel_name?: string;
  total_playlists: number;
  processed_playlists: number;
  created_jobs_count: number;
  failed_playlists_count: number;
  max_concurrent_jobs: number;
  created_jobs: BatchCreatedChildJob[];
  failed_playlists: BatchFailedPlaylist[];
  error?: string;
  elapsed_time: number;
  message?: string;
}

export interface ApiErrorBody {
  detail?: string;
  message?: string;
}

const defaultJsonHeaders: HeadersInit = {
  "Content-Type": "application/json",
};

function buildAuthHeaders(token: string, headers?: HeadersInit): HeadersInit {
  return {
    ...defaultJsonHeaders,
    ...(headers ?? {}),
    Authorization: `Bearer ${token}`,
  };
}

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as ApiErrorBody;
    return payload.detail || payload.message || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

export async function startBatchPlaylistDownload(
  apiBaseUrl: string,
  token: string,
  body: BatchPlaylistDownloadRequest,
): Promise<StartBatchPlaylistDownloadResponse> {
  const response = await fetch(`${apiBaseUrl}/playlist/download/batch-selected`, {
    method: "POST",
    headers: buildAuthHeaders(token),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return (await response.json()) as StartBatchPlaylistDownloadResponse;
}

export async function getBatchPlaylistDownloadStatus(
  apiBaseUrl: string,
  token: string,
  batchJobId: string,
): Promise<BatchPlaylistDownloadStatusResponse> {
  const response = await fetch(
    `${apiBaseUrl}/playlist/download/batch-status/${batchJobId}`,
    {
      method: "GET",
      headers: buildAuthHeaders(token, { Accept: "application/json" }),
    },
  );

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return (await response.json()) as BatchPlaylistDownloadStatusResponse;
}

export function isBatchTerminalStatus(status: string): boolean {
  return status === "completed" || status === "completed_with_errors" || status === "failed";
}
