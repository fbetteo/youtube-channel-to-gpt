# YouTube Transcript API Service

## Project Overview

**PRIMARY FOCUS: Transcript API** (`src/transcript_api.py`)

A standalone FastAPI service for downloading YouTube transcripts with:
- **Credit-based access control** for authenticated users
- **Stripe payment integration** for credit purchases
- **AWS Lambda-based concurrent transcript processing**
- **PostgreSQL database** for job tracking and user management
- **S3 storage** for transcript files
- **Channel and Playlist support** with quota-optimized YouTube API usage

---

## **CORE ARCHITECTURE**

### System Components

1. **FastAPI Backend** (`src/transcript_api.py`)
   - REST API endpoints for user requests
   - Authentication (Supabase JWT) and credit management
   - Job creation and status tracking
   - Stripe webhook handling

2. **YouTube Service Layer** (`src/youtube_service.py`)
   - YouTube Data API v3 integration (quota-optimized)
   - Video metadata pre-fetching (batch operations)
   - Lambda function dispatch and coordination
   - Job timeout monitoring

3. **AWS Lambda Workers** (`lambda-transcript-processor/lambda_function.py`)
   - Concurrent transcript extraction (20+ simultaneous)
   - YouTube Transcript API integration with proxy support
   - S3 upload for completed transcripts
   - Callback to FastAPI for status updates

4. **Hybrid Job Manager** (`src/hybrid_job_manager.py`)
   - Unified interface for database operations
   - Job and video status tracking
   - Atomic progress updates with concurrency control

5. **PostgreSQL Database** (`db_youtube_transcripts/`)
   - `user_credits`: Credit balance tracking
   - `jobs`: Job metadata, status, and progress
   - `job_videos`: Individual video status and metadata

6. **S3 Storage**
   - Temporary storage for completed transcripts
   - Organized by: `{user_id}/{job_id}/{video_id}.txt`
   - Retrieved and packaged into ZIP files on download

### Key Data Flow (Channel/Playlist Download)

```
1. USER REQUEST → FastAPI Endpoint
   ├─ Validate JWT & check credits
   ├─ Reserve credits upfront
   └─ Create job in database (status: "initializing")
   
2. BACKGROUND TASK (prefetch_and_dispatch_task)
   ├─ Pre-fetch video metadata in batches (50 videos per API call)
   ├─ Update job status: "dispatching"
   ├─ Dispatch 20+ Lambda functions concurrently (fire-and-forget)
   └─ Update job status: "processing"
   
3. LAMBDA WORKERS (parallel execution)
   ├─ Fetch transcript from YouTube Transcript API
   ├─ Format with user-specified options
   ├─ Upload to S3: s3://{bucket}/{user_id}/{job_id}/{video_id}.txt
   └─ Callback to FastAPI: /internal/job/{job_id}/video-complete
   
4. CALLBACK PROCESSING (for each video)
   ├─ Update job_videos table (status: "completed")
   ├─ Increment processed_count and credits_used
   ├─ Check if all videos processed
   └─ If complete: finalize credits (refund unused)
   
5. USER DOWNLOAD REQUEST
   ├─ Fetch all S3 files for job_id
   ├─ Assemble into ZIP (individual files or concatenated)
   └─ Stream ZIP to user
```

### Authentication & Access Control

```python
# Supabase JWT validation for authenticated users
async def validate_jwt(credentials: HTTPAuthorizationCredentials = Depends(security))
# Validates token, extracts user_id from payload

# Anonymous users: Rate limiting via IP address
check_anonymous_rate_limit(request: Request)
# 3 downloads per hour per IP (configurable in rate_limiter.py)
```

### Database Schema (Reference)

**user_credits table**
- `user_id` (UUID, PK): Supabase user identifier
- `credits` (INTEGER): Current credit balance
- `credits_reserved` (INTEGER): Credits in active reservations
- `total_credits_purchased` (INTEGER): Lifetime purchases
- `created_at`, `updated_at` (TIMESTAMP)

**jobs table**
- `job_id` (UUID, PK): Unique job identifier
- `user_id` (UUID, FK): User who created the job
- `status` (VARCHAR): Job status (initializing, dispatching, processing, completed, failed, etc.)
- `source_type` (VARCHAR): "channel" or "playlist"
- `source_id`, `source_name` (TEXT): Channel ID/name or Playlist ID/name
- `total_videos`, `processed_count`, `completed`, `failed_count` (INTEGER)
- `credits_reserved`, `credits_used` (INTEGER)
- `reservation_id` (UUID): Credit reservation identifier
- `formatting_options` (JSONB): User preferences for transcript format
- `videos_metadata` (JSONB): Pre-fetched metadata for all videos
- `lambda_dispatched_count` (INTEGER), `lambda_dispatch_time` (TIMESTAMP)
- Concurrency control via `version` field

**job_videos table**
- `job_id` (UUID, FK): Reference to jobs table
- `video_id` (TEXT): YouTube video ID
- `title`, `url`, `description` (TEXT): Video metadata
- `duration_seconds`, `duration_category` (INTEGER, VARCHAR)
- `view_count`, `like_count`, `comment_count` (BIGINT/INTEGER)
- `status` (VARCHAR): Video processing status
- `s3_key` (TEXT): S3 location of completed transcript
- `transcript_length` (INTEGER): Character count
- `error_message` (TEXT): Failure details if applicable

### S3 Storage Structure

```
s3://{bucket_name}/
└── {user_id}/
    └── {job_id}/
        ├── {video_id_1}.txt
        ├── {video_id_2}.txt
        └── {video_id_n}.txt
```

Transcripts are stored temporarily in S3 and retrieved on-demand when users download results.

## Development Workflows

### Running the Service

```bash
# Start the Transcript API
uvicorn src.transcript_api:app --reload --port 8001

# Alternative startup method
cd src && python transcript_api.py
```

### Environment Setup

Required environment variables:

```bash
# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key

# Authentication
SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS=your_supabase_secret

# Database (PostgreSQL)
DB_HOST_YOUTUBE_TRANSCRIPTS=your_db_host
DB_NAME_YOUTUBE_TRANSCRIPTS=your_db_name
DB_USERNAME_YOUTUBE_TRANSCRIPTS=your_db_user
DB_PASSWORD_YOUTUBE_TRANSCRIPTS=your_db_pass
DB_PORT_YOUTUBE_TRANSCRIPTS=5432

# Stripe Payments
STRIPE_SECRET_KEY_LIVE=your_stripe_secret
STRIPE_WEBHOOK_SECRET_TRANSCRIPTS=your_webhook_secret

# AWS Lambda
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
LAMBDA_FUNCTION_NAME=transcript-processor
S3_BUCKET_NAME=your-s3-bucket

# API Configuration
API_BASE_URL=https://your-api.com
FRONTEND_URL_YOUTUBE_TRANSCRIPTS=http://localhost:3000

# Optional: Proxy for rate limit bypass
WEBSHARE_PROXY_USERNAME=your_proxy_user
WEBSHARE_PROXY_PASSWORD=your_proxy_pass

# Job Configuration
JOB_TIMEOUT_MINUTES=10
```

### Active API Endpoints

**Core Transcript Functionality**:
- `GET /` - Health check and API info
- `POST /download/transcript/raw` - Single video transcript download (raw text)
- `GET /video-info` - Single video metadata

**Channel Endpoints**:
- `GET /channel/{channel_name}` - Channel info with basic video listing
- `GET /channel/{channel_name}/all-videos` - Complete video list with duration categorization
- `POST /channel/download/selected` - Download transcripts for selected videos
- `GET /channel/videos-status/{job_id}` - Detailed per-video status
- `GET /channel/download/status/{job_id}` - Job progress and statistics
- `GET /channel/download/results/{job_id}` - Download completed transcript ZIP

**Playlist Endpoints** (identical workflow to channels):
- `GET /playlist/{playlist_id}` - Playlist info with metadata
- `GET /playlist/{playlist_id}/all-videos` - Complete playlist video list
- `POST /playlist/download/selected` - Download transcripts for selected playlist videos

**User Management & Billing**:
- `GET /user/credits` - Check credit balance
- `GET /user/profile` - User profile information
- `GET /user/download-history` - Past download jobs
- `POST /payments/create-checkout-session` - Stripe checkout
- `POST /payments/webhook` - Stripe webhook handler

**Internal Endpoints** (called by Lambda, not for direct user access):
- `POST /internal/job/{job_id}/video-complete` - Lambda callback for successful processing
- `POST /internal/job/{job_id}/video-failed` - Lambda callback for failed processing

##### Testing Endpoints
```bash
# Single video transcript download
curl -X POST "http://localhost:8001/download/transcript/raw" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://youtu.be/VIDEO_ID", "include_timestamps": false}'

# Get channel info
curl -X GET "http://localhost:8001/channel/channelname"

# Get all videos for a channel (for selection UI)
curl -X GET "http://localhost:8001/channel/channelname/all-videos"

# Download selected channel videos (authenticated)
curl -X POST "http://localhost:8001/channel/download/selected" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "channel_name": "channelname",
    "videos": [
      {"id": "video1", "title": "Video 1", "url": "https://youtube.com/watch?v=video1"},
      {"id": "video2", "title": "Video 2", "url": "https://youtube.com/watch?v=video2"}
    ],
    "include_timestamps": true,
    "include_video_title": false,
    "concatenate_all": true
  }'

# Check download progress
curl -X GET "http://localhost:8001/channel/download/status/your-job-id" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Download completed transcripts
curl -X GET "http://localhost:8001/channel/download/results/your-job-id" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Check user credits
curl -X GET "http://localhost:8001/user/credits" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# PLAYLIST ENDPOINTS (same workflow as channels)

# Get playlist info
curl -X GET "http://localhost:8001/playlist/PLxxxxxx"

# Get all videos for playlist selection
curl -X GET "http://localhost:8001/playlist/PLxxxxxx/all-videos"

# Download selected playlist videos
curl -X POST "http://localhost:8001/playlist/download/selected" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "playlist_name": "PLxxxxxx",
    "videos": [{"id": "video1", "title": "Video 1"}],
    "include_timestamps": true,
    "concatenate_all": false
  }'
```

## Core API Patterns

### Unified Download Workflow (Channels & Playlists)

Both channels and playlists follow the same processing workflow with minimal differences:

**Similarities:**
- Same API structure and endpoints (replace `/channel/` with `/playlist/`)
- Same metadata pre-fetching and batch processing
- Same Lambda dispatch mechanism
- Same credit management and job tracking
- Same formatting options and ZIP generation
- Both use YouTube's `playlistItems.list()` API internally (channels use "uploads playlist")

**Differences:**
- **Input**: Channels use `channel_name`, playlists use `playlist_id`
- **Discovery**: Channels require extra step to find "uploads playlist ID"
- **URL Parsing**: Different regex patterns for URL extraction

**Workflow Steps:**

1. **Video Discovery** (`GET /channel/{channel_name}/all-videos` or `GET /playlist/{playlist_id}/all-videos`)
   - Channels: Resolve channel → get uploads playlist ID → fetch videos
   - Playlists: Directly fetch videos from playlist ID
   - Returns: Complete list with duration categorization (short/medium/long)
   - Uses efficient `playlistItems.list()` (1 unit) + batch `videos.list()` (1 unit per 50 videos)
   - **Quota efficiency**: 99%+ savings vs. deprecated `search.list` approach

2. **Selected Download** (`POST /channel/download/selected` or `POST /playlist/download/selected`)
   - Validates JWT and checks credits
   - Reserves credits upfront (1 per video)
   - Creates job in database (status: "initializing")
   - Starts background task: `prefetch_and_dispatch_task(job_id)`
   - Returns job_id immediately (~200ms response time)

3. **Background Processing** (`youtube_service.prefetch_and_dispatch_task`)
   - Pre-fetches metadata for all videos in batches (50 per API call)
   - Updates job status: "prefetching_metadata" → "dispatching"
   - Invokes Lambda functions concurrently (20+ simultaneous, fire-and-forget)
   - Updates job status: "processing"
   - Starts timeout monitoring task

4. **Lambda Processing** (`lambda_function.lambda_handler`)
   - Fetches transcript from YouTube Transcript API
   - Applies formatting options (timestamps, headers, etc.)
   - Uploads to S3: `s3://{bucket}/{user_id}/{job_id}/{video_id}.txt`
   - Calls back to FastAPI: `/internal/job/{job_id}/video-complete` or `/video-failed`

5. **Progress Tracking** (`GET /channel/download/status/{job_id}`)
   - Real-time updates from database
   - Shows: processed_count, completed, failed_count, credits_used
   - Status values: initializing, prefetching_metadata, dispatching, processing, completed, completed_with_errors, failed

6. **Result Retrieval** (`GET /channel/download/results/{job_id}`)
   - Fetches all completed transcripts from S3
   - Assembles into ZIP file (individual files or concatenated)
   - Streams ZIP to user

**Opportunity for Simplification:**
Since both channels and playlists ultimately use the same `playlistItems.list()` API and processing logic, there's potential to:
- Merge channel/playlist endpoints into unified `/source/` endpoints
- Auto-detect input type (channel name vs playlist ID) from URL patterns
- Reduce code duplication in URL parsing and validation

However, keeping them separate provides:
- Clearer API documentation for users
- Explicit error messages for each source type
- Easier to add source-specific features in future

### Transcript Formatting Options

Users can customize transcript output via request body:

```python
{
  "channel_name": "example_channel",  # or "playlist_name" for playlists
  "videos": [...],
  "include_timestamps": false,      # Toggle [MM:SS] timestamps
  "include_video_title": true,      # Toggle video title in header
  "include_video_id": true,         # Toggle video ID in header  
  "include_video_url": true,        # Toggle video URL in header
  "include_view_count": false,      # Toggle view count in header
  "concatenate_all": false          # Single file vs individual files
}
```

**Output Formats:**

- **Individual Files** (default): ZIP contains separate `.txt` files per video
  ```
  transcripts_{channel_name}_{timestamp}.zip
  ├── video_id_1.txt
  ├── video_id_2.txt
  └── video_id_n.txt
  ```

- **Concatenated Mode**: ZIP contains single file with all transcripts
  ```
  transcripts_{channel_name}_{timestamp}.zip
  └── {channel_name}_all_transcripts.txt
      ├── [Channel Header]
      ├── === Video 1 Title ===
      │   [Transcript content]
      ├── === Video 2 Title ===
      │   [Transcript content]
      └── ...
```

### AWS Lambda Integration

**Lambda Function Dispatch** (`youtube_service.dispatch_lambdas_concurrently`):
- Uses boto3 Lambda client with `InvocationType="Event"` (fire-and-forget)
- Dispatches 20+ Lambda functions concurrently without semaphore limits
- Each Lambda receives: video_id, job_id, user_id, metadata, formatting options
- No waiting for Lambda completion - true async execution

**Lambda Processing** (`lambda_function.lambda_handler`):
```python
# Lambda receives event with:
{
  "video_id": "youtube_video_id",
  "job_id": "unique_job_identifier",
  "user_id": "user_identifier",
  "api_base_url": "https://your-api.com",
  "include_timestamps": true/false,
  "include_video_title": true/false,
  "include_video_id": true/false,
  "include_video_url": true/false,
  "include_view_count": true/false,
  "pre_fetched_metadata": {...}
}
```

**Lambda Workflow:**
1. Create fresh YouTubeTranscriptApi instance (with proxy if configured)
2. List available transcripts, prefer English, fallback to first available
3. Fetch transcript with retry logic (max 2 retries)
4. Format transcript based on user options
5. Upload to S3: `s3://{bucket}/{user_id}/{job_id}/{video_id}.txt`
6. Call FastAPI callback: `/internal/job/{job_id}/video-complete` or `/video-failed`

**Lambda Error Handling:**
- Retries transcript fetch up to 2 times with exponential backoff
- Reports failures via `/internal/job/{job_id}/video-failed` callback
- Includes error message and error type in callback payload

**S3 Upload Details:**
- Content-Type: `text/plain`
- UTF-8 encoding
- Metadata includes: video_id, job_id, user_id, transcript_language, transcript_type

### Job Timeout Monitoring

**Timeout Task** (`youtube_service.monitor_job_timeout`):
- Started automatically after Lambda dispatch completes
- Default timeout: 10 minutes (configurable via `JOB_TIMEOUT_MINUTES`)
- Polls job status every 30 seconds

**Timeout Behavior:**
- If job is still "processing" after timeout period:
  - Marks all pending videos (status != "completed" and != "failed") as "timed_out"
  - Updates job status to "completed_with_errors"
  - Finalizes credits (refunds for videos not processed)
  - Sets `timeout_occurred=True` flag on job

**Late-Arriving Lambda Responses:**
- Lambda responses after timeout are still processed
- Counts are updated accurately (late completions increment completed count)
- Job status remains "completed_with_errors" (not reverted)
- This ensures accurate credit usage tracking

**Current Implementation Note:**
The timeout system is currently set up to still process late-arriving Lambda responses to maintain accurate counts. This can be disabled by uncommenting the early return in the `/internal/job/{job_id}/video-complete` callback handler if late responses should be ignored.

### Credit Management System

**Credit Reservation Pattern:**
```python
class CreditManager:
    # Phase 1: Upfront credit reservation (prevents overselling)
    @staticmethod
    async def reserve_credits(user_id: str, amount: int) -> str:
        # Returns reservation_id for tracking
    
    # Phase 2: Credit finalization with refunds for unused credits
    @staticmethod 
    async def finalize_credit_usage(
        user_id: str, 
        reservation_id: str, 
        credits_used: int, 
        credits_reserved: int
    ):
        # Refunds unused credits (credits_reserved - credits_used)
```

**Credit Flow:**
1. **Reservation** (at job creation):
   - Reserve total credits needed (1 per video)
   - Store reservation_id in job record
   - Prevents user from spending reserved credits elsewhere

2. **Usage Tracking** (per video completion):
   - Increment credits_used atomically in database
   - Each successful transcript = 1 credit
   - Failed transcripts also consume 1 credit (attempt was made)

3. **Finalization** (at job completion):
   - Calculate actual usage: `credits_used` from job record
   - Refund unused: `credits_reserved - credits_used`
   - Release reservation

**Database Operations:**
- All credit operations use PostgreSQL transactions
- Atomic updates via `hybrid_job_manager`
- Concurrent-safe with version field for optimistic locking

**Stripe Integration:**
```python
# Price tiers for credit packages
PRICE_CREDITS_MAP = {
    "price_1SKoPc9nwfLYxL59M1ht3uqP": 400,   # 400 credits
    "price_1SKoPc9nwfLYxL59pXxIYFXm": 1000,  # 1000 credits  
    "price_1SKoPc9nwfLYxL59RjL6qg2L": 3000,  # 3000 credits
}

# Webhook distinguishes services via metadata
metadata = {"project": "transcript-api", "user_uuid": user_id}
```

### Rate Limiting Strategy
- **Anonymous users**: IP-based rate limiting via `rate_limiter.py`
  - 3 downloads per hour per IP (configurable)
  - Applied to `/download/transcript/raw` endpoint only
- **Authenticated users**: Credit-based access (bypass rate limits)
- **Fallback**: When credits exhausted, users cannot download (must purchase credits)
- **Fallback**: When credits exhausted, users cannot download (must purchase credits)

## Integration Points (YouTube Service & Database)

### YouTube API Usage (`youtube_service.py`) - QUOTA OPTIMIZED
```python
# Channel/video ID extraction and validation
def extract_youtube_id(url: str) -> str
def extract_playlist_id(url_or_id: str) -> str

# Video metadata fetching with retry logic
async def get_video_info(video_id: str) -> Dict

# QUOTA EFFICIENT: Channel video discovery using uploads playlist
async def get_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]
async def get_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]
# Uses playlistItems.list (1 unit) + batch videos.list (1 unit per 50 videos)
# Replaces expensive search.list calls (100 units each)
# Achieves 99%+ quota reduction for video discovery

# Duration categorization with efficient batch processing
def _categorize_duration(duration_seconds: int) -> str:
    # Returns 'short' (≤60s), 'medium' (61s-20min), 'long' (>20min)

# Metadata pre-fetching (batch optimization)
async def pre_fetch_videos_metadata(video_ids: List[str], batch_size: int = 50) -> Dict[str, Dict[str, Any]]:
    # Fetches metadata for all videos in batches to minimize API calls
    # Returns dict mapping video_id to metadata

# Transcript extraction with proxy support
def get_ytt_api() -> YouTubeTranscriptApi:
    # Uses WebshareProxyConfig for rate limit bypass if credentials provided

# Channel/Playlist metadata fetching
async def get_channel_info(channel_name: str) -> Dict[str, Any]:
    # Uses channels.list (1 unit) for channel metadata
async def get_playlist_info(playlist_id: str) -> Dict[str, Any]:
    # Uses playlists.list (1 unit) for playlist metadata

# Lambda dispatch coordination
async def dispatch_lambdas_concurrently(
    job_id: str,
    videos: List[Any],
    videos_metadata: Dict[str, Dict[str, Any]],
    user_id: str,
    formatting_options: Dict[str, Any],
    max_concurrent: int = 20,
) -> int:
    # Fire-and-forget Lambda invocation using boto3
    # No semaphore limits - dispatches all videos simultaneously
```

### Hybrid Job Manager (`hybrid_job_manager.py`)

The hybrid job manager provides a unified interface for database operations with atomic updates:

```python
class HybridJobManager:
    # Job creation
    async def create_job(job_id: str, job_data: Dict, videos: List[Dict]) -> str
    
    # Job retrieval
    async def get_job(job_id: str) -> Optional[Dict]
    async def get_job_status(job_id: str) -> Optional[Dict]
    
    # Job updates (atomic operations)
    async def update_job(job_id: str, **updates) -> Optional[Dict]
    async def update_job_status_safe(job_id: str, new_status: str, expected_current_status: str) -> bool
    
    # Video-level operations (atomic)
    async def mark_video_completed(job_id: str, video_id: str, file_info: Dict) -> Optional[Dict]
    async def mark_video_failed(job_id: str, video_id: str, error_message: str) -> Optional[Dict]
    
    # Metadata updates
    async def update_videos_metadata(job_id: str, videos_metadata: Dict) -> None
```

**Key Features:**
- Atomic database operations with transaction support
- Optimistic locking via version field
- Automatic timestamp updates (updated_at)
- Handles Pydantic object serialization
- Fallback support for migration scenarios

### Database Operations
```python
# Dedicated database connection for transcript service
from db_youtube_transcripts.database import get_db_youtube_transcripts, init_db_pool, close_db_pool

# Credit management operations
class CreditManager:
    @staticmethod
    async def get_user_credits(user_id: str) -> int
    
    @staticmethod
    async def reserve_credits(user_id: str, amount: int) -> str
    
    @staticmethod
    async def finalize_credit_usage(
        user_id: str, 
        reservation_id: str, 
        credits_used: int, 
        credits_reserved: int
    ) -> None
    
    @staticmethod
    def create_user_credits(user_id: str, initial_credits: int = 0) -> None
```

## Common Debugging & Maintenance

### Transcript API Debugging
```bash
# Check service status
curl http://localhost:8001/

# Test channel functionality
curl http://localhost:8001/channel/channelname
curl http://localhost:8001/channel/channelname/all-videos

# Test playlist functionality
curl http://localhost:8001/playlist/PLxxxxxx
curl http://localhost:8001/playlist/PLxxxxxx/all-videos

# Database connection test
python db_youtube_transcripts/database.py

# YouTube API quota monitoring
# Check console logs for rate limit errors
```

### Lambda Debugging
```bash
# Test Lambda locally (from lambda-transcript-processor directory)
python test_local.py

# Check Lambda logs in AWS CloudWatch
aws logs tail /aws/lambda/transcript-processor --follow

# Monitor S3 uploads
aws s3 ls s3://your-bucket/{user_id}/{job_id}/
```

### Common Issues

**Job stuck in "processing" status:**
- Check Lambda CloudWatch logs for errors
- Verify S3 bucket permissions
- Check if timeout monitoring is working (default 10 minutes)
- Verify API_BASE_URL is accessible from Lambda

**Credit not refunded after failures:**
- Check if job completed (status: "completed" or "completed_with_errors")
- Verify reservation_id is present in job record
- Check CreditManager.finalize_credit_usage calls in logs

**Playlist errors:**
- 404: Playlist not found or private
- 422: Invalid playlist_name in request body
- Verify playlist is public or unlisted

## Key Configuration Files

- `src/config_v2.py`: Pydantic settings for transcript service
- `db_youtube_transcripts/database.py`: Database connection and pool management
- `db_youtube_transcripts/schema.py`: Database schema definitions
- `src/rate_limiter.py`: Anonymous user rate limiting
- `src/youtube_service.py`: Core YouTube interaction logic
- `src/hybrid_job_manager.py`: Database operations wrapper
- `lambda-transcript-processor/lambda_function.py`: Lambda worker code