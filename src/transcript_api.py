#!/usr/bin/env python3
"""
YouTube Transcript API - A dedicated API for downloading YouTube transcripts
"""
import os
import logging
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Optional
import io
import zipfile
import uuid
import asyncio
import tempfile
import time
import tracemalloc
import gc
import psutil
import json
from datetime import datetime
import glob

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Depends,
    status,
    BackgroundTasks,
    Response,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

# Authentication and Stripe imports
from jose import JWTError, jwt
import stripe

# Database imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db_youtube_transcripts"))
# from db_youtube_transcripts.database import get_db_youtube_transcripts

import youtube_service
from rate_limiter import transcript_limiter

# Using the Pydantic v2 compatible settings
from config_v2 import settings

# Import hybrid job manager for database operations
from hybrid_job_manager import hybrid_job_manager


# Stripe configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY_LIVE")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET_TRANSCRIPTS")
stripe.api_key = STRIPE_SECRET_KEY

# JWT Configuration for Supabase
ALGORITHM = "HS256"
SUPABASE_SECRET = os.getenv("SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS")
FRONTEND_URL = os.getenv("FRONTEND_URL_YOUTUBE_TRANSCRIPTS", "http://localhost:3000")

# Price ID mapping for Stripe products
PRICE_CREDITS_MAP = {
    "price_1SKoPc9nwfLYxL59M1ht3uqP": 400,  # 400 credits
    "price_1SKoPc9nwfLYxL59pXxIYFXm": 1000,  # 1000 credits
    "price_1SKoPc9nwfLYxL59RjL6qg2L": 3000,  # 3000 credits
}

# Security setup
security = HTTPBearer(auto_error=False)

# Initialize FastAPI
app = FastAPI(
    title=settings.api_title,
    description="API for downloading YouTube video transcripts",
    version=settings.api_version,
)

# CORS configuration from settings
origins = settings.cors_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache for user sessions
# In production, consider a more robust solution like Redis
user_cache = {}

# Video jobs persistent storage directory
VIDEO_JOBS_STORAGE_DIR = os.path.join(settings.temp_dir, "jobs")
os.makedirs(VIDEO_JOBS_STORAGE_DIR, exist_ok=True)

# Get API key from settings
API_KEY = settings.api_key

# =============================================
# AUTHENTICATION AND CREDIT MANAGEMENT
# =============================================


async def validate_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Validates Supabase JWT token and returns user payload
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT token is missing"
        )

    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT token is missing"
        )

    try:
        payload = jwt.decode(
            token,
            SUPABASE_SECRET,
            algorithms=[ALGORITHM],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


def get_user_id_from_payload(payload: dict) -> str:
    """Extract user ID from JWT payload"""
    return payload.get("sub")


class CreditManager:
    """Manages user credits for transcript downloads - FULLY ASYNC"""

    @staticmethod
    async def get_user_credits(user_id: str) -> int:
        """Get current credit balance for user - ASYNC"""
        try:
            logger.debug(f"Getting credits for user: {user_id}")
            from db_youtube_transcripts.database import get_db_connection

            async with get_db_connection() as conn:
                result = await conn.fetchrow(
                    "SELECT credits FROM user_credits WHERE user_id = $1", user_id
                )

                if result:
                    logger.debug(f"User {user_id} has {result['credits']} credits")
                    return result["credits"]
                else:
                    logger.info(
                        f"User {user_id} not found in credits table, creating with 0 credits"
                    )
                    # Create user with 0 credits if doesn't exist
                    await CreditManager.create_user_credits(user_id, 0)
                    return 0

        except Exception as e:
            logger.error(
                f"Error getting credits for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve user credits",
            )

    @staticmethod
    async def create_user_credits(user_id: str, credits: int = 0) -> None:
        """Create user credits record - ASYNC"""
        try:
            logger.debug(
                f"Creating credits record for user: {user_id} with {credits} credits"
            )
            from db_youtube_transcripts.database import get_db_connection

            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_credits (user_id, credits) 
                    VALUES ($1, $2) 
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    user_id,
                    credits,
                )

            logger.info(
                f"Created credits record for user {user_id} with {credits} credits"
            )

        except Exception as e:
            logger.error(
                f"Error creating credits for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user credits",
            )

    @staticmethod
    async def deduct_credit(user_id: str) -> bool:
        """
        Deduct 1 credit from user - ASYNC
        Returns True if successful, False if insufficient credits.
        """
        try:
            logger.debug(f"Attempting to deduct credit for user: {user_id}")
            from db_youtube_transcripts.database import get_db_connection

            async with get_db_connection() as conn:
                # Deduct credit only if user has credits available
                result = await conn.execute(
                    "UPDATE user_credits SET credits = credits - 1 WHERE user_id = $1 AND credits > 0",
                    user_id,
                )

                # PostgreSQL returns "UPDATE N" where N is number of rows affected
                success = result == "UPDATE 1"

                if success:
                    logger.info(f"Deducted 1 credit from user {user_id}")
                else:
                    logger.warning(
                        f"Failed to deduct credit from user {user_id} - insufficient credits"
                    )

                return success

        except Exception as e:
            logger.error(
                f"Error deducting credit for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process credit deduction",
            )

    @staticmethod
    async def add_credits(user_id: str, credits: int) -> None:
        """Add credits to user account - ASYNC"""
        try:
            logger.debug(f"Adding {credits} credits to user: {user_id}")
            from db_youtube_transcripts.database import get_db_connection

            async with get_db_connection() as conn:
                # Insert or update user credits
                await conn.execute(
                    """
                    INSERT INTO user_credits (user_id, credits) 
                    VALUES ($1, $2)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET credits = user_credits.credits + $2
                    """,
                    user_id,
                    credits,
                )

            logger.info(f"Added {credits} credits to user {user_id}")

        except Exception as e:
            logger.error(
                f"Error adding credits for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add credits",
            )

    @staticmethod
    async def reserve_credits(user_id: str, credit_count: int) -> str:
        """
        Reserve credits for a batch operation - ASYNC with atomic transaction
        Returns reservation ID if successful.

        Args:
            user_id: User identifier
            credit_count: Number of credits to reserve

        Returns:
            Reservation ID string for tracking

        Raises:
            HTTPException: If insufficient credits or database error
        """
        import uuid

        try:
            logger.info(
                f"Attempting to reserve {credit_count} credits for user: {user_id}"
            )
            from db_youtube_transcripts.database import get_db_transaction

            async with get_db_transaction() as conn:
                # Check and deduct credits in single atomic operation
                result = await conn.fetchrow(
                    """
                    UPDATE user_credits 
                    SET credits = credits - $1 
                    WHERE user_id = $2 AND credits >= $1
                    RETURNING credits
                    """,
                    credit_count,
                    user_id,
                )

                if not result:
                    # Either user doesn't exist or insufficient credits
                    current = await conn.fetchrow(
                        "SELECT credits FROM user_credits WHERE user_id = $1", user_id
                    )

                    if not current:
                        logger.warning(f"User {user_id} not found in credits table")
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail="User credits not found",
                        )

                    logger.warning(
                        f"User {user_id} has insufficient credits: {current['credits']} < {credit_count}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail=f"Insufficient credits. Need {credit_count}, have {current['credits']}",
                    )

                # Generate reservation ID for tracking
                reservation_id = str(uuid.uuid4())

                logger.info(
                    f"Reserved {credit_count} credits for user {user_id}, reservation: {reservation_id}"
                )

                return reservation_id

        except HTTPException:
            # Re-raise HTTP exceptions from credit checks
            raise
        except Exception as e:
            logger.error(
                f"Error reserving credits for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reserve credits",
            )

    @staticmethod
    async def finalize_credit_usage(
        user_id: str, reservation_id: str, credits_used: int, credits_reserved: int
    ) -> None:
        """
        Finalize credit usage for a reservation - ASYNC
        Refunds unused credits.

        Args:
            user_id: User identifier
            reservation_id: Reservation ID from reserve_credits
            credits_used: Actual credits used
            credits_reserved: Credits that were reserved
        """
        try:
            unused_credits = credits_reserved - credits_used

            if unused_credits > 0:
                logger.info(
                    f"Refunding {unused_credits} unused credits to user {user_id} "
                    f"(reservation: {reservation_id})"
                )

                from db_youtube_transcripts.database import get_db_connection

                async with get_db_connection() as conn:
                    # Refund unused credits
                    await conn.execute(
                        "UPDATE user_credits SET credits = credits + $1 WHERE user_id = $2",
                        unused_credits,
                        user_id,
                    )

                logger.info(
                    f"Successfully refunded {unused_credits} credits to user {user_id}"
                )
            else:
                logger.info(
                    f"No refund needed for user {user_id}, used {credits_used}/{credits_reserved} credits"
                )

        except Exception as e:
            logger.error(
                f"Error finalizing credit usage for user {user_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Don't raise exception here - we don't want to fail the job completion
            # but log the error for investigation
            logger.error("Credit finalization failed but job will complete normally")


def get_user_or_anonymous(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Get user info if authenticated, otherwise return anonymous user info

    NOTE: This is a sync function but calls async CreditManager.
    For proper async usage, use validate_jwt directly and call CreditManager separately.
    """
    if credentials:
        try:
            payload = jwt.decode(
                credentials.credentials,
                SUPABASE_SECRET,
                algorithms=[ALGORITHM],
                options={"verify_aud": False},
            )
            user_id = get_user_id_from_payload(payload)
            # NOTE: Calling async function from sync context - not ideal
            # This works but frontend should use authenticated endpoints instead
            ## REMOVE AND SEE LATER HOW TO HANDLE CREDITS, NOT CRITICAL BECAUSE USERS HAVE LOTS OF CREDITS
            # import asyncio

            # try:
            #     credits = asyncio.run(CreditManager.get_user_credits(user_id))
            # except RuntimeError:
            #     # If event loop is already running, return 0 credits
            #     # Frontend should call /user/credits endpoint for accurate balance
            #     credits = 0
            return {"is_authenticated": True, "user_id": user_id, "credits": credits}
        except JWTError:
            # Invalid token, treat as anonymous
            pass

    return {"is_authenticated": False, "user_id": None, "credits": 0}


# =============================================
# PYDANTIC MODELS
# =============================================


class DownloadURLRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL or ID")
    include_timestamps: bool = Field(
        True, description="Whether to include timestamps in the transcript"
    )


class ChannelRequest(BaseModel):
    channel_name: str = Field(..., description="YouTube channel name or ID")
    max_results: int = Field(30, description="Maximum number of videos to fetch")
    # Formatting options
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in transcript"
    )
    include_video_title: bool = Field(
        default=True, description="Include video title in header"
    )
    include_video_id: bool = Field(
        default=True, description="Include video ID in header"
    )
    include_video_url: bool = Field(
        default=True, description="Include video URL in header"
    )
    include_view_count: bool = Field(
        default=False, description="Include video view count in header"
    )
    concatenate_all: bool = Field(
        default=False,
        description="Return single concatenated file instead of individual files",
    )

    @validator("max_results")
    def validate_max_results(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("max_results must be between 1 and 100")
        return v


class CheckoutSessionRequest(BaseModel):
    price_id: str = Field(..., description="Stripe price ID for the credit package")

    @validator("price_id")
    def validate_price_id(cls, v):
        if v not in PRICE_CREDITS_MAP:
            raise ValueError(
                f"Invalid price_id. Valid options: {list(PRICE_CREDITS_MAP.keys())}"
            )
        return v


class UserCreditsResponse(BaseModel):
    user_id: str
    credits: int


class PaymentSuccessResponse(BaseModel):
    message: str
    credits_added: int
    total_credits: int


class VideoInfo(BaseModel):
    id: str
    title: str
    publishedAt: str = None
    duration: str = None
    url: str


class SelectedVideosRequest(BaseModel):
    channel_name: str = Field(..., description="YouTube channel name or ID")
    videos: List[VideoInfo] = Field(
        ..., description="List of selected videos to download transcripts for"
    )
    # Formatting options
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in transcript"
    )
    include_video_title: bool = Field(
        default=True, description="Include video title in header"
    )
    include_video_id: bool = Field(
        default=True, description="Include video ID in header"
    )
    include_video_url: bool = Field(
        default=True, description="Include video URL in header"
    )
    include_view_count: bool = Field(
        default=False, description="Include video view count in header"
    )
    concatenate_all: bool = Field(
        default=False,
        description="Return single concatenated file instead of individual files",
    )


class PlaylistRequest(BaseModel):
    playlist_name: str = Field(..., description="YouTube playlist ID or URL")
    max_results: int = Field(30, description="Maximum number of videos to fetch")
    # Formatting options
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in transcript"
    )
    include_video_title: bool = Field(
        default=True, description="Include video title in header"
    )
    include_video_id: bool = Field(
        default=True, description="Include video ID in header"
    )
    include_video_url: bool = Field(
        default=True, description="Include video URL in header"
    )
    include_view_count: bool = Field(
        default=False, description="Include video view count in header"
    )
    concatenate_all: bool = Field(
        default=False,
        description="Return single concatenated file instead of individual files",
    )

    @validator("max_results")
    def validate_max_results(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("max_results must be between 1 and 100")
        return v


class SelectedPlaylistVideosRequest(BaseModel):
    playlist_name: str = Field(..., description="YouTube playlist ID or URL")
    videos: List[VideoInfo] = Field(
        ..., description="List of selected videos to download transcripts for"
    )
    # Formatting options
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in transcript"
    )
    include_video_title: bool = Field(
        default=True, description="Include video title in header"
    )
    include_video_id: bool = Field(
        default=True, description="Include video ID in header"
    )
    include_video_url: bool = Field(
        default=True, description="Include video URL in header"
    )
    include_view_count: bool = Field(
        default=False, description="Include video view count in header"
    )
    concatenate_all: bool = Field(
        default=False,
        description="Return single concatenated file instead of individual files",
    )


class DownloadHistoryItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the download")
    date: str = Field(..., description="ISO date string when download was initiated")
    sourceName: str = Field(..., description="Name of the channel or playlist")
    sourceType: str = Field(..., description="Type of source: 'channel' or 'playlist'")
    videoCount: int = Field(..., description="Total number of videos in the download")
    status: str = Field(
        ..., description="Download status: 'completed', 'processing', 'failed'"
    )
    downloadUrl: Optional[str] = Field(
        None, description="URL to download the transcript files if available"
    )
    jobId: str = Field(..., description="Job ID for tracking the download")
    createdAt: str = Field(..., description="ISO timestamp when download was created")
    completedAt: Optional[str] = Field(
        None, description="ISO timestamp when download was completed"
    )
    # Additional useful fields for S3-based storage
    successfulFiles: int = Field(
        default=0, description="Number of successfully downloaded transcript files"
    )
    failedFiles: int = Field(default=0, description="Number of failed video downloads")
    successRate: float = Field(
        default=0.0, description="Success rate as percentage (0-100)"
    )
    creditsUsed: int = Field(
        default=0, description="Credits consumed for this download"
    )


def verify_api_key(request: Request):
    """Verify API key for authenticated endpoints"""
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True


async def get_user_download_history(user_id: str) -> List[DownloadHistoryItem]:
    """
    Get download history for a specific user from the database.
    Returns completed, failed, and partially completed jobs ordered by creation date (newest first).
    """
    history_items = []

    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            # Query jobs table for all completed/failed jobs for this user
            query = """
                SELECT 
                    job_id,
                    status,
                    source_type,
                    source_name,
                    total_videos,
                    completed,
                    failed_count,
                    credits_used,
                    created_at,
                    start_time,
                    end_time
                FROM jobs
                WHERE user_id = $1
                    AND status IN ('completed', 'completed_with_errors','processing', 'failed')
                ORDER BY created_at DESC
            """

            rows = await conn.fetch(query, user_id)

            for row in rows:
                # Determine display status
                display_status = row["status"]
                if display_status == "completed_with_errors":
                    display_status = "completed"

                # Calculate success metrics
                total_videos = row["total_videos"] or 0
                successful_files = row["completed"] or 0
                failed_count = row["failed_count"] or 0
                credits_used = row["credits_used"] or 0

                success_rate = (
                    (successful_files / total_videos * 100) if total_videos > 0 else 0
                )

                # Determine download URL availability
                download_url = None
                if (
                    row["status"] in ["completed", "completed_with_errors"]
                    and successful_files > 0
                ):
                    download_url = f"/channel/download/results/{row['job_id']}"

                # Convert timestamps to ISO strings
                created_at_iso = (
                    row["created_at"].isoformat() if row["created_at"] else ""
                )
                start_time_iso = (
                    row["start_time"].isoformat()
                    if row["start_time"]
                    else created_at_iso
                )
                end_time_iso = row["end_time"].isoformat() if row["end_time"] else None

                # Map database row to DownloadHistoryItem
                history_item = DownloadHistoryItem(
                    id=str(row["job_id"]),
                    date=start_time_iso,
                    sourceName=row["source_name"] or "Unknown",
                    sourceType=row["source_type"] or "channel",
                    videoCount=total_videos,
                    status=display_status,
                    downloadUrl=download_url,
                    jobId=str(row["job_id"]),
                    createdAt=created_at_iso,
                    completedAt=end_time_iso,
                    successfulFiles=successful_files,
                    failedFiles=failed_count,
                    successRate=round(success_rate, 1),
                    creditsUsed=credits_used,
                )

                history_items.append(history_item)

        logger.info(
            f"Retrieved {len(history_items)} download history items from database for user {user_id}"
        )
        return history_items

    except Exception as e:
        logger.error(
            f"Error retrieving download history from database for user {user_id}: {e}"
        )
        # Return empty list on error rather than raising exception
        return []


def check_anonymous_rate_limit(request: Request):
    """
    Check rate limit for anonymous users

    Anonymous users are identified by their IP address and are limited
    to 3 transcript downloads per hour.
    """
    # Extract client IP (handle proxy forwarding)
    client_ip = request.client.host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Use the first IP in the chain if forwarded
        client_ip = forwarded.split(",")[0].strip()

    # Check if rate limit is exceeded
    free_limit = 100  # 3 downloads per hour

    if not transcript_limiter.can_make_request(client_ip, free_limit):
        # Calculate remaining time
        wait_time = transcript_limiter.get_wait_time(client_ip)
        minutes = int(wait_time // 60)
        seconds = int(wait_time % 60)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {minutes} minutes and {seconds} seconds.",
            headers={
                "X-RateLimit-Limit": str(free_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(
                    int(transcript_limiter.get_reset_time(client_ip))
                ),
                "Retry-After": str(int(wait_time)),
            },
        )

    # Add this request to the counter
    transcript_limiter.add_request(client_ip)

    # Add rate limit headers
    remaining = transcript_limiter.get_remaining_requests(client_ip, free_limit)

    # Return IP and remaining requests for logging
    return {
        "client_ip": client_ip,
        "remaining_requests": remaining,
    }


def get_user_session(request: Request):
    """Simple session management - replace with proper auth in production"""
    session_id = request.headers.get("X-Session-ID", f"anonymous-{request.client.host}")
    if session_id not in user_cache:
        user_cache[session_id] = {"id": session_id, "transcript_retrievals": {}}
    return user_cache[session_id]


@app.get("/")
def read_root():
    return {
        "service": "YouTube Transcript API",
        "status": "online",
        "docs_url": "/docs",
    }


@app.on_event("startup")
async def startup_event():
    """Run when the application starts - initialize database connection pool"""
    logger.info("Starting YouTube Transcript API...")

    try:
        # Initialize async database connection pool
        from db_youtube_transcripts.database import init_db_pool

        await init_db_pool()
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}", exc_info=True)
        # Don't raise - allow app to start even if DB pool init fails
        # Individual requests will retry connection


#     # Recover jobs from persistent storage
#     try:
#         import youtube_service

#         recovered_jobs = youtube_service.recover_jobs_from_storage()
#         if recovered_jobs:
#             logger.info(f"Recovered {len(recovered_jobs)} jobs from persistent storage")
#         else:
#             logger.info("No jobs to recover from persistent storage")
#     except Exception as e:
#         logger.error(f"Failed to recover jobs on startup: {e}")

#     logger.info("YouTube Transcript API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on FastAPI shutdown"""
    logger.info("Shutting down YouTube Transcript API...")

    try:
        # Close async database connection pool
        from db_youtube_transcripts.database import close_db_pool

        await close_db_pool()
        logger.info("Database connection pool closed")

        # Note: We could add a cleanup_resources function to youtube_service if needed
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")

    logger.info("YouTube Transcript API shutdown complete")


# =============================================
# PAYMENT ENDPOINTS
# =============================================


@app.post("/payments/create-checkout-session")
async def create_checkout_session(
    request: CheckoutSessionRequest, payload: dict = Depends(validate_jwt)
):
    """
    Create a Stripe checkout session for purchasing credits
    """
    try:
        user_id = get_user_id_from_payload(payload)
        credits_to_add = PRICE_CREDITS_MAP[request.price_id]

        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": request.price_id,
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=f"{FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/payment/cancel",
            client_reference_id=user_id,
            metadata={
                "project": "transcript-api",
                "user_id": user_id,
                "credits": credits_to_add,
            },
        )

        logger.info(
            f"Created checkout session {checkout_session.id} for user {user_id}"
        )

        return {
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id,
            "credits": credits_to_add,
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment processing error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment session",
        )


@app.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events for payment completion
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        # Check if this is for our transcript-api project
        if session.get("metadata", {}).get("project") != "transcript-api":
            logger.info(
                f"Ignoring webhook for different project: {session.get('metadata', {}).get('project')}"
            )
            return {"status": "ignored"}

        try:
            user_id = session["metadata"]["user_id"]
            credits_to_add = int(session["metadata"]["credits"])

            # Add credits to user account - NOW ASYNC
            await CreditManager.add_credits(user_id, credits_to_add)

            logger.info(
                f"Successfully added {credits_to_add} credits to user {user_id} from session {session['id']}"
            )

            return {"status": "success"}

        except Exception as e:
            logger.error(f"Error processing payment completion: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process payment",
            )

    else:
        logger.info(f"Unhandled event type: {event['type']}")
        return {"status": "unhandled"}


@app.get("/user/credits")
async def get_user_credits(
    payload: dict = Depends(validate_jwt),
) -> UserCreditsResponse:
    """
    Get current credit balance for authenticated user
    """
    try:
        logger.debug("Starting get_user_credits endpoint")
        user_id = get_user_id_from_payload(payload)
        logger.debug(f"Extracted user_id: {user_id}")

        credits = await CreditManager.get_user_credits(user_id)
        logger.debug(f"Retrieved credits: {credits}")

        response = UserCreditsResponse(user_id=user_id, credits=credits)
        logger.debug(f"Created response: {response}")
        return response

    except HTTPException:
        # Re-raise HTTPExceptions from CreditManager
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in get_user_credits endpoint: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve credits",
        )


@app.get("/user/profile")
async def get_user_profile(payload: dict = Depends(validate_jwt)):
    """
    Get user profile information
    """
    try:
        user_id = get_user_id_from_payload(payload)
        credits = await CreditManager.get_user_credits(user_id)

        return {
            "user_id": user_id,
            "credits": credits,
            "email": payload.get("email"),
            "created_at": payload.get("iat"),
        }

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile",
        )


@app.get("/user/download-history", response_model=List[DownloadHistoryItem])
async def get_user_download_history_endpoint(payload: dict = Depends(validate_jwt)):
    """
    Get user's download history from database
    """
    try:
        user_id = get_user_id_from_payload(payload)
        history_items = await get_user_download_history(user_id)

        logger.info(f"Retrieved {len(history_items)} history items for user {user_id}")
        return history_items

    except Exception as e:
        logger.error(f"Error getting download history for user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve download history",
        )


# =============================================
# VIDEO JOBS PERSISTENT STORAGE
# =============================================


def save_video_job_to_file(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save video job data to persistent storage"""
    try:
        file_path = os.path.join(VIDEO_JOBS_STORAGE_DIR, f"{job_id}.json")

        # Convert data to JSON serializable format
        serializable_data = {}
        for key, value in job_data.items():
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_data[key] = value
            else:
                # Convert other types to string representation
                serializable_data[key] = str(value)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved video job {job_id} to file")

    except Exception as e:
        logger.error(f"Failed to save video job {job_id} to file: {e}")


def load_video_job_from_file(job_id: str) -> Optional[Dict[str, Any]]:
    """Load video job data from persistent storage"""
    try:
        file_path = os.path.join(VIDEO_JOBS_STORAGE_DIR, f"{job_id}.json")

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        logger.debug(f"Loaded video job {job_id} from file")
        return job_data

    except Exception as e:
        logger.error(f"Failed to load video job {job_id} from file: {e}")
        return None


def update_video_job(job_id: str, **updates) -> Optional[Dict[str, Any]]:
    """Update video job data with atomic file operations"""
    try:
        # Load current job data
        job_data = load_video_job_from_file(job_id)
        if job_data is None:
            logger.error(f"Video job {job_id} not found for update")
            return None

        # Apply updates
        for key, value in updates.items():
            job_data[key] = value

        # Save updated data
        save_video_job_to_file(job_id, job_data)

        return job_data

    except Exception as e:
        logger.error(f"Failed to update video job {job_id}: {e}")
        return None


# =============================================
# BACKGROUND TASKS FOR VIDEO FETCHING
# =============================================


async def fetch_channel_videos_task(job_id: str, channel_name: str):
    """
    Background task to fetch all videos from a channel with timeout protection.
    Updates job status in video_jobs dictionary.
    """
    try:
        logger.info(
            f"Starting background fetch for channel {channel_name} (job: {job_id})"
        )

        # Fetch with 5-minute timeout
        timeout_seconds = 300  # 5 minutes

        async def _fetch_videos():
            # Get channel info
            channel_info = await youtube_service.get_channel_info(channel_name)
            channel_id = channel_info["channelId"]

            # Get ALL videos from channel (no limit)
            videos = await youtube_service.get_all_channel_videos(channel_id)

            return channel_info, videos

        # Execute with timeout
        try:
            channel_info, videos = await asyncio.wait_for(
                _fetch_videos(), timeout=timeout_seconds
            )

            # Update job with success
            update_video_job(
                job_id,
                status="completed",
                channel_info=channel_info,
                videos=videos,
                video_count=len(videos),
                duration_breakdown={
                    "short": len([v for v in videos if v.get("duration") == "short"]),
                    "medium": len([v for v in videos if v.get("duration") == "medium"]),
                    "long": len([v for v in videos if v.get("duration") == "long"]),
                },
                end_time=time.time(),
            )

            logger.info(
                f"Successfully fetched {len(videos)} videos for channel {channel_name} (job: {job_id})"
            )

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {timeout_seconds} seconds"
            update_video_job(
                job_id, status="failed", error=error_msg, end_time=time.time()
            )
            logger.error(
                f"Timeout fetching videos for channel {channel_name} (job: {job_id}): {error_msg}"
            )

    except Exception as e:
        error_msg = f"Error fetching videos: {str(e)}"
        update_video_job(job_id, status="failed", error=error_msg, end_time=time.time())
        logger.error(
            f"Error in background task for channel {channel_name} (job: {job_id}): {error_msg}",
            exc_info=True,
        )


async def fetch_playlist_videos_task(job_id: str, playlist_id: str):
    """
    Background task to fetch all videos from a playlist with timeout protection.
    Updates job status in video_jobs dictionary.
    """
    try:
        logger.info(
            f"Starting background fetch for playlist {playlist_id} (job: {job_id})"
        )

        # Fetch with 5-minute timeout
        timeout_seconds = 300  # 5 minutes

        async def _fetch_videos():
            # Get playlist info
            playlist_info = await youtube_service.get_playlist_info(playlist_id)

            # Get ALL videos from playlist (no limit)
            videos = await youtube_service.get_all_playlist_videos(playlist_id)

            return playlist_info, videos

        # Execute with timeout
        try:
            playlist_info, videos = await asyncio.wait_for(
                _fetch_videos(), timeout=timeout_seconds
            )

            # Update job with success
            update_video_job(
                job_id,
                status="completed",
                playlist_info=playlist_info,
                videos=videos,
                video_count=len(videos),
                duration_breakdown={
                    "short": len([v for v in videos if v.get("duration") == "short"]),
                    "medium": len([v for v in videos if v.get("duration") == "medium"]),
                    "long": len([v for v in videos if v.get("duration") == "long"]),
                },
                end_time=time.time(),
            )

            logger.info(
                f"Successfully fetched {len(videos)} videos for playlist {playlist_id} (job: {job_id})"
            )

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {timeout_seconds} seconds"
            update_video_job(
                job_id, status="failed", error=error_msg, end_time=time.time()
            )
            logger.error(
                f"Timeout fetching videos for playlist {playlist_id} (job: {job_id}): {error_msg}"
            )

    except Exception as e:
        error_msg = f"Error fetching videos: {str(e)}"
        update_video_job(job_id, status="failed", error=error_msg, end_time=time.time())
        logger.error(
            f"Error in background task for playlist {playlist_id} (job: {job_id}): {error_msg}",
            exc_info=True,
        )


# =============================================
# TRANSCRIPT ENDPOINTS (UPDATED WITH CREDIT LOGIC)
# =============================================


# @app.post("/download/transcript")
# async def download_transcript_by_url(
#     request: DownloadURLRequest,  # From JSON body: {"youtube_url": "...", "include_timestamps": false}
#     fastapi_request: Request,  # From HTTP metadata: client IP, headers, etc.
#     user_info: Dict = Depends(get_user_or_anonymous),
#     session: Dict = Depends(get_user_session),
# ):
#     """
#     Download a transcript for a single YouTube video by URL.
#     Returns a plain text file with the transcript.

#     For authenticated users: deducts 1 credit per attempt
#     For anonymous users: rate limited to 3 downloads per hour per IP
#     """
#     endpoint_start = time.time()

#     # Handle authentication and credit/rate limit logic
#     if user_info["is_authenticated"]:
#         user_id = user_info["user_id"]

#         # Deduct credit before attempting download
#         if not CreditManager.deduct_credit(user_id):
#             raise HTTPException(
#                 status_code=status.HTTP_402_PAYMENT_REQUIRED,
#                 detail="Insufficient credits. Please purchase more credits to continue.",
#             )

#         logger.info(f"Authenticated transcript download for user {user_id}")
#     else:
#         # Apply rate limiting for anonymous users
#         rate_limit_info = check_anonymous_rate_limit(fastapi_request)
#         logger.info(
#             f"Anonymous transcript download from {rate_limit_info['client_ip']} "
#             f"({rate_limit_info['remaining_requests']} requests remaining)"
#         )
#     try:
#         # Extract YouTube video ID from URL
#         extract_start = time.time()
#         video_id = youtube_service.extract_youtube_id(request.youtube_url)
#         extract_end = time.time()
#         logger.info(f"URL extraction took {extract_end - extract_start:.3f}s")

#         # Create directories for this user
#         dir_start = time.time()
#         user_dir = os.path.join(settings.temp_dir, session["id"])
#         os.makedirs(user_dir, exist_ok=True)
#         dir_end = time.time()
#         logger.info(f"Directory creation took {dir_end - dir_start:.3f}s")

#         # Get transcript using the new service
#         transcript_start = time.time()
#         transcript_text, file_path, metadata = (
#             await youtube_service.get_single_transcript(
#                 video_id, user_dir, request.include_timestamps
#             )
#         )
#         transcript_end = time.time()
#         logger.info(
#             f"Transcript retrieval took {transcript_end - transcript_start:.3f}s"
#         )

#         if not file_path:
#             # If file_path is None, create a temporary file
#             tmp_start = time.time()
#             with tempfile.NamedTemporaryFile(
#                 mode="w+", delete=False, suffix=".txt", dir=user_dir
#             ) as temp_file:
#                 temp_file.write(transcript_text)
#                 file_path = temp_file.name
#             tmp_end = time.time()
#             logger.info(f"Temporary file creation took {tmp_end - tmp_start:.3f}s")

#         response_start = time.time()
#         response = FileResponse(
#             path=file_path,
#             media_type="text/plain",
#             filename=f"{video_id}_transcript.txt",
#         )
#         response.headers["X-Transcript-Language"] = metadata.get(
#             "transcript_language", "unknown"
#         )
#         response.headers["X-Transcript-Type"] = metadata.get(
#             "transcript_type", "unknown"
#         )

#         response_end = time.time()
#         logger.info(f"Response preparation took {response_end - response_start:.3f}s")

#         endpoint_end = time.time()
#         total_time = endpoint_end - endpoint_start
#         logger.info(f"Total endpoint execution took {total_time:.3f}s")

#         return response

#     except ValueError as e:
#         logger.error(f"Invalid URL: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error downloading transcript: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail=f"Failed to download transcript: {str(e)}"
#         )


@app.post("/download/transcript/raw")
async def download_transcript_raw(
    request: DownloadURLRequest,  # From JSON body: {"youtube_url": "...", "include_timestamps": false}
    fastapi_request: Request,  # From HTTP metadata: client IP, headers, etc.
    user_info: Dict = Depends(get_user_or_anonymous),
    session: Dict = Depends(get_user_session),
):
    """
    Get a transcript for a single YouTube video as plain text without file creation.
    This is a faster alternative to /download/transcript when you just need the text.

    Returns the transcript text directly without creating intermediate files.

    For authenticated users: deducts 1 credit per attempt
    For anonymous users: rate limited downloads
    """
    endpoint_start = time.time()

    # Handle authentication and credit/rate limit logic
    if user_info["is_authenticated"]:
        user_id = user_info["user_id"]

        # Deduct credit before attempting download - NOW ASYNC
        if not await CreditManager.deduct_credit(user_id):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Insufficient credits. Please purchase more credits to continue.",
            )

        logger.info(f"Authenticated raw transcript download for user {user_id}")
    else:
        # Apply rate limiting for anonymous users
        rate_limit_info = check_anonymous_rate_limit(fastapi_request)
        logger.info(
            f"Anonymous raw transcript download from {rate_limit_info['client_ip']} "
            f"({rate_limit_info['remaining_requests']} requests remaining)"
        )
    try:
        # Extract YouTube video ID from URL
        extract_start = time.time()
        video_id = youtube_service.extract_youtube_id(request.youtube_url)
        extract_end = time.time()
        logger.info(f"URL extraction took {extract_end - extract_start:.3f}s")

        # Get transcript with timeout protection
        transcript_start = time.time()
        try:
            # Create a task and set a timeout
            transcript_task = asyncio.create_task(
                youtube_service.get_single_transcript(
                    video_id,
                    output_dir=None,
                    include_timestamps=request.include_timestamps,
                )
            )
            transcript_text, _, metadata = await asyncio.wait_for(
                transcript_task, timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Transcript retrieval timed out for {video_id} after 30 seconds"
            )
            raise HTTPException(
                status_code=504,
                detail="Transcript retrieval timed out. Please try again or try another video.",
            )

        transcript_end = time.time()
        logger.info(
            f"Raw transcript retrieval took {transcript_end - transcript_start:.3f}s"
        )

        response_start = time.time()
        response = PlainTextResponse(content=transcript_text)
        response.headers["X-Transcript-Language"] = metadata.get(
            "transcript_language", "unknown"
        )
        response.headers["X-Transcript-Type"] = metadata.get(
            "transcript_type", "unknown"
        )
        response_end = time.time()
        logger.info(
            f"Raw response preparation took {response_end - response_start:.3f}s"
        )

        endpoint_end = time.time()
        total_time = endpoint_end - endpoint_start
        logger.info(f"Total raw endpoint execution took {total_time:.3f}s")

        return response

    except ValueError as e:
        logger.error(f"Invalid URL: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading transcript: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to download transcript: {str(e)}"
        )


@app.get("/video-info")
async def get_video_info(url: str, session: Dict = Depends(get_user_session)):
    """
    Get metadata for a YouTube video by URL.

    Returns detailed video information including title, channel, views,
    thumbnail URL, and other metadata.

    This endpoint is available for anonymous users with no rate limit.
    """
    try:
        # Extract video ID from URL
        video_id = youtube_service.extract_youtube_id(url)

        # Get video metadata using the new service
        metadata = await youtube_service.get_video_info(video_id)
        return metadata

    except ValueError as e:
        logger.error(f"Invalid video URL: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving video metadata: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve video metadata: {str(e)}"
        )


@app.get("/channel/{channel_name}")
async def get_channel_info(
    channel_name: str,
    # auth: bool = Depends(verify_api_key),
    session: Dict = Depends(get_user_session),
):
    """
    Get information about a YouTube channel to validate user input.
    Returns channel title, description, thumbnail URL, and video count.
    """
    try:
        # Get channel info using the new service
        channel_info = await youtube_service.get_channel_info(channel_name)
        return channel_info

    except ValueError as e:
        logger.error(f"Invalid channel: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting channel info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get channel information: {str(e)}"
        )


# @app.post("/channel/download/async")
# async def start_channel_transcript_download(
#     request: ChannelRequest,
#     payload: dict = Depends(validate_jwt),
#     session: Dict = Depends(get_user_session),
# ):
#     """
#     Start asynchronous download of transcripts for all videos in a YouTube channel.
#     Returns a job ID that can be used to check progress and retrieve results.

#     REQUIRES AUTHENTICATION: This endpoint requires sufficient credits for all videos.
#     Each video transcript attempt will deduct 1 credit.
#     """
#     channel_name = request.channel_name
#     max_results = request.max_results
#     session_id = session["id"]
#     user_id = get_user_id_from_payload(payload)

#     try:
#         # Check if user has sufficient credits for the requested number of videos
#         user_credits = CreditManager.get_user_credits(user_id)
#         if user_credits < max_results:
#             raise HTTPException(
#                 status_code=status.HTTP_402_PAYMENT_REQUIRED,
#                 detail=f"Insufficient credits. You need {max_results} credits but only have {user_credits}. Please purchase more credits.",
#             )

#         # Start asynchronous transcript retrieval using the new service
#         job_id = await youtube_service.start_channel_transcript_download(
#             channel_name,
#             max_results,
#             # session_id,
#             user_id,  # Pass user_id for credit deduction
#             include_timestamps=request.include_timestamps,
#             include_video_title=request.include_video_title,
#             include_video_id=request.include_video_id,
#             include_video_url=request.include_video_url,
#             include_view_count=request.include_view_count,
#             concatenate_all=request.concatenate_all,
#         )  # Return job ID for polling status
#         return {
#             "job_id": job_id,
#             "status": "processing",
#             "total_videos": youtube_service.channel_download_jobs[job_id][
#                 "total_videos"
#             ],
#             "channel_name": channel_name,
#             "user_id": user_id,
#             "credits_reserved": max_results,
#             "user_credits_at_start": user_credits,
#             "message": f"Transcript retrieval started. Credits will be deducted per video attempt (1 credit each). You have {user_credits} credits available. Use the /channel/download/status endpoint to check progress and credit usage.",
#         }

#     except ValueError as e:
#         logger.error(f"Invalid channel request: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error starting transcript download: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail=f"Failed to start transcript download: {str(e)}"
#         )


# @app.get("/channel/{channel_name}/videos")
# async def list_channel_videos(
#     channel_name: str,
#     max_results: int = 30,
#     # auth: bool = Depends(verify_api_key),
#     session: Dict = Depends(get_user_session),
# ):
#     """
#     Get a list of videos from a channel with their metadata.
#     Useful for selective transcript downloading.
#     """
#     try:
#         # Get channel info
#         channel_info = await youtube_service.get_channel_info(channel_name)
#         channel_id = channel_info["channelId"]

#         # Get videos from channel
#         videos = await youtube_service.get_channel_videos(channel_id, max_results)

#         # Return combined info
#         return {
#             "channel": channel_info,
#             "videos": videos,
#         }

#     except ValueError as e:
#         logger.error(f"Invalid request: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error listing channel videos: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail=f"Failed to list channel videos: {str(e)}"
#         )


@app.get("/channel/{channel_name}/all-videos")
async def list_all_channel_videos(
    channel_name: str,
    background_tasks: BackgroundTasks,
):
    """
    Start fetching all videos from a YouTube channel asynchronously.
    Returns a job ID immediately that can be used to check progress.
    """
    try:
        # Create job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        job_data = {
            "status": "processing",
            "channel_name": channel_name,
            "start_time": time.time(),
            "videos": None,
            "error": None,
            "channel_info": None,
        }

        # Save job to persistent storage
        save_video_job_to_file(job_id, job_data)

        # Start background task
        background_tasks.add_task(fetch_channel_videos_task, job_id, channel_name)

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Fetching channel videos in background. Use /channel/videos-status/{job_id} to check progress.",
        }

    except Exception as e:
        logger.error(f"Error starting channel video fetch: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start video fetch: {str(e)}"
        )


@app.post("/channel/download/selected")
async def download_selected_videos(
    request: SelectedVideosRequest,
    payload: dict = Depends(validate_jwt),
    session: Dict = Depends(get_user_session),
):
    """
    Ultra-fast response workflow:
    1. Validate credits & channel (quick)
    2. Create job immediately
    3. Start background task for pre-fetching + Lambda dispatch
    4. Return job_id in ~200ms

    REQUIRES AUTHENTICATION: This endpoint requires sufficient credits for the selected videos.
    Each video transcript attempt will deduct 1 credit.
    """
    channel_name = request.channel_name
    videos = request.videos
    user_id = get_user_id_from_payload(payload)
    num_videos = len(videos)

    try:
        # Check if user has sufficient credits for the requested videos - NOW ASYNC
        user_credits = await CreditManager.get_user_credits(user_id)
        if user_credits < num_videos:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. You need {num_videos} credits but only have {user_credits}. Please purchase more credits.",
            )

        # 2. Quick channel validation (< 100ms)
        channel_info = await youtube_service.get_channel_info(request.channel_name)

        # 3. Reserve credits immediately - NOW ASYNC
        reservation_id = await CreditManager.reserve_credits(user_id, num_videos)

        # 4. Create job immediately (no metadata yet)
        job_id = str(uuid.uuid4())
        job_data = {
            "status": "initializing",  # New status for pre-fetching phase
            "channel_name": request.channel_name,
            "channel_info": channel_info,
            "source_id": channel_info.get("id", request.channel_name),
            "source_name": channel_info.get("title", request.channel_name),
            "source_type": "channel",
            "total_videos": num_videos,
            "completed": 0,
            "failed_count": 0,
            "processed_count": 0,
            "files": [],
            "videos": videos,
            "start_time": time.time(),
            "user_id": user_id,
            "credits_reserved": num_videos,
            "credits_used": 0,
            "reservation_id": reservation_id,
            "videos_metadata": {},  # Empty initially
            "prefetch_completed": False,  # Track pre-fetch progress
            "lambda_dispatched_count": 0,  # Track dispatched Lambda functions
            "formatting_options": {
                "include_timestamps": request.include_timestamps,
                "include_video_title": request.include_video_title,
                "include_video_id": request.include_video_id,
                "include_video_url": request.include_video_url,
                "include_view_count": request.include_view_count,
                "concatenate_all": request.concatenate_all,
            },
        }

        # Save job immediately using hybrid manager (database + file fallback)
        await hybrid_job_manager.create_job(job_id, job_data, videos)
        logger.info(
            f"Created job {job_id} - starting background pre-fetch for {num_videos} videos"
        )

        # 5. Start background task for pre-fetching + Lambda dispatch
        asyncio.create_task(youtube_service.prefetch_and_dispatch_task(job_id))

        return {
            "job_id": job_id,
            "status": "initializing",  # User knows pre-fetching is happening
            "total_videos": num_videos,
            "channel_name": channel_name,
            "user_id": user_id,
            "credits_reserved": num_videos,
            "user_credits_at_start": user_credits,
            "message": f"Job created. Pre-fetching metadata for {num_videos} videos, then starting Lambda processing.",
        }

    except ValueError as e:
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting transcript download: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start transcript download: {str(e)}"
        )


@app.get("/channel/videos-status/{job_id}")
async def get_videos_fetch_status(
    job_id: str,
):
    """
    Check the status of a video fetching job and return results if complete.
    """
    try:
        # Load job from file
        job = load_video_job_from_file(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        status = job["status"]

        if status == "processing":
            elapsed_time = time.time() - job["start_time"]
            return {
                "job_id": job_id,
                "status": "processing",
                "channel_name": job.get("channel_name"),
                "playlist_id": job.get("playlist_id"),
                "elapsed_time": elapsed_time,
                "message": "Still fetching videos...",
            }

        elif status == "completed":
            elapsed_time = job["end_time"] - job["start_time"]

            # Return the channel data or playlist data
            if "channel_info" in job:
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "elapsed_time": elapsed_time,
                    "channel": job["channel_info"],
                    "videos": job["videos"],
                    "video_count": job["video_count"],
                    "duration_breakdown": job["duration_breakdown"],
                }
            else:
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "elapsed_time": elapsed_time,
                    "playlist": job["playlist_info"],
                    "videos": job["videos"],
                    "video_count": job["video_count"],
                    "duration_breakdown": job["duration_breakdown"],
                }

        elif status == "failed":
            elapsed_time = job["end_time"] - job["start_time"]
            return {
                "job_id": job_id,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "error": job["error"],
            }

        else:
            return {"job_id": job_id, "status": status, "message": "Unknown status"}

    except Exception as e:
        logger.error(f"Error getting video fetch status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.get("/channel/download/status/{job_id}")
async def get_transcript_download_status(
    job_id: str,
    # auth: bool = Depends(verify_api_key),
    # session: Dict = Depends(get_user_session),
):
    """
    Check the status of an asynchronous transcript download job.
    Returns progress information and, if completed, a link to download the results.
    """
    try:
        # Get job status from the new service
        status = await youtube_service.get_job_status_async(job_id)

        # If the job is completed, include download URL information
        if status["status"] == "completed":
            status["download_url"] = f"/channel/download/results/{job_id}"
            status["message"] = (
                f"Transcript download complete. {status['completed']} videos processed successfully."
            )

        return status

    except ValueError as e:
        logger.error(f"Invalid job request: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking download status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to check download status: {str(e)}"
        )


@app.get("/channel/download/results/{job_id}")
async def download_transcript_results(
    job_id: str,
    payload: dict = Depends(validate_jwt),
    session: Dict = Depends(get_user_session),
):
    """
    Download the transcripts for a completed job as a ZIP file.
    Files are fetched from S3 and ZIP is generated on-demand with concurrent downloads.
    Only available when job status is 'completed' or 'completed_with_errors'.

    REQUIRES AUTHENTICATION: This endpoint requires a valid JWT token.
    """
    try:
        # Verify user owns this job
        job = await hybrid_job_manager.get_job(job_id, include_videos=True)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        user_id = get_user_id_from_payload(payload)
        job_user_id = job.get("user_id")

        # Convert both to strings for comparison (handles UUID vs string mismatch)
        if str(job_user_id) != str(user_id):
            raise HTTPException(
                status_code=403, detail="Access denied - you don't own this job"
            )

        # Verify job is completed
        job_status = job.get("status")
        if job_status not in ["completed", "completed_with_errors"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job is not ready for download. Current status: {job_status}",
            )

        # Create ZIP from S3 files using concurrent downloads
        logger.info(
            f"Creating ZIP for job {job_id} from S3 files with concurrent downloads"
        )
        zip_start_time = time.time()

        try:
            zip_buffer = await youtube_service.create_transcript_zip_from_s3_concurrent(
                job_id
            )
        except Exception as s3_error:
            logger.warning(
                f"Concurrent S3 download failed, falling back to sequential: {str(s3_error)}"
            )
            # Fallback to sequential downloads if concurrent fails
            zip_buffer = await youtube_service.create_transcript_zip_from_s3_sequential(
                job_id
            )

        zip_end_time = time.time()
        zip_size = len(zip_buffer.getvalue())

        logger.info(
            f"Created ZIP file for job {job_id} in {zip_end_time - zip_start_time:.2f}s, "
            f"size: {zip_size:,} bytes ({zip_size / 1024 / 1024:.2f} MB)"
        )

        # Get source name for filename
        source_name = job.get("source_name") or job.get("channel_name", "transcripts")
        safe_source_name = youtube_service.sanitize_filename(source_name)

        # Check if this is a concatenated download to adjust filename
        is_concatenated = job.get("formatting_options", {}).get(
            "concatenate_all", False
        )
        filename_suffix = (
            "_concatenated_transcripts.zip" if is_concatenated else "_transcripts.zip"
        )

        # Add statistics to response headers
        headers = {
            "Content-Disposition": f'attachment; filename="{safe_source_name}{filename_suffix}"',
            "X-Job-ID": job_id,
            "X-Files-Count": str(len(job.get("files", []))),
            "X-Generation-Time-Seconds": f"{zip_end_time - zip_start_time:.2f}",
            "X-Source-Type": job.get("source_type", "channel"),
        }

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers=headers,
        )

    except ValueError as e:
        logger.error(f"Invalid job request for {job_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(
            f"Error downloading results for job {job_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to download results: {str(e)}"
        )


# =============================================
# PLAYLIST ENDPOINTS
# =============================================


@app.get("/playlist/{playlist_id}")
async def get_playlist_info(
    playlist_id: str,
    session: Dict = Depends(get_user_session),
):
    """
    Get information about a YouTube playlist to validate user input.
    Returns playlist title, description, thumbnail URL, and video count.
    """
    try:
        # Extract playlist ID from URL if needed
        clean_playlist_id = youtube_service.extract_playlist_id(playlist_id)

        # Get playlist info using the new service
        playlist_info = await youtube_service.get_playlist_info(clean_playlist_id)
        return playlist_info

    except ValueError as e:
        logger.error(f"Invalid playlist: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting playlist info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get playlist information: {str(e)}"
        )


@app.get("/playlist/{playlist_id}/all-videos")
async def list_all_playlist_videos(
    playlist_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Start fetching all videos from a YouTube playlist asynchronously.
    Returns a job ID immediately that can be used to check progress.
    """
    try:
        # Extract playlist ID from URL if needed
        clean_playlist_id = youtube_service.extract_playlist_id(playlist_id)

        # Create job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        job_data = {
            "status": "processing",
            "playlist_id": clean_playlist_id,
            "start_time": time.time(),
            "videos": None,
            "error": None,
            "playlist_info": None,
        }

        # Save job to persistent storage
        save_video_job_to_file(job_id, job_data)

        # Start background task
        background_tasks.add_task(fetch_playlist_videos_task, job_id, clean_playlist_id)

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Fetching playlist videos in background. Use /channel/videos-status/{job_id} to check progress.",
        }

    except Exception as e:
        logger.error(f"Error starting playlist video fetch: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start video fetch: {str(e)}"
        )


@app.post("/playlist/download/selected")
async def download_selected_playlist_videos(
    request: SelectedPlaylistVideosRequest,
    payload: dict = Depends(validate_jwt),
    session: Dict = Depends(get_user_session),
):
    """
    Ultra-fast response workflow for playlist downloads:
    1. Validate credits & playlist (quick)
    2. Create job immediately
    3. Start background task for pre-fetching + Lambda dispatch
    4. Return job_id in ~200ms

    REQUIRES AUTHENTICATION: This endpoint requires sufficient credits for the selected videos.
    Each video transcript attempt will deduct 1 credit.
    """
    playlist_name = request.playlist_name
    videos = request.videos
    user_id = get_user_id_from_payload(payload)
    num_videos = len(videos)

    try:
        # Check if user has sufficient credits for the requested videos - NOW ASYNC
        user_credits = await CreditManager.get_user_credits(user_id)
        if user_credits < num_videos:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. You need {num_videos} credits but only have {user_credits}. Please purchase more credits.",
            )

        # 2. Quick playlist validation (< 100ms)
        clean_playlist_id = youtube_service.extract_playlist_id(playlist_name)
        playlist_info = await youtube_service.get_playlist_info(clean_playlist_id)

        # 3. Reserve credits immediately - NOW ASYNC
        reservation_id = await CreditManager.reserve_credits(user_id, num_videos)

        # 4. Create job immediately (no metadata yet)
        job_id = str(uuid.uuid4())
        job_data = {
            "status": "initializing",  # New status for pre-fetching phase
            "playlist_id": clean_playlist_id,
            "playlist_info": playlist_info,
            "source_id": clean_playlist_id,
            "source_name": playlist_info.get("title", clean_playlist_id),
            "source_type": "playlist",
            "total_videos": num_videos,
            "completed": 0,
            "failed_count": 0,
            "processed_count": 0,
            "files": [],
            "videos": videos,
            "start_time": time.time(),
            "user_id": user_id,
            "credits_reserved": num_videos,
            "credits_used": 0,
            "reservation_id": reservation_id,
            "videos_metadata": {},  # Empty initially
            "prefetch_completed": False,  # Track pre-fetch progress
            "lambda_dispatched_count": 0,  # Track dispatched Lambda functions
            "is_playlist": True,  # Flag to indicate this is a playlist download
            "formatting_options": {
                "include_timestamps": request.include_timestamps,
                "include_video_title": request.include_video_title,
                "include_video_id": request.include_video_id,
                "include_video_url": request.include_video_url,
                "include_view_count": request.include_view_count,
                "concatenate_all": request.concatenate_all,
            },
        }

        # Save job immediately using hybrid manager (database + file fallback)
        await hybrid_job_manager.create_job(job_id, job_data, videos)
        logger.info(
            f"Created playlist job {job_id} - starting background pre-fetch for {num_videos} videos"
        )

        # 5. Start background task for pre-fetching + Lambda dispatch
        asyncio.create_task(youtube_service.prefetch_and_dispatch_task(job_id))

        return {
            "job_id": job_id,
            "status": "initializing",  # User knows pre-fetching is happening
            "total_videos": num_videos,
            "playlist_id": clean_playlist_id,
            "user_id": user_id,
            "credits_reserved": num_videos,
            "user_credits_at_start": user_credits,
            "message": f"Job created. Pre-fetching metadata for {num_videos} videos, then starting Lambda processing.",
        }

    except ValueError as e:
        logger.error(f"Invalid playlist request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error starting playlist transcript download: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start playlist transcript download: {str(e)}",
        )


# =============================================
# INTERNAL LAMBDA CALLBACK ENDPOINTS
# =============================================


@app.post("/internal/job/{job_id}/video-complete")
async def video_completed(job_id: str, completion_data: dict):
    """
    Internal endpoint for Lambda to report video completion.
    Updates job progress and file tracking.
    """
    try:
        # Check execution time for monitoring
        job = await hybrid_job_manager.get_job_status(job_id)
        if job and job.get("lambda_dispatch_time"):
            # Convert datetime to timestamp if needed
            dispatch_time = job["lambda_dispatch_time"]
            if isinstance(dispatch_time, datetime):
                dispatch_time = dispatch_time.timestamp()

            execution_time = time.time() - dispatch_time
            if execution_time > 300:  # 5 minutes
                logger.warning(
                    f"Video {completion_data['video_id']} took {execution_time/60:.1f} minutes to complete "
                    f"(potential Lambda delay/timeout recovery)"
                )
            elif execution_time > 120:  # 2 minutes
                logger.info(
                    f"Video {completion_data['video_id']} took {execution_time/60:.1f} minutes to complete"
                )

        # Check if this video was already counted as timed out
        if job and job.get("timeout_occurred"):
            logger.info(
                f"Video {completion_data['video_id']} completed after timeout was triggered "
                f"- this is a late-arriving Lambda response"
            )

            # # Don't process further, job already finalized
            # # But currently I want to process it anyway to keep accurate counts. I will enable download before that.
            # return {
            #     "status": "ignored",
            #     "reason": "late_arrival_after_timeout",
            #     "job_id": job_id,
            # }

        # Mark video as completed with atomic database operations
        updated_job = await hybrid_job_manager.mark_video_completed(
            job_id=job_id,
            video_id=completion_data["video_id"],
            file_info={
                "s3_key": completion_data["s3_key"],
                "transcript_length": completion_data.get("transcript_length", 0),
                "status": "completed",
            },
        )

        logger.info(f"Video {completion_data['video_id']} completed for job {job_id}")

        # Check if job is complete
        if (
            updated_job
            and updated_job["processed_count"] >= updated_job["total_videos"]
        ):
            # Finalize credits (refund unused) - NOW ASYNC
            await CreditManager.finalize_credit_usage(
                user_id=updated_job["user_id"],
                reservation_id=updated_job["reservation_id"],
                credits_used=updated_job["credits_used"],
                credits_reserved=updated_job["credits_reserved"],
            )

            # Update job status to completed
            await hybrid_job_manager.update_job(job_id, status="completed")
            logger.info(f"Job {job_id} completed - all videos processed")

        return {"status": "updated", "job_id": job_id}

    except Exception as e:
        logger.error(f"Error updating job progress: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/internal/job/{job_id}/video-failed")
async def video_failed(job_id: str, failure_data: dict):
    """
    Internal endpoint for Lambda to report video failure.
    """
    try:
        # Check if this video was already counted as timed out
        job = await hybrid_job_manager.get_job_status(job_id)
        if job and job.get("timeout_occurred"):
            logger.info(
                f"Video {failure_data['video_id']} failed after timeout was triggered "
                f"- this is a late-arriving Lambda response"
            )
            # # Don't process further, job already finalized
            # return {
            #     "status": "ignored",
            #     "reason": "late_failure_after_timeout",
            #     "job_id": job_id,
            # }

        updated_job = await hybrid_job_manager.mark_video_failed(
            job_id=job_id,
            video_id=failure_data["video_id"],
            error_message=failure_data.get("error", "Unknown error"),
        )

        logger.warning(
            f"Video {failure_data['video_id']} failed for job {job_id}: {failure_data.get('error', 'Unknown error')}"
        )

        # Check if job is complete (including failures)
        if (
            updated_job
            and updated_job["processed_count"] >= updated_job["total_videos"]
        ):
            # Finalize credits - NOW ASYNC
            await CreditManager.finalize_credit_usage(
                user_id=updated_job["user_id"],
                reservation_id=updated_job["reservation_id"],
                credits_used=updated_job["credits_used"],
                credits_reserved=updated_job["credits_reserved"],
            )

            # Update job status
            status = (
                "completed_with_errors"
                if updated_job["failed_count"] > 0
                else "completed"
            )
            await hybrid_job_manager.update_job(job_id, status=status)
            logger.info(
                f"Job {job_id} completed with {updated_job['failed_count']} failures"
            )

        return {"status": "updated", "job_id": job_id}

    except Exception as e:
        logger.error(f"Error updating job failure: {str(e)}")
        return {"status": "error", "message": str(e)}


# =============================================
# DEBUG ENDPOINTS
# =============================================


@app.get("/debug/memory")
async def get_memory_stats():
    """
    Get current memory usage statistics and top memory allocations
    """
    # Get current process memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Get tracemalloc statistics if enabled
    stats = {}
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        stats = {
            "tracemalloc_current_mb": current / 1024 / 1024,
            "tracemalloc_peak_mb": peak / 1024 / 1024,
            "top_allocations": [
                {
                    "file": (
                        str(stat.traceback.format()[0]) if stat.traceback else "unknown"
                    ),
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats[:10]  # Top 10 allocations
            ],
        }
    else:
        stats = {"tracemalloc_enabled": False}

    return {
        "rss_memory_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_memory_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent_memory": process.memory_percent(),
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
        **stats,
    }


@app.post("/debug/gc")
async def force_garbage_collection():
    """
    Force garbage collection and return memory stats before/after
    """
    # Memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024

    # Force garbage collection
    collected = gc.collect()

    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024

    return {
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_freed_mb": memory_before - memory_after,
        "objects_collected": collected,
    }


@app.get("/debug/jobs")
async def get_jobs_debug():
    """
    Get information about current jobs in memory and on disk
    """
    # In-memory jobs
    in_memory_jobs = (
        len(youtube_service.channel_download_jobs)
        if hasattr(youtube_service, "channel_download_jobs")
        else 0
    )

    # Jobs on disk
    jobs_dir = os.path.join(settings.temp_dir, "jobs")
    disk_jobs = 0
    disk_job_files = []

    if os.path.exists(jobs_dir):
        job_files = [f for f in os.listdir(jobs_dir) if f.endswith(".json")]
        disk_jobs = len(job_files)

        # Get file sizes
        for job_file in job_files[:10]:  # Limit to first 10 for debugging
            file_path = os.path.join(jobs_dir, job_file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            disk_job_files.append({"filename": job_file, "size_kb": file_size})

    return {
        "in_memory_jobs": in_memory_jobs,
        "disk_jobs": disk_jobs,
        "sample_disk_jobs": disk_job_files,
        "temp_dir": settings.temp_dir,
    }


# Create a background task to clean up old jobs periodically


async def cleanup_job():
    """Periodically GC"""
    while True:
        try:
            # Wait for a while before cleaning up (e.g., every hour)
            await asyncio.sleep(3600)  # 1 hour

            # Log memory before cleanup
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory before cleanup: {memory_before:.2f} MB")

            # Run the cleanup function
            logger.info("Starting scheduled garbage collection")
            # remove porque esto no guardo mas en diccionario, todo va al json.
            # youtube_service.cleanup_old_jobs(max_age_hours=24)  # Keep jobs for 24 hours

            # Force garbage collection after cleanup
            collected = gc.collect()

            # Log memory after cleanup
            memory_after = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"Memory after cleanup: {memory_after:.2f} MB (freed: {memory_before - memory_after:.2f} MB, GC collected: {collected} objects)"
            )

            logger.info("Scheduled garbage collection completed")

        except Exception as e:
            logger.error(f"Error in cleanup job: {str(e)}")
            # Wait a bit before retrying if there was an error
            await asyncio.sleep(300)  # 5 minutes


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "transcript_api:app", host=settings.host, port=settings.port, reload=True
    )


#### TEST
