#!/usr/bin/env python3
"""
Consume Lambda transcript result messages from SQS and apply them to the local DB.

Run this as a separate long-lived worker on Hetzner so Lambda completion bursts are
buffered in SQS instead of hitting the public FastAPI service directly.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List

import boto3


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from config_v2 import settings
from db_youtube_transcripts.database import close_db_pool, init_db_pool
from job_result_processor import process_lambda_result_message


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SQSResultConsumer:
    def __init__(self):
        if not settings.lambda_results_queue_url:
            raise ValueError(
                "LAMBDA_RESULTS_QUEUE_URL is required for the SQS consumer"
            )

        self.queue_url = settings.lambda_results_queue_url
        self.wait_time_seconds = settings.sqs_consumer_wait_time_seconds
        self.max_messages = settings.sqs_consumer_max_messages
        self.visibility_timeout = settings.sqs_consumer_visibility_timeout
        self.concurrency = settings.sqs_consumer_concurrency
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.stop_event = asyncio.Event()
        self.sqs_client = boto3.client("sqs", region_name=settings.aws_default_region)

    def request_stop(self, signum=None, frame=None):
        if not self.stop_event.is_set():
            logger.info(f"Shutdown requested for SQS consumer (signal={signum})")
            self.stop_event.set()

    async def receive_messages(self) -> List[Dict[str, Any]]:
        response = await asyncio.to_thread(
            self.sqs_client.receive_message,
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=self.max_messages,
            WaitTimeSeconds=self.wait_time_seconds,
            VisibilityTimeout=self.visibility_timeout,
            AttributeNames=["ApproximateReceiveCount", "SentTimestamp"],
        )
        return response.get("Messages", [])

    async def delete_message(self, receipt_handle: str) -> None:
        await asyncio.to_thread(
            self.sqs_client.delete_message,
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
        )

    async def process_message(self, message: Dict[str, Any]) -> bool:
        async with self.semaphore:
            receipt_handle = message["ReceiptHandle"]
            receive_count = message.get("Attributes", {}).get(
                "ApproximateReceiveCount", "1"
            )

            try:
                body = json.loads(message["Body"])
                logger.info(
                    f"Processing SQS message event_type={body.get('event_type')} "
                    f"job_id={body.get('job_id')} video_id={body.get('video_id')} "
                    f"receive_count={receive_count}"
                )
                await process_lambda_result_message(body)
                await self.delete_message(receipt_handle)
                return True
            except Exception as e:
                logger.error(
                    f"Failed to process SQS message receive_count={receive_count}: {str(e)}",
                    exc_info=True,
                )
                return False

    async def run(self) -> None:
        logger.info(
            f"Starting SQS result consumer for queue {self.queue_url} "
            f"(max_messages={self.max_messages}, visibility_timeout={self.visibility_timeout}, "
            f"concurrency={self.concurrency})"
        )

        await init_db_pool()

        try:
            while not self.stop_event.is_set():
                messages = await self.receive_messages()
                if not messages:
                    continue

                logger.info(f"Received {len(messages)} message(s) from SQS")
                await asyncio.gather(
                    *(self.process_message(message) for message in messages)
                )
        finally:
            await close_db_pool()
            logger.info("SQS result consumer stopped")


async def main() -> None:
    consumer = SQSResultConsumer()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, consumer.request_stop)

    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())
