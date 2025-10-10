#!/usr/bin/env python3
"""
Performance test for concurrent Lambda dispatch
"""
import asyncio
import time
from typing import List, Dict, Any


# Mock video objects for testing
class MockVideo:
    def __init__(self, video_id: str):
        self.id = video_id


def mock_sequential_dispatch(videos: List[MockVideo]) -> int:
    """Simulate sequential Lambda dispatch (old way)"""
    print(f"Sequential dispatch: Processing {len(videos)} videos...")
    start_time = time.time()

    dispatched = 0
    for video in videos:
        # Simulate Lambda invocation delay (network call)
        time.sleep(0.1)  # 100ms per video
        dispatched += 1
        if dispatched % 10 == 0:
            print(f"  Dispatched {dispatched}/{len(videos)} videos")

    end_time = time.time()
    print(f"Sequential dispatch completed in {end_time - start_time:.2f} seconds")
    return dispatched


async def mock_concurrent_dispatch(
    videos: List[MockVideo], max_concurrent: int = 20
) -> int:
    """Simulate concurrent Lambda dispatch (new way)"""
    print(
        f"Concurrent dispatch: Processing {len(videos)} videos (max {max_concurrent} concurrent)..."
    )
    start_time = time.time()

    # Semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch_single(video: MockVideo) -> bool:
        async with semaphore:
            # Simulate Lambda invocation delay
            await asyncio.sleep(0.1)  # 100ms per video
            return True

    # Process in batches to control memory
    batch_size = 50
    total_dispatched = 0

    for i in range(0, len(videos), batch_size):
        batch = videos[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(videos) + batch_size - 1) // batch_size

        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} videos)")

        # Dispatch batch concurrently
        tasks = [dispatch_single(video) for video in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_dispatched = sum(1 for r in results if r is True)
        total_dispatched += batch_dispatched

        print(f"  Batch {batch_num} completed: {batch_dispatched} dispatched")

    end_time = time.time()
    print(f"Concurrent dispatch completed in {end_time - start_time:.2f} seconds")
    return total_dispatched


def calculate_memory_usage(num_videos: int) -> Dict[str, float]:
    """Calculate estimated memory usage"""
    # Rough estimates based on typical video metadata
    video_metadata_mb = 0.001  # 1KB per video metadata
    lambda_payload_mb = 0.002  # 2KB per Lambda payload

    sequential_memory = video_metadata_mb * num_videos  # All in memory at once
    concurrent_memory = lambda_payload_mb * 20  # Max 20 concurrent payloads

    return {
        "sequential_mb": sequential_memory,
        "concurrent_mb": concurrent_memory,
        "memory_savings_percent": (
            ((sequential_memory - concurrent_memory) / sequential_memory * 100)
            if sequential_memory > 0
            else 0
        ),
    }


async def main():
    """Test concurrent vs sequential Lambda dispatch"""

    test_scenarios = [
        {"num_videos": 10, "max_concurrent": 5},
        {"num_videos": 50, "max_concurrent": 10},
        {"num_videos": 200, "max_concurrent": 20},
        {"num_videos": 500, "max_concurrent": 20},
    ]

    print("=== Lambda Dispatch Performance Test ===")
    print()

    for scenario in test_scenarios:
        num_videos = scenario["num_videos"]
        max_concurrent = scenario["max_concurrent"]

        print(f"Scenario: {num_videos} videos, max {max_concurrent} concurrent")
        print("=" * 60)

        # Create mock videos
        videos = [MockVideo(f"video_{i}") for i in range(num_videos)]

        # Test sequential dispatch
        sequential_start = time.time()
        sequential_dispatched = mock_sequential_dispatch(videos)
        sequential_time = time.time() - sequential_start

        print()

        # Test concurrent dispatch
        concurrent_start = time.time()
        concurrent_dispatched = await mock_concurrent_dispatch(videos, max_concurrent)
        concurrent_time = time.time() - concurrent_start

        # Calculate performance improvement
        time_improvement = (
            ((sequential_time - concurrent_time) / sequential_time * 100)
            if sequential_time > 0
            else 0
        )
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0

        # Calculate memory usage
        memory_stats = calculate_memory_usage(num_videos)

        print()
        print("Results:")
        print(
            f"  Sequential: {sequential_time:.2f}s, {sequential_dispatched} dispatched"
        )
        print(
            f"  Concurrent: {concurrent_time:.2f}s, {concurrent_dispatched} dispatched"
        )
        print(f"  Time improvement: {time_improvement:.1f}% ({speedup:.1f}x faster)")
        print(
            f"  Memory usage: {memory_stats['sequential_mb']:.2f}MB → {memory_stats['concurrent_mb']:.2f}MB"
        )
        print(f"  Memory savings: {memory_stats['memory_savings_percent']:.1f}%")
        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    asyncio.run(main())
