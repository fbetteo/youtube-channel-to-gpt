#!/usr/bin/env python3
"""
Memory monitoring utility for the transcript API
"""
import requests
import time
import json


def monitor_memory(base_url="http://localhost:8001"):
    """Monitor memory usage of the transcript API"""

    print("Starting memory monitoring...")
    print("=" * 60)

    while True:
        try:
            # Get memory stats
            response = requests.get(f"{base_url}/debug/memory")
            if response.status_code == 200:
                stats = response.json()

                print(f"Time: {time.strftime('%H:%M:%S')}")
                print(f"RSS Memory: {stats['rss_memory_mb']:.2f} MB")
                print(f"Virtual Memory: {stats['vms_memory_mb']:.2f} MB")
                print(f"Memory %: {stats['percent_memory']:.2f}%")

                if "tracemalloc_current_mb" in stats:
                    print(
                        f"Tracemalloc Current: {stats['tracemalloc_current_mb']:.2f} MB"
                    )
                    print(f"Tracemalloc Peak: {stats['tracemalloc_peak_mb']:.2f} MB")

                    # Show top 3 memory allocations
                    if "top_allocations" in stats and stats["top_allocations"]:
                        print("Top allocations:")
                        for i, alloc in enumerate(stats["top_allocations"][:3]):
                            print(
                                f"  {i+1}. {alloc['size_mb']:.2f} MB - {alloc['file']}"
                            )

                print("-" * 40)

            else:
                print(f"Error getting memory stats: {response.status_code}")

            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\nMemory monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    monitor_memory()
