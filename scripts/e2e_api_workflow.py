"""
End-to-end walkthrough of the Developer API, the way an API user would use it.

Steps:
  1. Check credits
  2. Look up the channel
  3. List the channel's videos
  4. Download one video transcript (sync)
  5. Start a channel job for N videos (async)
  6. Poll the job until it finishes
  7. Download the ZIP and inspect it
  8. Check credits again and compare with what the job reported

Each run saves a JSON report (with timings and failure counts), a log file and
the ZIP under scripts/e2e_results/, and appends one row to runs.csv so runs can
be compared later.

COSTS REAL CREDITS: 1 for the single video + 1 per channel video.

Usage (cmd):
    set YOUTUBE_TRANSCRIPT_API_KEY=yt_...
    set YOUTUBE_TRANSCRIPT_API_BASE_URL=http://127.0.0.1:8000   (optional, default: hosted API)
    poetry run python scripts/e2e_api_workflow.py --channel @mkbhd --max-videos 5 --interactive
    poetry run python scripts/e2e_api_workflow.py --channel @mkbhd --max-videos 20 --label long-videos
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

DEFAULT_BASE_URL = "https://api.youtubetranscripts.fbetteo.com"
RESULTS_DIR = Path(__file__).parent / "e2e_results"
TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "cancelled"}

log = logging.getLogger("e2e")


class StopRun(Exception):
    """User chose to quit in interactive mode."""


class Runner:
    def __init__(self, args, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.base_url = args.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = args.api_key
        self.report = {
            "label": args.label,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.base_url,
            "channel": args.channel,
            "max_videos": args.max_videos,
            "steps": {},
        }

    # ---- helpers ----

    def call(self, method: str, path: str, **kwargs) -> requests.Response:
        """One HTTP call, logged with its status and duration."""
        url = f"{self.base_url}/api/v1{path}"
        start = time.time()
        resp = self.session.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
        log.info(f"{method} {path} -> {resp.status_code} in {time.time() - start:.2f}s")
        if resp.status_code >= 400:
            log.error(f"Response: {resp.text[:500]}")
            resp.raise_for_status()
        return resp

    def pause(self, title: str, note: str = "") -> bool:
        """Interactive gate before each step. Returns False to skip the step."""
        log.info("")
        log.info(f"=== {title} ===")
        if note:
            log.info(note)
        if not self.args.interactive:
            return True
        answer = input("  [Enter] run  [s] skip  [q] quit > ").strip().lower()
        if answer == "q":
            raise StopRun()
        return answer != "s"

    def show(self, data) -> None:
        """Print a response body in interactive mode so you see what users see."""
        if self.args.interactive:
            text = json.dumps(data, indent=2, default=str)
            print(text if len(text) < 2000 else text[:2000] + "\n  ... (truncated)")

    def step(self, name: str, title: str, fn, note: str = ""):
        """Run one step: pause, time it, record success/failure in the report."""
        if not self.pause(title, note):
            self.report["steps"][name] = {"skipped": True}
            return None
        start = time.time()
        try:
            result = fn()
            ok = True
        except StopRun:
            raise
        except Exception as e:
            log.error(f"{title} failed: {e}")
            result, ok = {"error": str(e)}, False
        record = {"ok": ok, "seconds": round(time.time() - start, 2)}
        record.update(result or {})
        self.report["steps"][name] = record
        log.info(f"{title}: {'OK' if ok else 'FAILED'} in {record['seconds']}s")
        return record if ok else None

    # ---- steps ----

    def check_credits(self):
        data = self.call("GET", "/account/credits").json()
        self.show(data)
        log.info(f"Credits: {data['credits']}")
        return {"credits": data["credits"]}

    def channel_info(self):
        data = self.call("GET", f"/channels/{quote(self.args.channel)}/info").json()
        self.show(data)
        log.info(f"Channel: {data['title']} ({data.get('video_count')} videos)")
        return {"title": data["title"], "video_count": data.get("video_count")}

    def list_videos(self):
        # limit=max_videos lists exactly the videos the channel job will download.
        path = f"/channels/{quote(self.args.channel)}/videos?limit={self.args.max_videos}"
        data = self.call("GET", path, timeout=300).json()
        videos = data["videos"]
        self.show({k: v for k, v in data.items() if k != "videos"} | {"first_videos": videos[:3]})
        log.info(f"Listed {data['total_videos']} videos (has_more={data.get('has_more')}), "
                 f"durations: {data['duration_breakdown']}")
        for v in videos:
            minutes = (v.get("duration_seconds") or 0) / 60
            log.info(f"  {v.get('type') or '?':<5} {minutes:6.1f} min  {v['title'][:70]}")
        self.videos = videos
        return {
            "total_videos": data["total_videos"],
            "has_more": data.get("has_more"),
            "duration_breakdown": data["duration_breakdown"],
            "listed_minutes": round(sum(v.get("duration_seconds") or 0 for v in videos) / 60, 1),
        }

    def single_video(self):
        video_url = self.args.video
        duration = None
        if not video_url and getattr(self, "videos", None):
            video_url, duration = self.videos[0]["url"], self.videos[0].get("duration_seconds")
        if not video_url and self.args.interactive:
            video_url = input("  No video list available. Video URL/ID to use > ").strip()
        if not video_url:
            raise RuntimeError("No --video given and no channel video list to pick from")
        log.info(f"Video: {video_url}")
        data = self.call("POST", "/transcripts/single", json={"video_url": video_url}).json()
        self.show(data | {"transcript": data["transcript"][:300] + "..."})
        log.info(f"'{data['title']}': {data['character_count']:,} chars, language {data['language']}")
        return {
            "video_id": data["video_id"],
            "title": data["title"],
            "duration_seconds": duration,
            "character_count": data["character_count"],
            "language": data["language"],
        }

    def start_job(self):
        body = {"channel": self.args.channel, "max_videos": self.args.max_videos}
        data = self.call("POST", "/transcripts/channel", json=body).json()
        self.show(data)
        self.job_id = data["job_id"]
        log.info(f"Job {self.job_id} accepted with status '{data['status']}'")
        return {"job_id": self.job_id}

    def poll_job(self):
        start = time.time()
        timeline, last = [], None
        status = {}
        while time.time() - start < self.args.timeout:
            status = self.call("GET", f"/jobs/{self.job_id}").json()
            snapshot = (status["status"], status["processed_count"], status["failed_count"])
            if snapshot != last:  # only log/record when something changed
                elapsed = round(time.time() - start, 1)
                timeline.append({"t": elapsed, "status": status["status"],
                                 "total": status["total_videos"],
                                 "processed": status["processed_count"],
                                 "completed": status["completed"],
                                 "failed": status["failed_count"],
                                 "skipped": status["skipped_count"]})
                log.info(f"  [{elapsed:>6}s] {status['status']:<22} "
                         f"{status['processed_count']}/{status['total_videos']} processed, "
                         f"{status['completed']} ok, {status['failed_count']} failed, "
                         f"{status['skipped_count']} skipped")
                last = snapshot
            if status["status"] in TERMINAL_STATUSES:
                break
            time.sleep(self.args.poll_interval)
        else:
            log.warning(f"Timed out after {self.args.timeout}s, cancelling job (refunds unprocessed videos)")
            status = self.call("POST", f"/jobs/{self.job_id}/cancel").json()
            status["status"] = "timed_out_cancelled"

        self.show(status)
        # Discovery time = until the job left 'discovering'
        discovery = next((p["t"] for p in timeline if p["status"] != "discovering"), None)
        total = status.get("total_videos", 0)
        if status.get("error_message"):
            log.error(f"Job error: {status['error_message']}")
        return {
            "final_status": status["status"],
            "total_videos": total,
            "completed": status.get("completed", 0),
            "failed": status.get("failed_count", 0),
            "skipped": status.get("skipped_count", 0),
            "failure_rate": round(status.get("failed_count", 0) / total, 3) if total else None,
            "credits_used": status.get("credits_used", 0),
            "refunded_credits": status.get("refunded_credits", 0),
            "discovery_seconds": discovery,
            "server_elapsed_seconds": status.get("elapsed_seconds"),
            "error_message": status.get("error_message"),
            "timeline": timeline,
        }

    def download_zip(self):
        resp = self.call("GET", f"/jobs/{self.job_id}/download", timeout=600)
        zip_path = self.run_dir / "transcripts.zip"
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            files = [(i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]
        empty = [name for name, size in files if size == 0]
        log.info(f"ZIP: {len(resp.content):,} bytes, {len(files)} files, {len(empty)} empty -> {zip_path}")
        for name, size in files[:10]:
            log.info(f"  {size:>9,}  {name}")
        if len(files) > 10:
            log.info(f"  ... and {len(files) - 10} more")
        return {
            "zip_bytes": len(resp.content),
            "file_count": len(files),
            "empty_files": empty,
            "total_uncompressed_bytes": sum(size for _, size in files),
            "server_generation_time": resp.headers.get("X-Generation-Time"),
        }

    # ---- run ----

    def run(self):
        steps = self.report["steps"]
        self.step("credits_before", "1. Check credits", self.check_credits)
        self.step("channel_info", "2. Channel info (free)", self.channel_info)
        self.step("list_videos", "3. List channel videos (free)", self.list_videos)
        self.step("single_video", "4. Single video transcript", self.single_video,
                  note="Costs 1 credit.")
        if self.step("start_job", "5. Start channel job", self.start_job,
                     note=f"Costs up to {self.args.max_videos} credits."):
            self.step("job", "6. Poll job until done", self.poll_job)
            if steps.get("job", {}).get("completed"):
                self.step("download", "7. Download ZIP", self.download_zip)
        self.step("credits_after", "8. Check credits again", self.check_credits)

        # Credit sanity check: before - after should match what was charged.
        before = steps.get("credits_before", {}).get("credits")
        after = steps.get("credits_after", {}).get("credits")
        if before is not None and after is not None:
            expected = (1 if steps.get("single_video", {}).get("ok") else 0) + \
                steps.get("job", {}).get("credits_used", 0)
            self.report["credits"] = {"spent": before - after, "expected": expected}
            level = logging.INFO if before - after == expected else logging.WARNING
            log.log(level, f"Credits spent: {before - after}, expected from responses: {expected}")


def save_results(report: dict, run_dir: Path) -> None:
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    s = report["steps"]
    row = {
        "started_at": report["started_at"],
        "label": report["label"],
        "base_url": report["base_url"],
        "channel": report["channel"],
        "max_videos": report["max_videos"],
        "total_seconds": report.get("total_seconds"),
        "list_videos_s": s.get("list_videos", {}).get("seconds"),
        "listed_minutes": s.get("list_videos", {}).get("listed_minutes"),
        "single_s": s.get("single_video", {}).get("seconds"),
        "single_ok": s.get("single_video", {}).get("ok"),
        "single_chars": s.get("single_video", {}).get("character_count"),
        "single_duration_s": s.get("single_video", {}).get("duration_seconds"),
        "job_status": s.get("job", {}).get("final_status"),
        "job_s": s.get("job", {}).get("seconds"),
        "discovery_s": s.get("job", {}).get("discovery_seconds"),
        "job_total": s.get("job", {}).get("total_videos"),
        "job_completed": s.get("job", {}).get("completed"),
        "job_failed": s.get("job", {}).get("failed"),
        "job_skipped": s.get("job", {}).get("skipped"),
        "download_s": s.get("download", {}).get("seconds"),
        "zip_files": s.get("download", {}).get("file_count"),
        "zip_bytes": s.get("download", {}).get("zip_bytes"),
        "credits_spent": report.get("credits", {}).get("spent"),
        "credits_expected": report.get("credits", {}).get("expected"),
        "run_dir": run_dir.name,
    }
    csv_path = RESULTS_DIR / "runs.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def print_summary(report: dict) -> None:
    s = report["steps"]
    log.info("")
    log.info("=== Summary ===")
    for name, rec in s.items():
        state = "skipped" if rec.get("skipped") else ("ok" if rec.get("ok") else "FAILED")
        log.info(f"  {name:<15} {state:<8} {rec.get('seconds', '')}s")
    job = s.get("job", {})
    if job.get("total_videos"):
        log.info(f"  Job: {job['completed']}/{job['total_videos']} ok, {job['failed']} failed, "
                 f"{job['skipped']} skipped (failure rate {job['failure_rate']:.1%})")
        if job.get("seconds") and job.get("completed"):
            log.info(f"  Throughput: {job['seconds'] / job['completed']:.1f}s per completed video")
    log.info(f"  Total: {report.get('total_seconds')}s")


def main():
    parser = argparse.ArgumentParser(description="End-to-end Developer API workflow test (spends credits).")
    parser.add_argument("--channel", required=True, help="Channel handle, name or ID, e.g. @mkbhd")
    parser.add_argument("--max-videos", type=int, default=5, help="Videos for the channel job (default 5)")
    parser.add_argument("--video", help="Video URL/ID for the single step (default: first listed channel video)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Pause before each step")
    parser.add_argument("--label", default="", help="Free-text tag for comparing runs, e.g. 'long-videos'")
    parser.add_argument("--poll-interval", type=float, default=5, help="Seconds between job polls")
    parser.add_argument("--timeout", type=int, default=1800, help="Max seconds to wait for the job")
    parser.add_argument("--base-url", default=os.getenv("YOUTUBE_TRANSCRIPT_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("YOUTUBE_TRANSCRIPT_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set YOUTUBE_TRANSCRIPT_API_KEY or pass --api-key")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_channel = "".join(c if c.isalnum() else "_" for c in args.channel).strip("_")
    run_dir = RESULTS_DIR / f"{stamp}_{safe_channel}_{args.max_videos}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
    )
    log.info(f"API: {args.base_url} | channel: {args.channel} | max videos: {args.max_videos}")
    log.info(f"Results: {run_dir}")

    runner = Runner(args, run_dir)
    start = time.time()
    try:
        runner.run()
    except (StopRun, KeyboardInterrupt):
        log.warning("Stopped by user")
        runner.report["stopped_by_user"] = True
        if getattr(runner, "job_id", None):
            log.warning(f"Job {runner.job_id} may still be running; cancel with POST /api/v1/jobs/{runner.job_id}/cancel")
    finally:
        runner.report["total_seconds"] = round(time.time() - start, 2)
        save_results(runner.report, run_dir)
        print_summary(runner.report)
        log.info(f"Saved {run_dir / 'report.json'} and appended to {RESULTS_DIR / 'runs.csv'}")


if __name__ == "__main__":
    main()
