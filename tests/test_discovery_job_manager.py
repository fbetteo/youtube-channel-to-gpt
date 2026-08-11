import asyncio
import json
from contextlib import asynccontextmanager

from db_youtube_transcripts import discovery_job_manager as discovery_module
from db_youtube_transcripts.discovery_job_manager import DiscoveryJobManager


class FakeConnection:
    def __init__(self, record=None):
        self.record = record
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append(("execute", query, args))
        return "UPDATE 1"

    async def fetchrow(self, query, *args):
        self.calls.append(("fetchrow", query, args))
        return self.record


def connection_context(connection):
    @asynccontextmanager
    async def context():
        yield connection

    return context


def test_discovery_job_round_trip_uses_shared_json_payload(monkeypatch):
    connection = FakeConnection(
        {
            "payload": json.dumps(
                {"job_id": "job-id", "status": "processing", "videos": []}
            ),
            "status": "completed",
            "error_message": None,
        }
    )
    monkeypatch.setattr(
        discovery_module, "get_db_transaction", connection_context(connection)
    )

    job = asyncio.run(DiscoveryJobManager.get_job("job-id", 6))

    assert job == {"job_id": "job-id", "status": "completed", "videos": []}
    assert "make_interval(mins => $2::int)" in connection.calls[0][1]
    assert connection.calls[-1][2] == ("job-id",)


def test_discovery_updates_do_not_revive_terminal_jobs(monkeypatch):
    connection = FakeConnection(
        {
            "payload": {"status": "completed", "videos": [{"id": "video"}]},
            "status": "completed",
            "error_message": None,
        }
    )
    monkeypatch.setattr(
        discovery_module, "get_db_connection", connection_context(connection)
    )

    result = asyncio.run(
        DiscoveryJobManager.update_job(
            "job-id", status="completed", videos=[{"id": "video"}]
        )
    )

    assert result["status"] == "completed"
    assert "status NOT IN ('completed', 'completed_with_errors', 'failed')" in (
        connection.calls[0][1]
    )
