"""Fast validation for B1 model-revision resolution."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from benchmarks.beir.b1 import resolve_model_revision, wall_clock_measurement


class FakeModelInfoClient:
    """In-memory Hub client for resolver tests without network access."""

    def __init__(self, sha: str = "a" * 40) -> None:
        self.sha = sha
        self.requests: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
        self.requests.append((repo_id, revision))
        return SimpleNamespace(sha=self.sha)


def test_resolve_model_revision_uses_main_when_revision_is_omitted() -> None:
    """An unpinned request is resolved and recorded as an immutable SHA."""
    client = FakeModelInfoClient()

    resolved = resolve_model_revision("organisation/model", None, client=client)

    assert resolved == "a" * 40
    assert client.requests == [("organisation/model", "main")]


def test_resolve_model_revision_validates_requested_revision_for_its_model() -> None:
    """The Hub receives the exact model/revision pair supplied by the caller."""
    client = FakeModelInfoClient(sha="b" * 40)

    resolved = resolve_model_revision(
        "organisation/model", "release-tag", client=client
    )

    assert resolved == "b" * 40
    assert client.requests == [("organisation/model", "release-tag")]


def test_resolve_model_revision_rejects_an_unavailable_revision() -> None:
    """A mismatched model revision fails before any inference begins."""

    class FailingClient:
        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            del repo_id, revision
            raise RuntimeError("not found")

    with pytest.raises(ValueError, match="model revision is unavailable"):
        resolve_model_revision(
            "organisation/model", "wrong-sha", client=FailingClient()
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wall_clock_measurement_is_valid_for_cpu_and_gpu_runs(device: str) -> None:
    """Both device modes persist comparable, timezone-aware elapsed-time metadata."""
    started_at = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(seconds=2.5)

    timing = wall_clock_measurement(
        device=device,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_ms=2_500.0,
    )

    assert timing == {
        "device": device,
        "started_at_utc": "2026-08-19T12:00:00+00:00",
        "completed_at_utc": "2026-08-19T12:00:02.500000+00:00",
        "elapsed_ms": 2_500.0,
    }


def test_wall_clock_measurement_rejects_invalid_duration() -> None:
    """Invalid timing data cannot be persisted as a benchmark result."""
    now = datetime(2026, 8, 19, tzinfo=UTC)

    with pytest.raises(ValueError, match="elapsed_ms"):
        wall_clock_measurement(
            device="cpu", started_at=now, completed_at=now, elapsed_ms=-0.1
        )
