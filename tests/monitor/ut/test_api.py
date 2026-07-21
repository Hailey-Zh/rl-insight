# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the public monitor API."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from rl_insight import api
from rl_insight.utils.constants import MonitorEventKind


class RecordingClient:
    """Small client double that preserves every submitted event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def apply_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def reset_monitor_state() -> Generator[None, None, None]:
    api.finish()
    yield
    api.finish()


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    client = RecordingClient()
    monkeypatch.setattr(api, "create_monitor_client", lambda _conf: client)
    api.init(
        project="project-a",
        experiment_name="experiment-a",
        config={"server": {"url": "http://monitor:18080"}},
    )
    return client


def test_init_should_enable_monitoring_when_server_and_client_are_available(
    recording_client: RecordingClient,
) -> None:
    assert api._STATE.enabled is True
    assert api._STATE.client is recording_client
    assert api._STATE.labels == {
        "project": "project-a",
        "experiment_name": "experiment-a",
    }


def test_init_should_leave_monitoring_disabled_when_server_url_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_called = False

    def create_client(_conf: Any) -> RecordingClient:
        nonlocal factory_called
        factory_called = True
        return RecordingClient()

    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.setattr(api, "create_monitor_client", create_client)

    api.init(config={"server": {"url": ""}})

    assert api._STATE.enabled is False
    assert factory_called is False


def test_metric_helpers_should_emit_typed_events_when_monitoring_is_enabled(
    recording_client: RecordingClient,
) -> None:
    api.metric_count("steps", amount=2, worker="w0")
    api.metric_gauge("reward", value=1.5, documentation="Latest reward", worker="w0")
    api.metric_histogram("latency", value=12, worker="w0")

    assert [event["kind"] for event in recording_client.events] == [
        MonitorEventKind.COUNTER,
        MonitorEventKind.GAUGE,
        MonitorEventKind.HISTOGRAM,
    ]
    assert [event["value"] for event in recording_client.events] == [2.0, 1.5, 12.0]
    assert recording_client.events[0]["documentation"] == "Counter steps"
    assert recording_client.events[1]["documentation"] == "Latest reward"
    assert recording_client.events[2]["documentation"] == "Histogram latency"
    assert recording_client.events[0]["labels"] == {
        "project": "project-a",
        "experiment_name": "experiment-a",
        "worker": "w0",
    }


def test_trace_state_should_merge_same_state_and_ignore_shadow_when_lane_is_busy(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([100, 200])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    with api.trace_state("rollout", state_lane_id="replica-0", step=3):
        with api.trace_state("rollout", state_lane_id="replica-0"):
            pass
        with api.trace_state("shadowed", state_lane_id="replica-0"):
            pass

    assert len(recording_client.events) == 1
    event = recording_client.events[0]
    assert event["kind"] == MonitorEventKind.TRACE
    assert event["name"] == "rollout"
    assert (event["start_time_ns"], event["end_time_ns"]) == (100, 200)
    assert event["attributes"] == {
        "process_id": api._STATE.process_id,
        "project": "project-a",
        "experiment_name": "experiment-a",
        "step": 3,
        "monitor.trace_segment": "state_interval",
        "state_name": "rollout",
        "state_lane_id": "replica-0",
    }


def test_trace_op_should_report_duration_when_wrapped_function_raises(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([300, 450])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    @api.trace_op("train", extra_labels=lambda item: {"worker": item}, phase="update")
    def train(worker: str) -> None:
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        train("w1")

    event = recording_client.events[0]
    assert event["name"] == "train"
    assert (event["start_time_ns"], event["end_time_ns"]) == (300, 450)
    assert event["attributes"]["worker"] == "w1"
    assert event["attributes"]["phase"] == "update"
    assert event["attributes"]["monitor.trace_segment"] == "duration"


def test_finish_should_disable_future_events_when_monitoring_was_enabled(
    recording_client: RecordingClient,
) -> None:
    api.finish()
    api.metric_count("ignored")

    assert recording_client.events == []
    assert api._STATE.enabled is False


# ---------------------------------------------------------------------------
# trace_trajectory
# ---------------------------------------------------------------------------


def test_trace_trajectory_builds_lazily_and_reuses_the_builder() -> None:
    assert api._TRACE_BUILDER is None

    api.trace_trajectory({"event": "trajectory_begin", "uid": "u"})
    builder = api._TRACE_BUILDER
    assert builder is not None

    api.trace_trajectory({"event": "step", "uid": "u", "finish_reason": "stop"})
    assert api._TRACE_BUILDER is builder  # reused, not rebuilt


def test_trace_trajectory_passes_the_event_object_through_without_copying() -> None:
    class FakeBuilder:
        def __init__(self) -> None:
            self.fed: list[dict[str, Any]] = []

        def feed(self, event: dict[str, Any]) -> None:
            self.fed.append(event)

    fake = FakeBuilder()
    api._TRACE_BUILDER = fake  # type: ignore[assignment]

    event = {"event": "step", "uid": "u"}
    api.trace_trajectory(event)

    assert fake.fed[0] is event  # same object, not a copy


def test_trace_trajectory_emits_one_trace_span_per_step_on_a_shared_lane(
    recording_client: RecordingClient,
) -> None:
    # Timing arithmetic is covered exhaustively by the tempo_sample unit tests
    # (injected FakeClock); here we assert clock-independent invariants only.
    api.trace_trajectory({"event": "trajectory_begin", "uid": "task-1"})
    api.trace_trajectory(
        {
            "event": "step",
            "uid": "task-1",
            "step_index": 1,
            "finish_reason": "tool_calls",
            "thought": "t1",
            "tool_results": [{"name": "search"}],
        }
    )
    api.trace_trajectory(
        {
            "event": "step",
            "uid": "task-1",
            "step_index": 2,
            "finish_reason": "stop",
            "assistant_msg": {"content": "answer"},
        }
    )

    assert [e["kind"] for e in recording_client.events] == [
        MonitorEventKind.TRACE,
        MonitorEventKind.TRACE,
    ]

    first, second = recording_client.events
    assert first["name"] == "tool_calls"
    assert first["attributes"] == {
        "process_id": api._STATE.process_id,
        "project": "project-a",
        "experiment_name": "experiment-a",
        "monitor.trace_segment": "state_interval",
        "monitor.trace_source": "trajectory",
        "state_name": "tool_calls",
        "state_lane_id": "uid=task-1/sample=0/session=0/traj=0",
        "uid": "task-1",
        "sample": "0",
        "session": "0",
        "traj": "0",
        "turn": "1",
        "type": "tool",
        "tools": '["search"]',
        "finish_reason": "tool_calls",
        "content": "t1",
        "trajectory.timing_source": "receive_time",
    }

    assert second["name"] == "stop"
    assert second["attributes"]["type"] == "llm"
    assert second["attributes"]["tools"] == "[]"
    assert second["attributes"]["content"] == "answer"
    assert second["attributes"]["turn"] == "2"
    # Both steps share the same trajectory lane.
    assert first["attributes"]["state_lane_id"] == second["attributes"]["state_lane_id"]
    # Each span has positive duration; the non-terminal step1 hands its end
    # boundary to step2 (lane continuity), independent of the wall clock.
    assert first["start_time_ns"] < first["end_time_ns"]
    assert second["start_time_ns"] == first["end_time_ns"]
    assert second["start_time_ns"] < second["end_time_ns"]


def test_trace_trajectory_is_safe_when_monitoring_is_off() -> None:
    # No init(): monitoring disabled. Calls must not raise or emit.
    api.trace_trajectory({"event": "trajectory_begin", "uid": "u"})
    api.trace_trajectory({"event": "step", "uid": "u", "finish_reason": "stop"})

    assert api._STATE.enabled is False
    assert api._TRACE_BUILDER is not None  # built, but every emit was a no-op


def test_finish_resets_the_trace_builder() -> None:
    api.trace_trajectory({"event": "trajectory_begin", "uid": "u"})
    first = api._TRACE_BUILDER
    assert first is not None

    api.finish()
    assert api._TRACE_BUILDER is None

    api.trace_trajectory({"event": "trajectory_begin", "uid": "u"})
    assert api._TRACE_BUILDER is not None
    assert api._TRACE_BUILDER is not first  # fresh builder, no carried-over state


def test_trace_trajectory_propagates_builder_errors() -> None:
    with pytest.raises(ValueError):
        api.trace_trajectory({"event": "bogus", "uid": "u"})


def test_trace_trajectory_propagates_emit_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoomClient:
        def apply_event(self, event: dict[str, Any]) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(api, "create_monitor_client", lambda _conf: BoomClient())
    api.init(config={"server": {"url": "http://monitor:18080"}})

    api.trace_trajectory({"event": "trajectory_begin", "uid": "u"})  # no emit yet
    with pytest.raises(RuntimeError, match="boom"):
        api.trace_trajectory({"event": "step", "uid": "u", "finish_reason": "stop"})
