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

"""Unit tests for ``TempoSampleRecord`` (emit-through trajectory spans)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rl_insight.experimental.samples.base import BaseSample
from rl_insight.experimental.samples.sample import Step, ToolResult
from rl_insight.experimental.samples.tempo_sample import TempoSampleRecord


class SpanRecorder:
    """Emit double: captures every span emitted, keyword args as dicts."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: dict[str, Any],
    ) -> None:
        self.spans.append(
            {
                "name": name,
                "start_time_ns": start_time_ns,
                "end_time_ns": end_time_ns,
                "attributes": attributes,
            }
        )


class FakeClock:
    """Deterministic clock returning queued values, then holding the last."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)
        self._last = values[0] if values else 0

    def __call__(self) -> int:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def _sample(recorder: SpanRecorder, clock: FakeClock, **kw: Any) -> TempoSampleRecord:
    return TempoSampleRecord(
        uid=kw.pop("uid", "task-1"),
        sample_index=kw.pop("sample_index", 0),
        emit_span=recorder,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Span contract
# ---------------------------------------------------------------------------


def test_add_step_emits_full_span_contract_without_mutating_step() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([100, 250])  # new_trajectory=100, add_step now=250
    sample = _sample(recorder, clock, uid="task-1", sample_index=3)

    sample.new_trajectory(session_index=2, trajectory_index=1)
    step = Step(
        step_idx=5,
        thought="thinking hard",
        response="the answer",
        tool_results=[ToolResult(name="search"), ToolResult(name="calc")],
        exit_reason="tool_calls",
    )
    before = step.model_dump()

    sample.add_step(2, 1, step)

    assert step.model_dump() == before  # step untouched

    assert len(recorder.spans) == 1
    span = recorder.spans[0]
    assert span["name"] == "tool_calls"
    assert span["start_time_ns"] == 100
    assert span["end_time_ns"] == 250

    attrs = span["attributes"]
    assert attrs == {
        "monitor.trace_segment": "state_interval",
        "monitor.trace_source": "trajectory",
        "state_name": "tool_calls",
        "state_lane_id": "uid=task-1/sample=3/session=2/traj=1",
        "uid": "task-1",
        "sample": "3",
        "session": "2",
        "traj": "1",
        "turn": "5",  # turn == step.step_idx, not auto-incremented
        "type": "tool",  # has tool_results
        "tools": json.dumps(["search", "calc"]),
        "finish_reason": "tool_calls",
        "content": "thinking hard",  # thought preferred over response
        "trajectory.timing_source": "receive_time",
    }


def test_content_falls_back_to_response_and_truncates_to_500_llm_type() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0, 10])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    long_response = "x" * 900
    step = Step(step_idx=1, thought="", response=long_response, exit_reason="stop")

    sample.add_step(0, 0, step)

    attrs = recorder.spans[0]["attributes"]
    assert attrs["content"] == "x" * 500  # truncated, response used (no thought)
    assert attrs["type"] == "llm"  # no tool_results
    assert attrs["tools"] == "[]"  # empty tool list


def test_span_name_is_taken_from_step_exit_reason() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0, 1])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="length"))

    assert recorder.spans[0]["name"] == "length"
    assert recorder.spans[0]["attributes"]["state_name"] == "length"


# ---------------------------------------------------------------------------
# Boundary / monotonicity
# ---------------------------------------------------------------------------


def test_end_bumped_when_clock_does_not_advance_and_intervals_stay_monotonic() -> None:
    recorder = SpanRecorder()
    # new_trajectory=100, then three non-terminal steps with a frozen clock.
    clock = FakeClock([100, 100, 100, 100])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    for _ in range(3):
        sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))

    spans = recorder.spans
    # start_0 = 100, each end forced to start+1, next start = previous end.
    assert (spans[0]["start_time_ns"], spans[0]["end_time_ns"]) == (100, 101)
    assert (spans[1]["start_time_ns"], spans[1]["end_time_ns"]) == (101, 102)
    assert (spans[2]["start_time_ns"], spans[2]["end_time_ns"]) == (102, 103)


def test_end_uses_clock_when_it_advances() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([100, 500])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))

    assert recorder.spans[0]["end_time_ns"] == 500


def test_end_bumped_when_clock_goes_backwards() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([100, 40])  # clock regresses below start
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))

    assert recorder.spans[0]["end_time_ns"] == 101  # start + 1


def test_max_step_limit_emits_span_but_retains_boundary() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([100, 200, 300])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="max_step_limit"))
    # boundary retained at end=200; next step continues from there.
    sample.add_step(0, 0, Step(step_idx=2, exit_reason="tool_calls"))

    assert recorder.spans[0]["end_time_ns"] == 200
    assert recorder.spans[1]["start_time_ns"] == 200
    assert recorder.spans[1]["end_time_ns"] == 300


# ---------------------------------------------------------------------------
# Isolation between trajectories
# ---------------------------------------------------------------------------


def test_boundaries_isolated_across_session_and_trajectory() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([10, 20, 111, 222])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)  # boundary=10
    sample.new_trajectory(session_index=1, trajectory_index=0)  # boundary=20

    sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))  # now=111
    sample.add_step(1, 0, Step(step_idx=1, exit_reason="tool_calls"))  # now=222

    assert recorder.spans[0]["start_time_ns"] == 10
    assert recorder.spans[0]["end_time_ns"] == 111
    assert recorder.spans[1]["start_time_ns"] == 20
    assert recorder.spans[1]["end_time_ns"] == 222


# ---------------------------------------------------------------------------
# Terminal handling / errors
# ---------------------------------------------------------------------------


def test_stop_drops_boundary_then_add_step_raises_keyerror() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0, 5])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="stop"))

    with pytest.raises(KeyError):
        sample.add_step(0, 0, Step(step_idx=2, exit_reason="tool_calls"))


def test_length_drops_boundary_then_add_step_raises_keyerror() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0, 5])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="length"))

    with pytest.raises(KeyError):
        sample.add_step(0, 0, Step(step_idx=2, exit_reason="tool_calls"))


def test_add_step_without_new_trajectory_raises_keyerror() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0])
    sample = _sample(recorder, clock)

    with pytest.raises(KeyError):
        sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))
    assert recorder.spans == []


# ---------------------------------------------------------------------------
# Emit failure
# ---------------------------------------------------------------------------


def test_emit_failure_propagates_and_leaves_boundary_intact() -> None:
    calls: list[int] = []

    def failing_emit(**_: Any) -> None:
        calls.append(1)
        raise RuntimeError("backend down")

    clock = FakeClock([100, 200, 400])
    sample = TempoSampleRecord(uid="task-1", emit_span=failing_emit, clock=clock)
    sample.new_trajectory(session_index=0, trajectory_index=0)

    with pytest.raises(RuntimeError, match="backend down"):
        sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))

    # Boundary was neither advanced nor dropped: it is still the original 100,
    # so a retry starts from 100 (not from a bumped value), and no KeyError.
    recorded: list[dict[str, Any]] = []

    def ok_emit(**kw: Any) -> None:
        recorded.append(kw)

    sample._emit_span = ok_emit  # swap in a working backend
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="tool_calls"))

    assert recorded[0]["start_time_ns"] == 100
    assert recorded[0]["end_time_ns"] == 400


# ---------------------------------------------------------------------------
# finish_trajectory
# ---------------------------------------------------------------------------


def test_finish_trajectory_is_idempotent_and_emits_no_span() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0, 5])
    sample = _sample(recorder, clock)

    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1, exit_reason="stop"))
    assert len(recorder.spans) == 1

    # Called after a terminal step (boundary already gone) and again: no-op.
    sample.finish_trajectory(0, 0, "stop")
    sample.finish_trajectory(0, 0, "stop")

    assert len(recorder.spans) == 1  # no extra span


def test_finish_trajectory_on_unknown_key_does_not_raise() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0])
    sample = _sample(recorder, clock)

    sample.finish_trajectory(9, 9, "finished")  # never opened
    assert recorder.spans == []


# ---------------------------------------------------------------------------
# Return type contract (builder integration relies on this)
# ---------------------------------------------------------------------------


def test_tempo_sample_satisfies_base_sample_protocol() -> None:
    # D3 core claim: TempoSampleRecord IS a BaseSample (structural), which the
    # builder factory in Step 3 relies on entirely.
    sample = _sample(SpanRecorder(), FakeClock([0]))
    assert isinstance(sample, BaseSample)


def test_new_trajectory_returns_record_with_settable_prompt_len() -> None:
    recorder = SpanRecorder()
    clock = FakeClock([0])
    sample = _sample(recorder, clock, uid="task-9", sample_index=2)

    traj = sample.new_trajectory(session_index=1, trajectory_index=3)

    # The builder's _handle_trajectory_begin sets these; must not raise.
    traj.prompt_len = 42
    traj.tag.prompt_len = 42
    assert traj.uid == "task-9"
    assert traj.sample_index == 2
    assert traj.session_index == 1
    assert traj.trajectory_index == 3
