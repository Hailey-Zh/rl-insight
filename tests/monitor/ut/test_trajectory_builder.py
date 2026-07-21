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

"""Unit tests for ``TrajectoryBuilder``, focused on the plan-A change:

``_handle_step`` fills ``step.exit_reason`` from the event's ``finish_reason``
before persisting. Also guards that existing memory/file storage behavior does
not regress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_insight.experimental.builder import TrajectoryBuilder
from rl_insight.experimental.samples.file_sample import FileSampleRecord


def _begin(uid: str = "task-1", **kw: Any) -> dict[str, Any]:
    return {"event": "trajectory_begin", "uid": uid, **kw}


def _step(uid: str = "task-1", **kw: Any) -> dict[str, Any]:
    return {"event": "step", "uid": uid, **kw}


# ---------------------------------------------------------------------------
# Plan-A: step.exit_reason == event finish_reason
# ---------------------------------------------------------------------------


def test_each_step_exit_reason_matches_event_finish_reason() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    # Three non-terminal steps in one trajectory, then a terminal stop.
    builder.feed(_step(finish_reason="tool_calls"))
    builder.feed(_step(finish_reason="max_step_limit"))
    builder.feed(_step(finish_reason="tool_calls"))
    builder.feed(_step(finish_reason="stop"))

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert [s.exit_reason for s in traj.steps] == [
        "tool_calls",
        "max_step_limit",
        "tool_calls",
        "stop",
    ]


def test_length_finish_reason_recorded_on_step() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step(finish_reason="length"))

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert traj.steps[-1].exit_reason == "length"


def test_missing_finish_reason_defaults_to_tool_calls() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step())  # no finish_reason -> builder default "tool_calls"

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert traj.steps[-1].exit_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Terminal vs non-terminal semantics
# ---------------------------------------------------------------------------


def test_stop_finishes_each_trajectory() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step(finish_reason="stop"))
    # A second trajectory in the same session, also terminated by stop.
    builder.feed(_begin(trajectory_index=1))
    builder.feed(_step(finish_reason="stop"))

    sample = builder.get("task-1")
    t0 = sample.get_trajectory(0, 0)
    t1 = sample.get_trajectory(0, 1)
    assert t0.tag.finish_reason == "stop"
    assert t0.tag.status == "success"
    assert t0.steps[-1].done is True
    assert t1.tag.finish_reason == "stop"


def test_length_sets_truncated_status() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step(finish_reason="length"))

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert traj.tag.finish_reason == "length"
    assert traj.tag.status == "truncated"
    assert traj.steps[-1].done is True


def test_max_step_limit_is_not_builder_terminal() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step(finish_reason="max_step_limit"))
    # Cursor stayed on the same trajectory: the next step appends here.
    builder.feed(_step(finish_reason="tool_calls"))

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert traj.num_turns == 2  # both steps landed in the same trajectory
    # finish_trajectory was NOT called: no finish_reason on the tag, and the
    # max_step_limit step was never marked done.
    assert traj.tag.finish_reason == ""
    assert traj.tag.status == "success"
    assert traj.steps[0].exit_reason == "max_step_limit"
    assert traj.steps[0].done is False


def test_done_only_set_by_finish_for_terminal_steps() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin())
    builder.feed(_step(finish_reason="tool_calls"))
    builder.feed(_step(finish_reason="tool_calls"))
    builder.feed(_step(finish_reason="stop"))

    traj = builder.get("task-1").get_trajectory(0, 0)
    assert [s.done for s in traj.steps] == [False, False, True]
    assert traj.is_completed is True


# ---------------------------------------------------------------------------
# Storage structure not regressed
# ---------------------------------------------------------------------------


def test_memory_structure_and_step_indices_intact() -> None:
    builder = TrajectoryBuilder()
    builder.feed(_begin(prompt_len=7))
    builder.feed(
        _step(
            finish_reason="tool_calls",
            thought="think",
            assistant_msg={"content": "hi"},
            tool_results=[{"name": "search", "status": "ok"}],
        )
    )
    builder.feed(_step(finish_reason="stop", assistant_msg={"content": "done"}))

    sample = builder.get("task-1")
    traj = sample.get_trajectory(0, 0)
    assert traj.num_turns == 2
    assert [s.step_idx for s in traj.steps] == [1, 2]  # auto-incremented
    assert traj.steps[0].thought == "think"
    assert traj.steps[0].response == "hi"
    assert traj.steps[0].tool_results[0].name == "search"
    assert traj.steps[1].response == "done"
    assert traj.prompt_len == 7


def test_file_backed_sink_persists_exit_reason(tmp_path: Path) -> None:
    builder = TrajectoryBuilder(
        sample_factory=lambda uid, si: FileSampleRecord.create(
            str(tmp_path), uid=uid, sample_index=si
        )
    )
    builder.feed(_begin())
    builder.feed(_step(finish_reason="tool_calls"))
    builder.feed(_step(finish_reason="stop"))

    # Read back fresh from disk (no in-memory reuse).
    reopened = FileSampleRecord.open(str(tmp_path), uid="task-1")
    traj = reopened.get_trajectory(0, 0)
    assert [s.exit_reason for s in traj.steps] == ["tool_calls", "stop"]
    assert traj.tag.finish_reason == "stop"
    assert traj.steps[-1].done is True
