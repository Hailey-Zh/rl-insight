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

"""Emit-through sample that turns each trajectory step into a trace span.

``TempoSampleRecord`` implements ``BaseSample`` (the six-method CRUD
interface) but keeps **no local copy** of the trajectory data. The only
state it holds is the per-``(session, trajectory)`` receive-time boundary
used to bound each step's span. Everything else is forwarded straight to
an injected ``emit_span`` callable (in production ``api._emit_trace_span``,
which sends an OTLP root span to Tempo via the monitor hub).

One step becomes one independent root span:

- ``name``           = ``step.exit_reason`` (the finish_reason for that step)
- ``start_time_ns``  = previous receive-time boundary for this trajectory
- ``end_time_ns``    = now, forced strictly greater than start (bump 1ns)
- ``attributes``     = the 14-field span contract (see ``_build_span_attributes``)

Boundary lifecycle inside ``add_step``:

- terminal (``stop`` / ``length``) → span emitted, boundary dropped; a later
  ``add_step`` on that key raises ``KeyError``.
- non-terminal (``tool_calls`` / ``max_step_limit`` / ...) → span emitted,
  boundary advanced to this span's end so the next step continues from it.

The emit happens *before* the boundary is mutated, so a failing
``emit_span`` propagates and leaves the boundary untouched.

Implements ``BaseSample`` (Protocol) -- same trajectory CRUD methods as
``SampleRecord`` and ``FileSampleRecord``.

Usage::

    from rl_insight.experimental.samples.tempo_sample import TempoSampleRecord

    sample = TempoSampleRecord(uid="task-1", sample_index=0, emit_span=emit)
    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(exit_reason="tool_calls", tool_results=[...]))
    sample.add_step(0, 0, Step(exit_reason="stop"))  # emits + drops boundary
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from rl_insight.experimental.samples.sample import (
    Step,
    TrajectoryRecord,
    TrainingStatus,
)

# emit_span(name, start_time_ns, end_time_ns, attributes) -> None
SpanEmitter = Callable[..., None]

# (session_index, trajectory_index)
TrajectoryKey = tuple[int, int]


class TempoSampleRecord:
    """Emit-through ``BaseSample``: each step becomes one trace span.

    Holds only the per-``(session, trajectory)`` receive-time boundary; no
    trajectory, step, reward, or token data is retained locally. ``uid`` and
    ``sample_index`` are fixed for the life of the instance (one instance per
    ``uid``, mirroring how the builder factory creates samples).
    """

    def __init__(
        self,
        *,
        uid: str,
        sample_index: int = 0,
        emit_span: SpanEmitter,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        self.uid = uid
        self.sample_index = sample_index
        self._emit_span = emit_span
        self._clock = clock
        self._boundaries: dict[TrajectoryKey, int] = {}

    # ------------------------------------------------------------------
    # Trajectory lifecycle
    # ------------------------------------------------------------------

    def new_trajectory(self, session_index: int = 0, **kwargs: Any) -> TrajectoryRecord:
        """Open a receive-time boundary for a trajectory and return a record.

        The returned ``TrajectoryRecord`` is transient: the builder sets its
        ``prompt_len`` and is otherwise free to inspect it, but this class
        keeps no reference to it (D3: no local copy).
        """
        ti = kwargs.get("trajectory_index", 0)
        if ti is None:
            ti = 0
        self._boundaries[(session_index, ti)] = self._clock()
        return TrajectoryRecord.create(
            uid=self.uid,
            sample_index=self.sample_index,
            session_index=session_index,
            trajectory_index=ti,
        )

    def get_trajectory(self, session_index: int, trajectory_index: int) -> None:
        """Always ``None`` -- no trajectory data is stored locally."""
        return None

    def add_step(self, session_index: int, trajectory_index: int, step: Step) -> None:
        """Emit one root span for ``step`` over its receive-time interval.

        Raises ``KeyError`` if the trajectory has no open boundary (no
        ``new_trajectory`` was called, or it was already terminated). Does not
        mutate ``step``. Advances/drops the boundary only after a successful
        emit.
        """
        key = (session_index, trajectory_index)
        try:
            start_time_ns = self._boundaries[key]
        except KeyError as exc:
            raise KeyError(
                "tempo trace boundary not found for "
                f"uid={self.uid!r}, session={session_index}, "
                f"trajectory={trajectory_index}; call new_trajectory() first"
            ) from exc

        finish_reason = step.exit_reason
        now = self._clock()
        end_time_ns = now if now > start_time_ns else start_time_ns + 1
        attributes = self._build_span_attributes(session_index, trajectory_index, step)

        self._emit_span(
            name=finish_reason,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=attributes,
        )

        # Only reached if emit_span did not raise.
        if finish_reason in ("stop", "length"):
            del self._boundaries[key]
        else:
            self._boundaries[key] = end_time_ns

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        """Idempotent close: drop any lingering boundary, never emit a span.

        The step span is already emitted in ``add_step``; a terminal step
        (``stop`` / ``length``) has already dropped the boundary, so this is
        usually a no-op. Safe to call when no boundary exists.
        """
        self._boundaries.pop((session_index, trajectory_index), None)

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """No-op: reward is not part of the trajectory span contract."""

    def set_trajectory_token_data(
        self,
        session_index: int,
        trajectory_index: int,
        *,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        routed_experts: Any = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> None:
        """No-op: token-level data is not part of the trajectory span contract."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_span_attributes(
        self, session_index: int, trajectory_index: int, step: Step
    ) -> dict[str, Any]:
        """Build the 14-field span attribute contract for one step."""
        finish_reason = step.exit_reason
        tool_names = [result.name for result in step.tool_results]
        content = (step.thought or step.response)[:500]
        lane_id = (
            f"uid={self.uid}/sample={self.sample_index}/"
            f"session={session_index}/traj={trajectory_index}"
        )
        return {
            "monitor.trace_segment": "state_interval",
            "monitor.trace_source": "trajectory",
            "state_name": finish_reason,
            "state_lane_id": lane_id,
            "uid": self.uid,
            "sample": str(self.sample_index),
            "session": str(session_index),
            "traj": str(trajectory_index),
            "turn": str(step.step_idx),
            "type": "tool" if step.tool_results else "llm",
            "tools": json.dumps(tool_names),
            "finish_reason": finish_reason,
            "content": content,
            "trajectory.timing_source": "receive_time",
        }
