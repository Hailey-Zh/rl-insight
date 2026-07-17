"""Experiment data structures for RL training analysis.

``SampleRecord`` is the primary data model. ``TrajectoryBuilder`` is the
event-driven adapter that builds samples from ``trajectory_begin`` and
``step`` events.

Hierarchy::

    TrajectoryBuilder      ← ingests events
      └── SampleRecord     ← one RL dataset sample
            └── SessionRecord   ← one GatewaySession (rollout attempt)
                  └── TrajectoryRecord  ← one chain / trajectory
                        └── Step        ← one model-call + tool-execution cycle
                              └── ToolResult  ← one tool call result

Quick start::

    from rl_insight.experimental import TrajectoryBuilder

    # Build from events
    builder = TrajectoryBuilder()
    builder.feed({"event": "trajectory_begin", "uid": "...", ...})
    builder.feed({"event": "step", "uid": "...", ...})
    samples = builder.samples

    # Or load a JSONL directly
    builder = TrajectoryBuilder.from_jsonl("events.jsonl")

    # Query the built sample
    sample = builder.get("uid")
    traj = sample.get_trajectory(0, 0)
    traj.num_turns           # int
    traj.total_tool_calls    # int
"""

from rl_insight.experimental.base import BaseSample
from rl_insight.experimental.builder import TrajectoryBuilder
from rl_insight.experimental.file_sample import FileSampleRecord
from rl_insight.experimental.sample import (
    SampleRecord,
    SampleTag,
    SessionRecord,
    SessionTag,
    Step,
    ToolResult,
    ToolStatus,
    TrajectoryRecord,
    TrajectoryTag,
    TrainingStatus,
)

__all__ = [
    "BaseSample",
    "TrajectoryBuilder",
    "FileSampleRecord",
    "SampleRecord",
    "SampleTag",
    "SessionRecord",
    "SessionTag",
    "Step",
    "ToolResult",
    "ToolStatus",
    "TrajectoryRecord",
    "TrajectoryTag",
    "TrainingStatus",
]
