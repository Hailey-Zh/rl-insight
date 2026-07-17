"""Common trajectory CRUD interface.

``BaseSample`` is a ``Protocol`` -- any object with these six methods
satisfies the interface without explicit inheritance. This allows both
``SampleRecord`` (Pydantic model) and ``FileSampleRecord`` (plain class)
to be used interchangeably in type-annotated code.

Protocol (structural subtyping) avoids metaclass conflicts that would
occur if ``SampleRecord`` inherited from both ``ABC`` and ``BaseModel``.

Interface::

    class BaseSample(Protocol):
        def new_trajectory(session_index, **kwargs) -> TrajectoryRecord
        def get_trajectory(session_index, trajectory_index) -> TrajectoryRecord | None
        def add_step(session_index, trajectory_index, step)
        def finish_trajectory(session_index, trajectory_index, exit_reason, status)
        def set_trajectory_reward(session_index, trajectory_index, score, extra_info)
        def set_trajectory_token_data(session_index, trajectory_index, ...)

Usage::

    def process(sample: BaseSample) -> None:
        sample.new_trajectory(0)
        sample.add_step(0, 0, Step(...))
        sample.finish_trajectory(0, 0, "stop")

    process(SampleRecord.create(uid="x"))            # in-memory
    process(FileSampleRecord.create("/tmp", uid="y"))  # filesystem
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseSample(Protocol):
    """Common interface for managing trajectories within a sample.

    Implementations::

        SampleRecord      -- in-memory Pydantic model
        FileSampleRecord  -- filesystem-backed (one JSON file per trajectory)
    """

    # ------------------------------------------------------------------
    # Trajectory lifecycle
    # ------------------------------------------------------------------

    def new_trajectory(self, session_index: int = 0, **kwargs: Any) -> Any:
        """Create and return a new trajectory in the given session."""
        ...

    def get_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> Any:
        """Return the trajectory at ``(session, trajectory)``, or None."""
        ...

    def add_step(
        self, session_index: int, trajectory_index: int, step: Any
    ) -> None:
        """Append a step to a trajectory."""
        ...

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: str = "success",
    ) -> None:
        """Mark a trajectory as done."""
        ...

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """Set reward for a trajectory."""
        ...

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
        """Set token-level data for a trajectory."""
        ...
