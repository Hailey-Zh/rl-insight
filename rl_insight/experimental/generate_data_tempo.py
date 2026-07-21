#!/usr/bin/env python3

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

"""Generate trajectory data and report it to Tempo via ``trace_trajectory``.

A **Tempo-only launcher**: it reuses ``generate_data``'s event generation
(``generate`` / ``stream``) unchanged and forwards every event straight to
``trace_trajectory``, whose internal builder uses ``TempoSampleRecord``. This
script writes no local files and never touches ``FileSampleRecord`` /
``SampleRecord`` -- for the file + HTML timeline path use ``generate_data.py``.

The Tempo path activates only when ``RL_INSIGHT_SERVER_URL`` is set (then
``init()`` runs); otherwise ``trace_trajectory`` is a no-op (a dry run).
``finish()`` always runs, on both the normal and the exception path.

Usage::

    export RL_INSIGHT_SERVER_URL=http://<hub>:18080
    python generate_data_tempo.py --samples 8 --project demo
    python generate_data_tempo.py --stream
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path


_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

from rl_insight import finish, init, trace_trajectory  # noqa: E402
from rl_insight.experimental.generate_data import generate, stream  # noqa: E402


class _TempoSink:
    """Feed-only target that forwards each event to Tempo.

    ``generate()`` / ``stream()`` drive a single ``feed(event)`` target; this
    forwards straight to ``trace_trajectory`` (whose internal builder uses
    ``TempoSampleRecord``). The event object is passed through, never copied,
    and no local trajectory data is kept.
    """

    def feed(self, event: dict[str, Any]) -> None:
        trace_trajectory(event)


def _run(args: argparse.Namespace) -> None:
    """Generate events into the Tempo sink, managing the monitor lifecycle.

    ``init()`` is called only when ``RL_INSIGHT_SERVER_URL`` is set; ``finish()``
    always runs in a ``finally`` so state is cleaned up on success and on error,
    and the original exception propagates.
    """
    if os.environ.get("RL_INSIGHT_SERVER_URL"):
        init(project=args.project, experiment_name=args.experiment_name)

    sink = _TempoSink()
    try:
        if args.stream:
            stream(sink, args.samples, args.interval, args.seed)
        else:
            generate(sink, args.samples, args.seed)
    finally:
        finish()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory data and report it to Tempo"
    )
    parser.add_argument(
        "--samples", type=int, default=12, help="Number of samples (default: 12)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream data incrementally (step by step with sleeps)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Seconds between events in stream mode (default: 0.3)",
    )
    parser.add_argument(
        "--project", default=None, help="Project label attached to Tempo spans"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment label attached to Tempo spans",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _run(_parse_args(argv))


if __name__ == "__main__":
    main()
