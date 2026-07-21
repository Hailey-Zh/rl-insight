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

"""Unit tests for the Tempo-only launcher ``generate_data_tempo``.

Covers the single-sink forward (each event -> trace_trajectory, same object)
and the monitor lifecycle in ``_run`` (init gated on RL_INSIGHT_SERVER_URL,
finish always called, exceptions propagate).
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from rl_insight.experimental import generate_data_tempo as gdt


def _args(**overrides: Any) -> argparse.Namespace:
    base = dict(
        samples=3,
        seed=7,
        stream=False,
        interval=0.0,
        project=None,
        experiment_name=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Tempo sink
# ---------------------------------------------------------------------------


def test_tempo_sink_forwards_event_to_trace_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reported: list[dict[str, Any]] = []
    monkeypatch.setattr(gdt, "trace_trajectory", lambda e: reported.append(e))

    event = {"event": "step", "uid": "u"}
    gdt._TempoSink().feed(event)

    assert reported == [event]
    assert reported[0] is event  # same object, not copied


def test_generate_reports_every_event_to_tempo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real generate() from generate_data drives the sink end to end.
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    reported: list[dict[str, Any]] = []
    monkeypatch.setattr(gdt, "trace_trajectory", lambda e: reported.append(e))
    monkeypatch.setattr(gdt, "finish", lambda: None)

    gdt._run(_args(samples=2, seed=1))

    assert len(reported) > 0
    assert all(e.get("event") in ("trajectory_begin", "step") for e in reported)


# ---------------------------------------------------------------------------
# Monitor lifecycle in _run
# ---------------------------------------------------------------------------


def test_run_calls_init_when_server_url_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "http://monitor:18080")
    monkeypatch.setattr(gdt, "init", lambda **kw: events.append(("init", kw)))
    monkeypatch.setattr(gdt, "finish", lambda: events.append(("finish",)))
    monkeypatch.setattr(gdt, "generate", lambda b, n, s: events.append(("generate",)))

    gdt._run(_args(project="p", experiment_name="e"))

    assert events == [
        ("init", {"project": "p", "experiment_name": "e"}),
        ("generate",),
        ("finish",),
    ]


def test_run_skips_init_when_server_url_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    events: list[str] = []
    monkeypatch.setattr(gdt, "init", lambda **kw: events.append("init"))
    monkeypatch.setattr(gdt, "finish", lambda: events.append("finish"))
    monkeypatch.setattr(gdt, "generate", lambda b, n, s: events.append("generate"))

    gdt._run(_args())

    assert "init" not in events
    assert events == ["generate", "finish"]  # finish still runs


def test_run_uses_stream_when_stream_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    called: list[Any] = []
    monkeypatch.setattr(gdt, "finish", lambda: None)
    monkeypatch.setattr(
        gdt, "stream", lambda b, n, i, s: called.append(("stream", n, i, s))
    )
    monkeypatch.setattr(gdt, "generate", lambda *a: called.append(("generate",)))

    gdt._run(_args(stream=True, samples=5, interval=0.1, seed=9))

    assert called == [("stream", 5, 0.1, 9)]


def test_run_calls_finish_on_exception_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    finished: list[bool] = []
    monkeypatch.setattr(gdt, "finish", lambda: finished.append(True))

    def boom(b: Any, n: int, s: int) -> None:
        raise RuntimeError("gen boom")

    monkeypatch.setattr(gdt, "generate", boom)

    with pytest.raises(RuntimeError, match="gen boom"):
        gdt._run(_args())

    assert finished == [True]  # finish ran despite the exception
