# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for torchtitan.observability.structured_logger."""

import asyncio
import json
import logging
import os
import time
from unittest import mock

import pytest
from torchtitan.observability.structured_logger import step_state
from torchtitan.observability.structured_logger.jsonl_handler import (
    MAX_MESSAGE_SIZE,
    register_jsonl_handler,
    TraceJsonlFormatter,
    TraceJsonlHandler,
)
from torchtitan.observability.structured_logger.step_state import (
    add_step_tag,
    clear_step_tags,
    get_relative_step,
    get_step,
    get_step_tags,
    set_step,
)
from torchtitan.observability.structured_logger.structured_logging import (
    _structured_logger,
    event_extra,
    ExtraFields,
    init_structured_logger,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
    LogType,
    TraceEventsOnlyFilter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_context():
    """Reset step state before each test."""
    step_state._STEP_GLOBAL = None
    step_state._STEP_CV.set(None)
    step_state._RELATIVE_STEP_GLOBAL = None
    step_state._RELATIVE_STEP_CV.set(None)
    step_state._TAGS_GLOBAL = ()
    step_state._TAGS_CV.set(())
    yield
    step_state._STEP_GLOBAL = None
    step_state._STEP_CV.set(None)
    step_state._RELATIVE_STEP_GLOBAL = None
    step_state._RELATIVE_STEP_CV.set(None)
    step_state._TAGS_GLOBAL = ()
    step_state._TAGS_CV.set(())


@pytest.fixture
def structured_logger_fixture():
    """Provide a clean structured logger for testing."""
    import torchtitan.observability.structured_logger.structured_logging as sl_mod

    tl = _structured_logger
    orig = (tl.handlers[:], tl.level, tl.propagate)
    # Reset the module-level init sentinel so init_structured_logger re-runs
    # for each test (otherwise the second call short-circuits as "already
    # initialized").
    sl_mod._is_initialized = False
    yield tl
    tl.handlers, tl.level, tl.propagate = orig
    sl_mod._is_initialized = False


# ---------------------------------------------------------------------------
# Step context tests (hybrid ContextVar)
# ---------------------------------------------------------------------------


class TestSetStep:
    def test_set_step_stores_value(self):
        set_step(42)
        assert get_step() == 42

    def test_set_step_overwrites(self):
        set_step(1)
        set_step(2)
        assert get_step() == 2

    def test_step_default_is_none(self):
        assert get_step() is None

    def test_set_step_writes_to_both_global_and_cv(self):
        set_step(10)
        assert step_state._STEP_GLOBAL == 10
        assert step_state._STEP_CV.get() == 10

    def test_get_step_prefers_cv(self):
        """ContextVar takes priority over global."""
        step_state._STEP_GLOBAL = 1
        step_state._STEP_CV.set(2)
        assert get_step() == 2

    def test_get_step_falls_back_to_global(self):
        """If CV is None, falls back to global."""
        step_state._STEP_GLOBAL = 5
        step_state._STEP_CV.set(None)
        assert get_step() == 5


class TestStepTags:
    def test_add_step_tag_adds_tag(self):
        set_step(1)
        add_step_tag("gc")
        assert get_step_tags() == ("gc",)

    def test_add_step_tag_deduplicates(self):
        add_step_tag("gc")
        add_step_tag("gc")
        assert get_step_tags() == ("gc",)

    def test_add_step_tag_multiple_tags(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        add_step_tag("checkpoint")
        assert get_step_tags() == ("gc", "profiling", "checkpoint")

    def test_clear_step_tags_clears_all(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        clear_step_tags()
        assert get_step_tags() == ()

    def test_step_tags_are_tuples(self):
        """Step tags are tuples (immutable) to avoid shared-reference issues."""
        add_step_tag("a")
        tags = get_step_tags()
        assert isinstance(tags, tuple)

    def test_set_step_clears_tags(self):
        add_step_tag("gc")
        set_step(2)
        assert get_step_tags() == ()

    def test_add_step_tag_spmd_writes_global_only(self):
        # SPMD path: no asyncio task context. add_step_tag should write to
        # _TAGS_GLOBAL; _TAGS_CV stays at default (empty tuple).
        add_step_tag("gc")
        add_step_tag("checkpoint")
        assert step_state._TAGS_GLOBAL == ("gc", "checkpoint")
        assert step_state._TAGS_CV.get() == ()
        assert get_step_tags() == ("gc", "checkpoint")  # falls back to global

    def test_get_step_tags_reader_prefers_cv_over_global(self):
        # Invariant: the reader checks CV first (if non-empty) and falls
        # back to the global. Pins the contract so changing it would flip
        # which tag set async-task spans observe.
        step_state._TAGS_GLOBAL = ("gc",)
        step_state._TAGS_CV.set(("eval",))
        assert get_step_tags() == ("eval",)

    def test_add_step_tag_concurrent_async_tasks_are_isolated(self):
        # Async RL scenario: two actor tasks on the same process add
        # DIFFERENT tags for the same step. Each task's CV must be scoped;
        # neither should see the other's tag via the shared global. Also
        # verifies the exclusive-store semantic: a pre-existing global tag
        # is NOT visible inside the async tasks (actor contexts are
        # intentionally isolated; global tags don't bleed in).
        import asyncio

        # SPMD-side write before any async loop runs → goes to global.
        add_step_tag("pre_async_global_tag")
        assert step_state._TAGS_GLOBAL == ("pre_async_global_tag",)

        async def actor_gc():
            await asyncio.sleep(0.01)
            add_step_tag("gc")
            return get_step_tags()

        async def actor_eval():
            await asyncio.sleep(0.02)  # runs after actor_gc's write
            add_step_tag("eval")
            return get_step_tags()

        async def run():
            return await asyncio.gather(actor_gc(), actor_eval())

        gc_view, eval_view = asyncio.run(run())
        # Each actor sees ONLY its own tag -- no "gc" leak into actor_eval,
        # and (importantly) no "pre_async_global_tag" leak from global.
        assert gc_view == ("gc",), f"actor_gc saw {gc_view!r}, expected ('gc',)"
        assert eval_view == ("eval",), (
            f"actor_eval saw {eval_view!r}, expected ('eval',) "
            f"-- if 'gc' leaked in, add_step_tag is not task-isolated"
        )
        # Global is untouched by the tasks' writes.
        assert step_state._TAGS_GLOBAL == ("pre_async_global_tag",)

    def test_clear_step_tags_resets_both_stores(self):
        # clear_step_tags is called by set_step, so the SPMD happy path
        # cycles tags correctly between steps. Verify it also works when
        # both stores have content.
        step_state._TAGS_GLOBAL = ("gc",)
        step_state._TAGS_CV.set(("eval",))
        step_state.clear_step_tags()
        assert step_state._TAGS_GLOBAL == ()
        assert step_state._TAGS_CV.get() == ()
        assert get_step_tags() == ()

    def test_threading_thread_falls_back_to_global(self):
        # Plain threading.Thread does NOT inherit ContextVar state (that's
        # an asyncio-specific copy_context behavior). A reader on such a
        # thread must fall back to the global for anything to be visible.
        import threading

        add_step_tag("from_main")  # SPMD: writes global
        assert step_state._TAGS_GLOBAL == ("from_main",)

        result: list[tuple[str, ...]] = []

        def thread_reader() -> None:
            result.append(get_step_tags())

        t = threading.Thread(target=thread_reader)
        t.start()
        t.join()
        # Thread sees global via fallback (its CV is at default ()).
        assert result == [("from_main",)]


class TestRelativeStep:
    def test_relative_step_set_and_get(self):
        set_step(101, relative_step=1)
        assert get_relative_step() == 1
        assert get_step() == 101

    def test_relative_step_default_none(self):
        assert get_relative_step() is None

    def test_relative_step_defaults_to_step_when_omitted(self):
        """When set_step is called without relative_step, it defaults to step.

        Prevents a stale _RELATIVE_STEP_GLOBAL from leaking across calls
        (e.g. sync_log_step broadcasts that omit the kwarg). Correct for
        non-resumed runs; resumed trainers must pass relative_step
        explicitly.
        """
        set_step(1, relative_step=1)
        set_step(2)  # no relative_step -> defaults to step
        assert get_relative_step() == 2

    def test_relative_step_in_formatter_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(101, relative_step=1)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["relative_step"] == 1


class TestHybridContextVarAsync:
    def test_asyncio_task_inherits_step(self):
        """Asyncio tasks inherit the step from the parent context."""
        results = []

        async def worker():
            results.append(get_step())

        async def main():
            set_step(42)
            task = asyncio.create_task(worker())
            await task

        asyncio.run(main())
        assert results == [42]


# ---------------------------------------------------------------------------
# event_extra
# ---------------------------------------------------------------------------


class TestEventExtra:
    def test_basic_event(self):
        extra = event_extra("fwd_bwd")
        assert extra[str(ExtraFields.LOG_TYPE)] == str(LogType.EVENT)
        assert extra[str(ExtraFields.LOG_TYPE_NAME)] == str("fwd_bwd")

    def test_with_step_and_value(self):
        extra = event_extra("step", step=10, value=42.0)
        assert extra[str(ExtraFields.STEP)] == 10
        assert extra[str(ExtraFields.VALUE)] == 42.0

    def test_with_task_name(self):
        extra = event_extra("fwd_bwd", task_name="worker-0")
        assert extra[str(ExtraFields.TASK_NAME)] == "worker-0"


# ---------------------------------------------------------------------------
# TraceJsonlFormatter
# ---------------------------------------------------------------------------


class TestTraceJsonlFormatter:
    def test_format_produces_flat_json(self):
        """Output is flat JSON (not 4-column structured format)."""
        fmt = TraceJsonlFormatter(rank=0, source="trainer")
        set_step(5)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra(str("fwd_bwd") + "_start", step=5).items():
            setattr(record, k, v)

        output = fmt.format(record)
        parsed = json.loads(output)
        # Flat JSON: fields at top level, no "int"/"normal"/"double" keys
        assert "int" not in parsed
        assert "normal" not in parsed
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"
        assert parsed["step"] == 5

    def test_rank_and_source_from_self(self):
        fmt = TraceJsonlFormatter(rank=3, source="generator")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["rank"] == 3
        assert parsed["source"] == "generator"

    def test_step_from_global(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(42)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["step"] == 42

    def test_task_name_in_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step", task_name="worker-0").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["task_name"] == "worker-0"

    def test_message_truncation(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        long_msg = "x" * (MAX_MESSAGE_SIZE + 100)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_msg,
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "..." in parsed["message"]
        assert len(parsed["message"]) < len(long_msg)

    def test_step_tags_in_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(1)
        add_step_tag("gc")
        add_step_tag("profiling")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert set(parsed["step_tags"]) == {"gc", "profiling"}

    def test_seq_id_increments(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        seq_ids = []
        for i in range(3):
            r = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"msg{i}",
                args=None,
                exc_info=None,
            )
            for k, v in event_extra("step").items():
                setattr(r, k, v)
            parsed = json.loads(fmt.format(r))
            seq_ids.append(parsed["seq_id"])
        assert seq_ids == [0, 1, 2]

    def test_global_rank_in_output(self):
        """global_rank equals rank (needed for per-rank trace track assignment)."""
        fmt = TraceJsonlFormatter(rank=5, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["global_rank"] == 5
        assert parsed["rank"] == 5

    def test_local_rank_in_output(self):
        """local_rank from LOCAL_RANK env var (default 0)."""
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "local_rank" in parsed
        assert isinstance(parsed["local_rank"], int)

    def test_has_time_us(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "time_us" in parsed
        assert isinstance(parsed["time_us"], int)


# ---------------------------------------------------------------------------
# TraceEventsOnlyFilter
# ---------------------------------------------------------------------------


class TestTraceEventsOnlyFilter:
    def test_passes_event_records(self):
        f = TraceEventsOnlyFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        setattr(record, str(ExtraFields.LOG_TYPE_NAME), str("step"))
        assert f.filter(record) is True

    def test_blocks_non_event_records(self):
        f = TraceEventsOnlyFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="just text",
            args=None,
            exc_info=None,
        )
        assert f.filter(record) is False


# ---------------------------------------------------------------------------
# init_structured_logger
# ---------------------------------------------------------------------------


class TestInitStructuredLogger:
    def test_creates_trace_jsonl(self, tmp_path, structured_logger_fixture):
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)

        # Logger should have handlers
        assert any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )

        # File should exist after we log something
        set_step(1)
        structured_logger_fixture.info(
            "test event",
            extra=event_extra("step", step=1),
        )

        structured_logs_dir = os.path.join(output_dir, "structured_logs")
        assert os.path.exists(structured_logs_dir)
        jsonl_files = [
            f for f in os.listdir(structured_logs_dir) if f.endswith(".jsonl")
        ]
        assert len(jsonl_files) == 1

        with open(os.path.join(structured_logs_dir, jsonl_files[0])) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"

    def test_idempotent(self, tmp_path, structured_logger_fixture):
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(structured_logger_fixture.handlers)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        assert len(structured_logger_fixture.handlers) == handler_count

    def test_file_naming_pattern(self, tmp_path, structured_logger_fixture):
        """File name follows: {source}.global_rank_{rank}.{ts}-{rand}.jsonl"""
        init_structured_logger(rank=3, source="generator", output_dir=str(tmp_path))
        structured_logs_dir = os.path.join(str(tmp_path), "structured_logs")

        # Force a log event so file is created
        structured_logger_fixture.info("x", extra=event_extra("step"))

        files = os.listdir(structured_logs_dir)
        assert len(files) == 1
        assert files[0].startswith("generator.global_rank_3.")
        assert files[0].endswith(".jsonl")

    def test_second_call_is_noop(self, tmp_path, structured_logger_fixture):
        """init_structured_logger is idempotent: second call returns early and
        does not attach additional handlers."""
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(structured_logger_fixture.handlers)

        init_structured_logger(rank=0, source="other_actor", output_dir=output_dir)

        assert len(structured_logger_fixture.handlers) == handler_count


class TestFactoryMechanism:
    def test_default_creates_jsonl(self, tmp_path, structured_logger_fixture):
        """When TITAN_STRUCT_LOGGER_HANDLERS is unset, default JSONL factory runs."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TITAN_STRUCT_LOGGER_HANDLERS", None)
            init_structured_logger(rank=0, source="test", output_dir=str(tmp_path))
        assert any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )

    def test_custom_env_replaces_default(self, tmp_path, structured_logger_fixture):
        """When TITAN_STRUCT_LOGGER_HANDLERS is set, only specified factories run."""
        called = []

        def fake_factory(*, structured_logger, rank, source, output_dir, **kw):
            called.append((rank, source))

        with mock.patch.dict(
            os.environ,
            {
                "TITAN_STRUCT_LOGGER_HANDLERS": "tests.unit_tests.observability.test_structured_logging.fake_factory"
            },
        ):
            # Patch importlib to load our fake factory
            with mock.patch("importlib.import_module") as mock_import:
                mock_mod = mock.MagicMock()
                mock_mod.fake_factory = fake_factory
                mock_import.return_value = mock_mod
                init_structured_logger(rank=0, source="test", output_dir=str(tmp_path))

        assert len(called) == 1
        assert called[0] == (0, "test")
        # No TraceJsonlHandler since custom factory replaced default
        assert not any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )


# ---------------------------------------------------------------------------
# No-op flag
# ---------------------------------------------------------------------------


class TestNoOpFlag:
    def test_structured_logger_disabled_flag(self, tmp_path, structured_logger_fixture):
        """When the module-level ``_disabled`` flag is set, spans and instants produce no events."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            set_step(1)
            with log_trace_span("step"):
                pass
            log_trace_instant("structured_logger_started")

        trace_dir = tmp_path / "structured_logs"
        lines = []
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                lines.extend(f.read_text().strip().splitlines())
        assert (
            len(lines) == 0
        ), f"Expected no events when tracing disabled, got {len(lines)}"

    def test_init_with_enable_false_disables_logging(
        self, tmp_path, structured_logger_fixture
    ):
        """``init_structured_logger(enable=False)`` makes all subsequent trace calls no-ops."""
        import torchtitan.observability.structured_logger.structured_logging as sl_mod

        # Ensure clean state before this test
        sl_mod._disabled = False
        try:
            init_structured_logger(
                rank=0, source="trainer", output_dir=str(tmp_path), enable=False
            )
            set_step(1)
            with log_trace_span("step"):
                pass
            log_trace_instant("structured_logger_started")
            log_trace_scalar({"x": 1.0})
        finally:
            sl_mod._disabled = False

        # No structured_logs directory should be created (no handlers attached)
        trace_dir = tmp_path / "structured_logs"
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                assert (
                    f.read_text().strip() == ""
                ), "Expected no events when init'd with enable=False"


# ---------------------------------------------------------------------------
# log_trace_scalar
# ---------------------------------------------------------------------------


class TestLogTraceScalar:
    def test_writes_metric_events(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)
        log_trace_scalar({"train.loss": 2.5, "train.tflops": 45.6})

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["log_type_name"] == "metric_value"
        assert lines[0]["event_name"] == "train.loss"
        assert lines[0]["value"] == 2.5
        assert lines[0]["step"] == 5
        assert lines[1]["event_name"] == "train.tflops"
        assert lines[1]["value"] == 45.6

    def test_noop_when_disabled(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            log_trace_scalar({"should.not.appear": 1.0})

        trace_dir = tmp_path / "structured_logs"
        lines = []
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                lines.extend(f.read_text().strip().splitlines())
        assert len(lines) == 0

    def test_empty_dict_is_noop(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        log_trace_scalar({})

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# log_trace_instant
# ---------------------------------------------------------------------------


class TestLogTraceInstant:
    def test_writes_instant_marker(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        log_trace_instant("structured_logger_started")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1
        assert lines[0]["log_type_name"] == "structured_logger_started"
        assert lines[0]["log_type"] == "instant"
        assert lines[0]["event_name"] is None

    def test_noop_when_disabled(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            log_trace_instant("training_start")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# log_trace_span
# ---------------------------------------------------------------------------


class TestLogTraceSpan:
    def test_logs_start_and_end(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)

        with log_trace_span("fwd_bwd"):
            time.sleep(0.01)

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["log_type_name"] == str("fwd_bwd") + "_start"
        assert lines[1]["log_type_name"] == str("fwd_bwd") + "_end"
        assert lines[1]["value"] > 0
        assert lines[0]["step"] == 5
        assert lines[1]["step"] == 5
        # Span events must NOT have event_name (only in message field)
        assert lines[0].get("event_name") is None
        assert lines[1].get("event_name") is None

    def test_event_type_required(self):
        """event_type is required — cannot be omitted."""
        with pytest.raises(TypeError):
            log_trace_span()  # pyrefly: ignore[missing-argument]

    def test_event_type_accepts_plain_string(self, tmp_path, structured_logger_fixture):
        """Plain strings are accepted as log_type_name."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with log_trace_span("rl_rollout"):
            pass

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert lines[0]["log_type_name"] == "rl_rollout_start"
        assert lines[1]["log_type_name"] == "rl_rollout_end"

    def test_two_spans_emit_start_end_pairs(self, tmp_path, structured_logger_fixture):
        """Sanity check: each `with log_trace_span(...)` writes exactly one
        start and one end record, identified by their ``log_type_name``."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with log_trace_span("step"):
            pass
        with log_trace_span("optim"):
            pass

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        type_names = [line.get("log_type_name") for line in lines]
        assert type_names == ["step_start", "step_end", "optim_start", "optim_end"]

    def test_works_as_decorator(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        @log_trace_span("optim")
        def optimizer_step():
            pass

        optimizer_step()

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2

    def test_async_decorator_brackets_coroutine_body(
        self, tmp_path, structured_logger_fixture
    ):
        # ContextDecorator's default __call__ produces a sync wrapper that
        # exits before `await func(...)` runs, so for an async function the
        # _end record fires before the body executes. Override __call__ for
        # coroutine functions so __exit__ runs after the awaitable completes.
        # Verified by event order in the JSONL stream — no wall-clock timing
        # needed, so the test is deterministic.
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("rl_rollout")
        async def rollout():
            log_trace_instant("body_marker")

        asyncio.run(rollout())

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            type_names = [
                json.loads(line)["log_type_name"] for line in f if line.strip()
            ]

        assert type_names == ["rl_rollout_start", "body_marker", "rl_rollout_end"]

    def test_exception_recording(self, tmp_path, structured_logger_fixture):
        """When an exception occurs, an _error event is emitted."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with pytest.raises(ValueError):
            with log_trace_span("step"):
                raise ValueError("test error")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # 3 events: start, error, end
        assert len(lines) == 3
        type_names = [line["log_type_name"] for line in lines]
        assert type_names[0].endswith("_start")
        assert type_names[1].endswith("_error")
        assert type_names[2].endswith("_end")

    def test_sync_decorator_caller_points_at_user_site(
        self, tmp_path, structured_logger_fixture
    ):
        """``@log_trace_span`` on a sync function must attribute caller to the
        user's call site (test method below), not to the internal sync_wrapper
        in structured_logging.py. Regression guard for the ``stacklevel + 1``
        adjustment in ``log_trace_span.__call__``.
        """
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("decorated_sync")
        def my_fn():
            pass

        my_fn()

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            start = next(
                json.loads(line)
                for line in f
                if line.strip()
                and json.loads(line)["log_type_name"] == "decorated_sync_start"
            )

        path, _, funcname = start["caller"].rsplit(":", 2)
        assert os.path.basename(path) == "test_structured_logging.py", start["caller"]
        # Without stacklevel+1, funcname would be "sync_wrapper".
        assert funcname == "test_sync_decorator_caller_points_at_user_site", start[
            "caller"
        ]

    def test_nested_async_decorators_emit_in_order(
        self, tmp_path, structured_logger_fixture
    ):
        """Mirrors the RL actor pattern: an outer ``@log_trace_span`` async
        method awaits an inner ``@log_trace_span`` async method. All four
        records must emit in nested order (outer_start, inner_start,
        inner_end, outer_end) — proves ``__exit__`` runs after ``await``
        completes for both layers, so durations are real and Gantt pairing
        works.
        """
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("inner_async")
        async def inner():
            pass

        @log_trace_span("outer_async")
        async def outer():
            await inner()

        asyncio.run(outer())

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            type_names = [
                json.loads(line)["log_type_name"] for line in f if line.strip()
            ]

        assert type_names == [
            "outer_async_start",
            "inner_async_start",
            "inner_async_end",
            "outer_async_end",
        ]


# ---------------------------------------------------------------------------
# register_jsonl_handler
# ---------------------------------------------------------------------------


class TestRegisterJsonlHandler:
    def test_creates_handler_with_correct_path(
        self, tmp_path, structured_logger_fixture
    ):
        register_jsonl_handler(
            structured_logger=structured_logger_fixture,
            rank=2,
            source="test_src",
            output_dir=str(tmp_path),
        )

        # Should have added a TraceJsonlHandler
        handlers = [
            h
            for h in structured_logger_fixture.handlers
            if isinstance(h, TraceJsonlHandler)
        ]
        assert len(handlers) == 1

        # File path should be in structured_logs/
        filepath = handlers[0].baseFilename
        assert "structured_logs" in filepath
        assert "test_src.global_rank_2" in filepath
        assert filepath.endswith(".jsonl")
