from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_handler_module():
    runpod_mod = types.ModuleType("runpod")
    runpod_mod.serverless = SimpleNamespace(start=lambda *_args, **_kwargs: None)

    runpod_app_mod = types.ModuleType("runpod_app")
    pipeline_mod = types.ModuleType("runpod_app.pipeline")
    timing_mod = types.ModuleType("runpod_app.timing")

    async def fake_run_job(*_args, **_kwargs):
        return SimpleNamespace(
            job_id="job-test",
            text="ok",
            segments=[],
            files={},
            timing={},
        )

    pipeline_mod.run_job = fake_run_job
    timing_mod.emit = lambda *_args, **_kwargs: None

    with mock.patch.dict(
        sys.modules,
        {
            "runpod": runpod_mod,
            "runpod_app": runpod_app_mod,
            "runpod_app.pipeline": pipeline_mod,
            "runpod_app.timing": timing_mod,
        },
    ):
        module_path = Path(__file__).resolve().parents[1] / "runpod" / "handler.py"
        spec = importlib.util.spec_from_file_location("runpod_handler_test_target", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class RunPodHandlerTest(unittest.TestCase):
    def test_handler_runs_without_existing_event_loop(self) -> None:
        module = load_handler_module()

        result = module.handler({"input": {"audio_url": "https://example.test/audio.mp3"}})

        self.assertEqual(result["job_id"], "job-test")
        self.assertEqual(result["text"], "ok")

    def test_handler_runs_inside_existing_event_loop(self) -> None:
        module = load_handler_module()

        async def call_handler():
            return module.handler({"input": {"audio_url": "https://example.test/audio.mp3"}})

        result = asyncio.run(call_handler())

        self.assertEqual(result["job_id"], "job-test")
        self.assertEqual(result["text"], "ok")


if __name__ == "__main__":
    unittest.main()
