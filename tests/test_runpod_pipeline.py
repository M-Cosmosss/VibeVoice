from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


def load_pipeline_module():
    runpod_app_mod = types.ModuleType("runpod_app")
    httpx_mod = types.ModuleType("httpx")
    timing_mod = types.ModuleType("runpod_app.timing")

    @contextmanager
    def timed(*_args, **_kwargs):
        yield {}

    timing_mod.emit = lambda *_args, **_kwargs: None
    timing_mod.gpu_name = lambda: "test-gpu"
    timing_mod.timed = timed
    httpx_mod.AsyncClient = object
    httpx_mod.HTTPStatusError = RuntimeError
    httpx_mod.Limits = lambda *_args, **_kwargs: object()

    with mock.patch.dict(
        os.environ,
        {
            "MAX_MODEL_LEN": "32768",
            "ASR_PROMPT_TOKEN_RESERVE": "512",
        },
    ), mock.patch.dict(
        sys.modules,
        {
            "httpx": httpx_mod,
            "runpod_app": runpod_app_mod,
            "runpod_app.timing": timing_mod,
        },
    ):
        module_path = Path(__file__).resolve().parents[1] / "runpod" / "pipeline.py"
        spec = importlib.util.spec_from_file_location("runpod_pipeline_test_target", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            return module
        finally:
            sys.modules.pop(spec.name, None)


class RunPodPipelineTest(unittest.TestCase):
    def test_payload_uses_remaining_context_for_output_tokens(self) -> None:
        module = load_pipeline_module()

        payload = module._build_payload("ZmFrZQ==", "audio/mpeg", 1800.0, None)

        self.assertEqual(payload["max_tokens"], 18753)

    def test_payload_clips_output_tokens_to_model_context(self) -> None:
        module = load_pipeline_module()

        payload = module._build_payload("ZmFrZQ==", "audio/mpeg", 3600.0, None)

        self.assertEqual(payload["max_tokens"], 5253)

    def test_extract_chat_content_reads_openai_response(self) -> None:
        module = load_pipeline_module()

        content = module._extract_chat_content({
            "choices": [{"message": {"content": "[{\"Content\":\"ok\"}]"}}],
        })

        self.assertEqual(content, "[{\"Content\":\"ok\"}]")

    def test_extract_chat_content_reports_non_chat_response_body(self) -> None:
        module = load_pipeline_module()

        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            module._extract_chat_content({"error": {"message": "engine failed"}})


if __name__ == "__main__":
    unittest.main()
