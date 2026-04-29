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
        prompt = payload["messages"][1]["content"][1]["text"]
        self.assertIn("Return only a valid JSON array", prompt)
        self.assertIn("Use numeric seconds", prompt)

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

    def test_parse_segments_accepts_common_vibevoice_keys(self) -> None:
        module = load_pipeline_module()

        segments, parse_ok = module._parse_segments(
            '[{"Start": 1.25, "End": 3.5, "Speaker": 0, "Content": "hello"}]',
            offset_s=10.0,
        )

        self.assertTrue(parse_ok)
        self.assertEqual(
            segments,
            [{"start": 11.25, "end": 13.5, "speaker": "0", "text": "hello"}],
        )

    def test_parse_segments_accepts_document_wrapped_list(self) -> None:
        module = load_pipeline_module()

        segments, parse_ok = module._parse_segments(
            '{"segments":[{"Start time":"00:01.500","End time":"00:02.000",'
            '"Speaker ID":"1","Content":"ok"}]}',
            offset_s=20.0,
        )

        self.assertTrue(parse_ok)
        self.assertEqual(
            segments,
            [{"start": 21.5, "end": 22.0, "speaker": "1", "text": "ok"}],
        )

    def test_format_duration_is_human_readable(self) -> None:
        module = load_pipeline_module()

        self.assertEqual(module._format_duration(0.1234), "0.123s")
        self.assertEqual(module._format_duration(62.003), "1m2.003s")
        self.assertEqual(module._format_duration(3723.456), "1h2m3.456s")

    def test_timing_report_uses_readable_stage_and_chunk_names(self) -> None:
        module = load_pipeline_module()

        results = [
            module.ChunkResult(
                index=0,
                start_s=0.0,
                end_s=60.0,
                duration_s=60.0,
                raw_content="[]",
                segments=[{"text": "one"}],
                parse_ok=True,
                prepare_duration_s=0.125,
                asr_duration_s=3.5,
            ),
            module.ChunkResult(
                index=1,
                start_s=60.0,
                end_s=90.0,
                duration_s=30.0,
                raw_content="[]",
                segments=[],
                parse_ok=False,
                prepare_duration_s=0.25,
                asr_duration_s=2.0,
            ),
        ]

        timing = module._build_timing_report(
            total_s=10.25,
            audio_duration_s=90.0,
            chunk_seconds=60,
            concurrency=2,
            download_s=1.0,
            probe_s=0.2,
            split_s=0.3,
            asr_wall_s=3.7,
            merge_s=0.05,
            results=results,
        )

        self.assertEqual(timing["summary"]["total_elapsed_time"]["readable"], "10.250s")
        self.assertEqual(timing["configuration"]["chunk_duration"]["readable"], "1m0.000s")
        self.assertIn("download_audio", timing["stages"])
        self.assertIn("transcribe_audio_wall_time", timing["stages"])
        self.assertEqual(timing["chunk_summary"]["transcription_slowest_chunk"]["readable"], "3.500s")
        self.assertEqual(timing["chunks"][0]["transcription_time"]["readable"], "3.500s")
        self.assertEqual(timing["chunks"][1]["audio_start"]["readable"], "1m0.000s")
        self.assertFalse(timing["chunks"][1]["parsed_as_json"])


if __name__ == "__main__":
    unittest.main()
