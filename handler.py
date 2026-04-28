"""RunPod GitHub-integration scan target.

RunPod's pre-deploy validator checks the default branch for the magic call
``runpod.serverless.start(...)`` before allowing the Endpoint to build. The
real handler implementation lives in ``runpod/handler.py`` and is launched by
the Dockerfile ENTRYPOINT (``runpod/start.sh``) at runtime — this file is
never the actual entrypoint inside the container.

It is also written so that running ``python handler.py`` from the repo root
inside the built image still works (forwards to the real handler), but the
canonical path is the start.sh flow.
"""
from __future__ import annotations

import sys

# Ensure the in-image package path is importable when this file is the entry.
sys.path.insert(0, "/app")

import runpod  # noqa: E402  (pip package, not the local ``runpod/`` dir)

try:
    from runpod_app.handler import handler  # type: ignore[import-not-found]
except ImportError:
    # Fallback for direct local execution from the repo root.
    sys.path.insert(0, "runpod")
    from handler import handler  # type: ignore[no-redef]


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
