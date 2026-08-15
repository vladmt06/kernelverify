"""The canonical external toolchain: where llama.cpp and the models live.

The external dependencies sit deliberately outside the repo (AGENTS.md): one
llama.cpp checkout at a pinned commit, and the GGUF models beside it. These
paths are canonical across lanes as of 2026-08-14; ~/src/llama.cpp is
abandoned. Every bench script that shells out to that checkout takes the
paths, the build flags and the invoker from here, because two spellings of a
path or a flag list are how a lane ends up measuring a different binary than
it recorded.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LLAMA_CPP = Path("/Users/vlad/llama.cpp")
LLAMA_BENCH = LLAMA_CPP / "build" / "bin" / "llama-bench"
LLAMA_QUANTIZE = LLAMA_CPP / "build" / "bin" / "llama-quantize"
TEST_BACKEND_OPS = LLAMA_CPP / "build" / "bin" / "test-backend-ops"
GGUF_DIR = Path("/Users/vlad/models/gguf")

# How the pinned checkout was configured. Recorded with the results because a
# baseline that cannot be rebuilt is not a baseline.
CMAKE_FLAGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_METAL=ON",
    "-DGGML_METAL_EMBED_LIBRARY=ON",
]


def run_llama_bench(model: Path, args: list[str]) -> list[dict]:
    """One llama-bench invocation over `model`, parsed JSON rows out.

    `args` carries everything workload-specific (-p/-n/-d/-r/...); the model
    and the JSON output format are fixed here so every caller parses the same
    shape. Raises with the command and the tail of stderr on a non-zero exit.
    """
    cmd = [str(LLAMA_BENCH), "-m", str(model), "-o", "json"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)
