"""On-host capture of GPU artifacts and performance (P4 / artifact track).

These helpers run on a GPU host (vast.ai image with CUDA + torch/triton). They
are written defensively so importing this module never fails off-GPU; each
function degrades to a clear "unavailable" result when its toolchain is missing.

Outputs feed two places:
  * PTX/SASS text -> gpuemu daemon `LintKernel`/`StoreArtifact` (static metrics).
  * timing/occupancy -> results.jsonl alongside the correctness verdict (P4
    ground truth to validate static metrics against measured regressions).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


# --------------------------------------------------------------------------
# PTX / SASS extraction
# --------------------------------------------------------------------------
def ptx_from_cu(cu_path: str, arch: str = "sm_80") -> Optional[str]:
    """Compile a .cu file to PTX with nvcc (no GPU needed, just the toolkit)."""
    if not have("nvcc"):
        return None
    out = subprocess.run(
        ["nvcc", "-ptx", f"-arch={arch}", cu_path, "-o", "/dev/stdout"],
        capture_output=True, text=True,
    )
    return out.stdout if out.returncode == 0 else None


def ptx_from_triton_cache(cache_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """Collect PTX emitted into the Triton cache after a kernel launch.

    Triton writes <hash>/<name>.ptx into ~/.triton/cache (or TRITON_CACHE_DIR).
    Call this *after* running the Triton kernel at least once.
    """
    import os

    root = Path(cache_dir or os.environ.get("TRITON_CACHE_DIR")
                or (Path.home() / ".triton" / "cache"))
    found = []
    if root.exists():
        for ptx in root.rglob("*.ptx"):
            found.append({"name": ptx.stem, "path": str(ptx), "ptx": ptx.read_text()})
    return found


def sass_from_cubin(cubin_path: str, function: Optional[str] = None) -> Optional[str]:
    """Disassemble a cubin to SASS via cuobjdump (requires NVIDIA tools)."""
    if not have("cuobjdump"):
        return None
    args = ["cuobjdump", "--dump-sass"]
    if function:
        args += ["--function", function]
    args.append(cubin_path)
    out = subprocess.run(args, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


# --------------------------------------------------------------------------
# Performance + occupancy measurement (CUDA events via torch)
# --------------------------------------------------------------------------
def time_kernel(run: Callable[[], Any], warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    """Warmup + CUDA-event timing of a no-arg callable that launches the kernel.

    Returns median/mean/min milliseconds, or {"available": False} off-GPU.
    """
    try:
        import torch
    except Exception:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "available": True,
        "iters": iters,
        "ms_min": ms[0],
        "ms_median": ms[len(ms) // 2],
        "ms_mean": sum(ms) / len(ms),
        "device": torch.cuda.get_device_name(0),
    }


def device_info() -> Dict[str, Any]:
    """GPU identity + key occupancy limits (for normalizing P4 measurements)."""
    try:
        import torch
    except Exception:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "total_mem_mb": props.total_memory // (1024 * 1024),
        "capability": f"{props.major}.{props.minor}",
    }
