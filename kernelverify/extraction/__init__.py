"""Extract the MSL that MLX actually runs, and validate the extraction.

MLX surfaces built on `mx.fast.metal_kernel` generate their final Metal source
at call time: the template is expanded, the signature synthesised, shape and
stride buffers appended when the body refers to them. The verifier needs that
final source, because a certificate about "the kernel MLX runs" is worthless
if it describes the template instead of the translation unit the compiler saw.

The capture path is shaped by two measured facts (mlx 0.32.0):

- The verbose dump is written by the C++ layer straight to file descriptor 1.
  A Python-level `sys.stdout` redirect captures nothing at all, so the capture
  worker rebinds fd 1 itself before touching MLX.
- One capture process serves exactly one specialization. The eng review's
  stated reason was that the in-process kernel cache suppresses reprints; on
  0.32.0 that suppression does not reproduce (a repeat verbose call reprints,
  even after the cache is warm). The rule survives on attribution grounds: a
  process whose fd 1 carries exactly one dump cannot mis-attribute source to
  the wrong specialization, whatever a future MLX does about reprints.

The live arm runs the same surface natively in a disposable subprocess. It
has no verdict authority: it returns arrays, and the match between the
extracted kernel (run by our Metal runner) and the live arm is judged by the
shipped oracle's verdict function across the battery, structured modes
included, never bitwise. Both arms compile under MLX's documented default
math mode (safe) - Metal's own default is fast - and the options used are
recorded as certificate inputs. A validation failure pattern that
concentrates on reassociation-sensitive cases is raised as the compile-option
alarm, because that is what a math-mode divergence between the arms looks
like.
"""

from kernelverify.extraction.bridge import (
    extracted_spec,
    metadata_inputs,
    signature_bindings,
)
from kernelverify.extraction.mlx_arm import (
    CaptureRecord,
    ExtractionError,
    LiveResult,
    capture_specialization,
    run_live,
)
from kernelverify.extraction.surface import LiveCall, MLXKernelSurface
from kernelverify.extraction.validate import ExtractionReport, validate_extraction

__all__ = [
    "CaptureRecord",
    "ExtractionError",
    "ExtractionReport",
    "LiveCall",
    "LiveResult",
    "MLXKernelSurface",
    "capture_specialization",
    "extracted_spec",
    "metadata_inputs",
    "run_live",
    "signature_bindings",
    "validate_extraction",
]
