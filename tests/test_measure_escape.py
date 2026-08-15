"""bench/measure_escape.py is Frozen (AGENTS.md): reruns must reproduce the
ADR 0001 tables exactly.

Nothing here touches a measurement path. What is pinned is the reference-cache
seam the verdict builder uses to release memory operator by operator, which
exists so kernelverify/battery/core.py stops reaching into this module's
private global to do it.
"""

import numpy as np

import measure_escape


def test_drop_references_frees_one_operator_and_leaves_the_others():
    cache = measure_escape._REFERENCE_CACHE
    saved = dict(cache)
    cache.update({
        ("softmax_triton", "case-a"): np.zeros(1),
        ("softmax_triton", "case-b"): np.zeros(1),
        ("matmul_triton", "case-a"): np.zeros(1),
    })
    try:
        measure_escape.drop_references("softmax_triton")
        assert ("softmax_triton", "case-a") not in cache
        assert ("softmax_triton", "case-b") not in cache
        # Every key a later catalogue entry can still hit survives untouched.
        assert ("matmul_triton", "case-a") in cache
    finally:
        cache.clear()
        cache.update(saved)


def test_dropping_an_operator_with_nothing_cached_is_a_no_op():
    before = set(measure_escape._REFERENCE_CACHE)
    measure_escape.drop_references("an_operator_that_was_never_scored")
    assert set(measure_escape._REFERENCE_CACHE) == before
