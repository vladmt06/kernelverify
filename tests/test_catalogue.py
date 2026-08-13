"""The catalogue must stay in lockstep with the corpus ports.

The product claim rests on the synthetic fault population provably containing
the published faults; that holds only while every from_corpus mutation pins
exactly the keywords the port pins.
"""

import functools
import inspect

from cpu_ports import BUGGY_TO_CONTROL, PORTS
from kernelverify.mutation.catalogue import CATALOGUE, KERNEL_TO_CORPUS_OP
from kernelverify.reference.kernels import KERNELS


def test_catalogue_size_and_unique_names():
    names = [m.name for m in CATALOGUE]
    assert len(names) == 44
    assert len(set(names)) == 44


def test_every_published_fault_is_in_the_catalogue():
    from_corpus = {m.corpus_name: m for m in CATALOGUE if m.from_corpus}
    buggy_ports = {k for k in PORTS if k.endswith("_buggy")}
    assert set(from_corpus) == buggy_ports
    assert len(from_corpus) == 10


def test_corpus_mutations_pin_exactly_the_port_keywords():
    for m in CATALOGUE:
        if not m.from_corpus:
            continue
        port = PORTS[m.corpus_name]
        assert isinstance(port, functools.partial)
        assert port.func is KERNELS[m.kernel]
        assert port.keywords == m.params


def test_every_mutation_builds_against_a_real_seam():
    for m in CATALOGUE:
        assert m.kernel in KERNELS
        assert m.kernel in KERNEL_TO_CORPUS_OP
        seams = {
            name
            for name, p in inspect.signature(KERNELS[m.kernel]).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert set(m.params) <= seams, f"{m.name} pins a keyword that does not exist"
        assert callable(m.build())


def test_every_buggy_port_has_a_control():
    assert set(BUGGY_TO_CONTROL) == {k for k in PORTS if k.endswith("_buggy")}
    for control in BUGGY_TO_CONTROL.values():
        assert control in PORTS and not control.endswith("_buggy")
