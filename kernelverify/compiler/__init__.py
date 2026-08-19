"""The generate-gate-price-keep loop.

The verifier is the part of this repository that already exists and is
measured. This package is the part that feeds it: something proposes a
kernel, the gates judge it, the pricing times the survivors, and what
survives both is kept with a certificate.

Nothing here decides whether a kernel is correct or fast. Those answers
live in `kernelverify.pack` and `bench/`, and this package's only job is to
route candidates through them and to remember what happened to every one,
including the ones that died.
"""
