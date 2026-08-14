"""Turn a kernel template into a concrete `KernelSpec`, outside the runner.

A generated kernel often arrives as a family: one source with a hole where the
element type or a tile size goes, instantiated per dtype or per shape class.
Metal has no generics over raw source, so the family is expressed textually:
the template is an ordinary `KernelSpec` whose source, entry point, or name
contain `$NAME` placeholders, and `specialize` fills them and returns a new,
fully validated `KernelSpec`.

This lives outside the runner on purpose (ruling 1A): the runner only ever
sees finished specs, `KernelSpec` itself carries no template state, and the
specialization a candidate was built from stays the caller's record, where the
certificate can cite it.

`$` never occurs in Metal shading language, so a placeholder can never collide
with kernel code. Both mistakes a caller can make are errors, not silence: a
placeholder the substitutions do not fill raises, and a substitution the
template never mentions raises, because it is almost always a misspelled
placeholder name.
"""

from __future__ import annotations

from string import Template

from kernelverify.runners.spec import KernelSpec, SpecError


def specialize(template: KernelSpec, substitutions: dict) -> KernelSpec:
    """A new `KernelSpec` with every `$NAME` placeholder filled.

    Placeholders are replaced in the source, the entry point, and the name;
    bindings and launch pass through unchanged, since per-case sizes already
    arrive as run parameters. Values are converted with `str`, so dtypes and
    integer tile sizes both work.
    """
    fields = {
        "source": template.source,
        "entry_point": template.entry_point,
        "name": template.name,
    }
    used: set[str] = set()
    filled: dict[str, str] = {}
    for where, text in fields.items():
        parsed = Template(text)
        used.update(parsed.get_identifiers())
        try:
            filled[where] = parsed.substitute(substitutions)
        except KeyError as error:
            raise SpecError(
                f"the template's {where} names ${error.args[0]}, "
                "which the substitutions do not supply"
            ) from error
        except ValueError as error:
            raise SpecError(f"the template's {where} is malformed: {error}") from error

    unused = sorted(set(substitutions) - used)
    if unused:
        raise SpecError(
            f"substitutions {unused} appear nowhere in the template; "
            "a misspelled placeholder name looks exactly like this"
        )

    return KernelSpec(
        source=filled["source"],
        entry_point=filled["entry_point"],
        bindings=template.bindings,
        launch=template.launch,
        name=filled["name"],
    )
