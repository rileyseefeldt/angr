"""Utilities for concrete execution of dynamically-linked binaries.

These helpers set up the hooks and GOT patches required when a concrete engine
(such as icicle) runs a dynamically-linked binary whose shared libraries are
loaded by CLE but whose glibc initialisation would otherwise not execute.
"""

from __future__ import annotations

import struct

from archinfo import Endness


def resolve_got_entries(project):
    """Eagerly resolve GOT entries (equivalent to ``LD_BIND_NOW=1``).

    CLE resolves symbols but doesn't always patch the GOT, leaving entries
    pointing at the dynamic linker's lazy resolver which may not be mapped in
    the concrete engine.  This function returns a dict of
    ``{addr: packed_bytes}`` that should be written into the initial state's
    memory.
    """
    arch = project.arch
    word_size = arch.bytes
    fmt = (">" if arch.memory_endness == Endness.BE else "<") + ("I" if word_size == 4 else "Q")

    patches: dict[int, bytes] = {}
    for obj in project.loader.all_objects:
        for reloc in obj.relocs:
            if not reloc.resolved or not reloc.resolvedby:
                continue
            got_addr = getattr(reloc, "rebased_addr", None)
            if not got_addr:
                continue
            target = reloc.resolvedby.rebased_addr
            cur = struct.unpack(fmt, project.loader.memory.load(got_addr, word_size))[0]
            if cur != target:
                patches[got_addr] = struct.pack(fmt, target)

    return patches


def setup_concrete_hooks(project):
    """Install SimProcedure hooks needed for concrete execution of dynamically-linked binaries.

    Hooks the following symbols (when present and not already hooked):

    * ``__libc_start_main`` -- replaced by :class:`~angr.procedures.glibc.concrete_libc_start_main.ConcreteLibcStartMain`
    * ``exit`` / ``_exit`` / ``_Exit`` -- standard exit procedure
    * ``clock_gettime`` / ``__clock_gettime`` -- concrete wall-clock stub
    * ``__ctype_b_loc``, ``__ctype_tolower_loc``, ``__ctype_toupper_loc`` -- locale tables
    """
    from angr.procedures.glibc.concrete_libc_start_main import ConcreteLibcStartMain
    from angr.procedures.libc.exit import exit as ExitProcedure
    from angr.procedures.posix.sim_time import concrete_clock_gettime
    from angr.procedures.glibc.__ctype_b_loc import __ctype_b_loc  # noqa: N812
    from angr.procedures.glibc.__ctype_tolower_loc import __ctype_tolower_loc  # noqa: N812
    from angr.procedures.glibc.__ctype_toupper_loc import __ctype_toupper_loc  # noqa: N812

    sym = project.loader.find_symbol("__libc_start_main")
    if sym is not None and sym.rebased_addr not in project._sim_procedures:
        project.hook(sym.rebased_addr, ConcreteLibcStartMain())

    for name in ("exit", "_exit", "_Exit"):
        sym = project.loader.find_symbol(name)
        if sym is not None and sym.rebased_addr not in project._sim_procedures:
            project.hook(sym.rebased_addr, ExitProcedure())

    for name in ("clock_gettime", "__clock_gettime"):
        sym = project.loader.find_symbol(name)
        if sym is not None and sym.rebased_addr not in project._sim_procedures:
            project.hook(sym.rebased_addr, concrete_clock_gettime())

    for name, proc_cls in [
        ("__ctype_b_loc", __ctype_b_loc),
        ("__ctype_tolower_loc", __ctype_tolower_loc),
        ("__ctype_toupper_loc", __ctype_toupper_loc),
    ]:
        sym = project.loader.find_symbol(name)
        if sym is not None and sym.rebased_addr not in project._sim_procedures:
            project.hook(sym.rebased_addr, proc_cls())
