"""Utilities for concrete execution of dynamically-linked binaries.

These helpers set up the hooks required when a concrete engine (such as
icicle) runs a dynamically-linked binary whose shared libraries are loaded by
CLE but whose glibc initialisation would otherwise not execute.
"""

from __future__ import annotations


def setup_concrete_hooks(project):
    """Install SimProcedure hooks needed for concrete execution of dynamically-linked binaries.

    Hooks the following symbols (when present and not already hooked):

    * ``__libc_start_main`` -- standard angr SimProcedure. Real glibc's version
      runs ld.so handshakes, atexit/fini setup, and module init that depend on
      ld.so internal state the loader does not reproduce.
    * ``__ctype_b_loc``, ``__ctype_tolower_loc``, ``__ctype_toupper_loc`` --
      locale tables. Real glibc reads these from a TLS slot populated by
      ``__ctype_init``, which itself depends on ``_nl_global_locale`` being
      initialized -- not tractable without running glibc init.
    """
    from angr.procedures.glibc.__libc_start_main import __libc_start_main as LibcStartMain  # noqa: N812
    from angr.procedures.glibc.__ctype_b_loc import __ctype_b_loc  # noqa: N812
    from angr.procedures.glibc.__ctype_tolower_loc import __ctype_tolower_loc  # noqa: N812
    from angr.procedures.glibc.__ctype_toupper_loc import __ctype_toupper_loc  # noqa: N812

    sym = project.loader.find_symbol("__libc_start_main")
    if sym is not None and sym.rebased_addr not in project._sim_procedures:
        project.hook(sym.rebased_addr, LibcStartMain())

    for name, proc_cls in [
        ("__ctype_b_loc", __ctype_b_loc),
        ("__ctype_tolower_loc", __ctype_tolower_loc),
        ("__ctype_toupper_loc", __ctype_toupper_loc),
    ]:
        sym = project.loader.find_symbol(name)
        if sym is not None and sym.rebased_addr not in project._sim_procedures:
            project.hook(sym.rebased_addr, proc_cls())
