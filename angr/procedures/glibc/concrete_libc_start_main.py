import claripy

import angr


def _get_libc_start_main_cls():
    """Deferred import to avoid circular references and name-mangling issues
    (the module name starts with double underscores)."""
    from angr.procedures.glibc import __libc_start_main as mod  # noqa: N812

    return mod.__libc_start_main


class ConcreteLibcStartMain(angr.SimProcedure):
    """Lightweight ``__libc_start_main`` for concrete execution.

    Skips glibc initializers and jumps straight to ``main`` with the real
    *argc* / *argv*.  Pushes the address of ``exit`` onto the stack so that
    a ``ret`` from ``main`` lands on a known symbol the concrete engine can
    intercept.
    """

    NO_RET = True

    def run(self, main, argc, argv, init, fini):
        lsm = _get_libc_start_main_cls()

        main, argc, argv, _, _ = lsm._extract_args(
            self.state, main, argc, argv, init, fini
        )

        # Initialize ctype tables and errno just like the full procedure does.
        lsm._initialize_b_loc_table(self)
        lsm._initialize_tolower_loc_table(self)
        lsm._initialize_toupper_loc_table(self)
        lsm._initialize_errno(self)

        # Pass real argc/argv/envp to main via registers.
        self.state.regs.rdi = argc
        self.state.regs.rsi = argv
        envp = argv + (argc + 1) * self.state.arch.bytes
        self.state.regs.rdx = envp

        # Use ``exit`` as the return sentinel so the concrete engine sees a
        # breakpoint when main returns.
        sentinel = self._resolve_exit_addr()
        self.state.memory.store(
            self.state.regs.sp,
            claripy.BVV(sentinel, self.state.arch.bits),
            endness=self.state.arch.memory_endness,
        )
        self.jump(main)

    def _resolve_exit_addr(self):
        for name in ("exit", "_exit", "_Exit"):
            sym = self.project.loader.find_symbol(name)
            if sym is not None:
                return sym.rebased_addr
        raise ValueError("Cannot find exit symbol for return sentinel")
