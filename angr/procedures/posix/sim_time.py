from __future__ import annotations
import time


import angr

# pylint: disable=arguments-differ,unused-argument


class gettimeofday(angr.SimProcedure):
    def run(self, tv, tz):
        if self.state.solver.is_true(tv == 0):
            return -1

        if angr.options.USE_SYSTEM_TIMES in self.state.options:
            flt = time.time()
            result = {"tv_sec": int(flt), "tv_usec": int(flt * 1000000)}
        else:
            result = {
                "tv_sec": self.state.solver.BVS("tv_sec", self.arch.bits, key=("api", "gettimeofday", "tv_sec")),
                "tv_usec": self.state.solver.BVS("tv_usec", self.arch.bits, key=("api", "gettimeofday", "tv_usec")),
            }

        self.state.mem[tv].struct.timeval = result
        return 0


class clock_gettime(angr.SimProcedure):
    def run(self, which_clock, timespec_ptr):
        if not self.state.solver.is_true(which_clock == 0):
            raise angr.errors.SimProcedureError(
                "clock_gettime doesn't know how to deal with a clock other than CLOCK_REALTIME"
            )

        if self.state.solver.is_true(timespec_ptr == 0):
            return -1

        if angr.options.USE_SYSTEM_TIMES in self.state.options:
            flt = time.time()
            result = {"tv_sec": int(flt), "tv_nsec": int(flt * 1000000000)}
        else:
            result = {
                "tv_sec": self.state.solver.BVS("tv_sec", self.arch.bits, key=("api", "clock_gettime", "tv_sec")),
                "tv_nsec": self.state.solver.BVS("tv_nsec", self.arch.bits, key=("api", "clock_gettime", "tv_nsec")),
            }

        self.state.mem[timespec_ptr].struct.timespec = result
        return 0


class concrete_clock_gettime(angr.SimProcedure):
    """Stub ``clock_gettime`` for concrete execution.

    Unlike the symbolic-friendly :class:`clock_gettime` above, this variant
    always returns the host wall-clock time and does **not** validate
    *which_clock*.  This is useful when running under a concrete engine
    (e.g. icicle) where the vDSO is not mapped.
    """

    def run(self, which_clock, timespec_ptr):  # noqa: ARG002
        if self.state.solver.is_true(timespec_ptr == 0):
            return -1
        flt = time.time()
        self.state.mem[timespec_ptr].struct.timespec = {
            "tv_sec": int(flt),
            "tv_nsec": int(flt * 1000000000) % 1000000000,
        }
        return 0
