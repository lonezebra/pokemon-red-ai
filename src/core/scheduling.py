"""
Steer worker processes away from the machine's fastest core tier when the
user is keeping some cores for themselves.

The request this answers: "if I set workers to 13 on my 6-super/
12-performance M5, use the performance cores and leave me the supers."
macOS offers no hard core pinning -- there is no affinity API -- but its
scheduler places work by QoS class, and QOS_CLASS_UTILITY tells it this
work is throughput-oriented and should keep off the highest-performance
cores when there is any contention. Marking each worker UTILITY therefore
biases the pool onto the lower fast tier, leaving the top tier available
for the user's foreground apps. It is a strong bias, not a guarantee, and
that distinction is documented rather than papered over: with fewer
workers than cores, expect the super cores to sit mostly idle in Activity
Monitor's per-core view, not provably always-idle.

Policy, via POKEMON_RED_YIELD_SUPER:
  unset  -> automatic: yield the top tier exactly when the worker count is
            below the machine's physical core count (a deliberately
            partial run means the user wants the leftover capacity to be
            the *good* cores, or they would have run full);
  "1"    -> always yield;  "0" -> never yield.

The driver decides once and passes the decision to workers through the
environment (spawned children inherit it); each worker then applies the
QoS to itself. Self-application is the reliable direction: macOS QoS is
fundamentally per-thread, set from within, and the worker's main thread
is where PyBoy runs.

The actual call is pthread_set_qos_class_self_np via ctypes, guarded to
darwin and wrapped so any failure degrades to normal scheduling rather
than an error -- a scheduling preference is never worth crashing a
training run over. On Linux (the development container) the whole thing
is a documented no-op.
"""

import ctypes
import ctypes.util
import os
import sys

ENV_FLAG = "POKEMON_RED_YIELD_SUPER"

# From <sys/qos.h>. UTILITY is the designed class for long-running,
# user-doesn't-wait-on-it computation: below default priority, above
# BACKGROUND's hard efficiency clamp (which on tiered Apple silicon can
# confine work to the slowest cores entirely -- too blunt here, where the
# point is to *use* the performance tier and spare only the supers).
QOS_CLASS_UTILITY = 0x11


def decide_yield(num_workers, total_cores=None, env=None):
    """
    The driver-side policy decision, separated from any syscall so it is
    testable on any platform. Returns True when workers should keep off
    the top core tier.
    """
    env = os.environ if env is None else env
    flag = env.get(ENV_FLAG, "").strip()
    if flag == "1":
        return True
    if flag == "0":
        return False

    if total_cores is None:
        total_cores = os.cpu_count() or 0
    return 0 < num_workers < total_cores


def mark_decision_for_workers(should_yield):
    """
    Record the driver's decision where spawned children will see it.
    Spawn re-imports everything from scratch, so the environment is the
    one channel that needs no extra plumbing through pickled arguments.
    """
    os.environ[ENV_FLAG] = "1" if should_yield else "0"


def apply_worker_qos():
    """
    Called by each worker at startup: if the driver decided to yield the
    top tier, demote this process's main thread to UTILITY. Returns True
    if the QoS call was actually made and succeeded -- callers only use
    this for logging, never for control flow.
    """
    if os.environ.get(ENV_FLAG) != "1":
        return False
    if sys.platform != "darwin":
        return False

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib",
                           use_errno=True)
        result = libc.pthread_set_qos_class_self_np(
            ctypes.c_uint(QOS_CLASS_UTILITY), ctypes.c_int(0)
        )
        return result == 0
    except Exception:
        # A scheduling preference is never worth crashing a run over.
        return False
