"""A task can run to completion and never resolve its future.

    dragon -s tests/dragon/test_service_blocks_results_dragon.py

This is what stops a ROME training round from completing while inference
streams are up. The round *runs* -- it writes its checkpoint to disk -- but the
driver's future stays PENDING forever, so TrainerStatus never leaves RUNNING and
no checkpoint is ever published.

The distinction this script draws is between two different things:

  ran       the task body executed (observed through a side effect in a DDict)
  resolved  the driver's future for it completed

An earlier version of this script only measured "ran", which is why a first run
on a real allocation looked healthy. A task that runs but never resolves is the
failure.

No ROME here -- plain Dragon, rhapsody and asyncflow. The variable is simply
whether a service task is running: an *idle* one that only sleeps is enough.
Measured on a single node:

    scenario        ran     resolved
    no service      True    True
    idle service    True    False     <- ran, future never resolved
    busy service    False   False

Exit status is non-zero if any task ran without resolving.
"""

import asyncio
import os
import sys
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def idle_service(ser, marker):
    """Service that only sleeps."""
    from dragon.data.ddict import DDict

    DDict.attach(ser)[marker] = "up"
    for _ in range(6000):
        await asyncio.sleep(0.1)


async def busy_service(ser, marker):
    """Service that scans and pops, like a ROME stream replica."""
    from dragon.data.ddict import DDict

    d = DDict.attach(ser)
    d[marker] = "up"
    for _ in range(6000):
        try:
            for k in list(d.keys()):
                if isinstance(k, str) and k.startswith("work|"):
                    try:
                        d.pop(k)
                    except Exception:
                        pass
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def worker(ser, marker, seconds):
    """An ordinary task: does some work, leaves a mark, returns a value."""
    from dragon.data.ddict import DDict

    await asyncio.sleep(seconds)
    DDict.attach(ser)[marker] = "ran"
    return f"{marker}-returned"


async def main():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    try:
        from rhapsody.backends import DragonExecutionBackend
    except ImportError:
        from rhapsody.backends import DragonExecutionBackendV3 as DragonExecutionBackend

    backend = await DragonExecutionBackend({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)
    d = DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2)
    ser = d.serialize()

    seconds = float(os.environ.get("WORK_SECONDS", 0.5))
    wait_for = float(os.environ.get("RESOLVE_TIMEOUT", 45))

    def ran(marker):
        try:
            d[marker]
            return True
        except Exception:
            return False

    async def up(marker, secs=30):
        for _ in range(int(secs / 0.5)):
            if ran(marker):
                return True
            await asyncio.sleep(0.5)
        return False

    async def probe(label, marker):
        """Submit an ordinary task; report whether it ran and whether it resolved."""
        fut = flow.function_task()(worker)(ser, marker, seconds, task_description={})
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=wait_for)
            resolved = True
        except asyncio.TimeoutError:
            resolved = False
        except Exception as ex:
            say(f"   {label}: future raised {type(ex).__name__}: {ex}")
            resolved = True
        say(f"   {label}: ran={ran(marker)} resolved={resolved}")
        return ran(marker), resolved

    results = {}

    say("1. ordinary task, nothing else running")
    results["no service"] = await probe("no service", "w1")

    say("2. start an IDLE service (sleeps only)")
    flow.function_task(service=True)(idle_service)(ser, "svc_idle", task_description={})
    say(f"   idle service up: {await up('svc_idle')}")
    results["idle service"] = await probe("idle service", "w2")

    say("3. start a BUSY service (keys()+pop, like a stream replica)")
    flow.function_task(service=True)(busy_service)(ser, "svc_busy", task_description={})
    say(f"   busy service up: {await up('svc_busy')}")
    results["busy service"] = await probe("busy service", "w3")

    say("")
    say(f"{'scenario':<16}{'ran':<8}{'resolved':<10}")
    for name, (r, s) in results.items():
        say(f"{name:<16}{str(r):<8}{str(s):<10}")
    say("")
    broken = [n for n, (r, s) in results.items() if r and not s]
    if broken:
        say(f"RAN BUT NEVER RESOLVED: {broken}")
    else:
        say("every task that ran also resolved")

    d.destroy()
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
