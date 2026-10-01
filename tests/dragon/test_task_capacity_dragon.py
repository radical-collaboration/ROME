"""How many never-returning service tasks can run at once on this backend?

    dragon -s tests/dragon/test_task_capacity_dragon.py

ROME runs one persistent service task per stream replica, and a service task
holds its slot for the whole run. So this number is the budget: replicas past
it sit in STARTING forever, the requests routed to them are never claimed, and
if the streams take every slot then training never gets one either.

Measured 2 on a 4-CPU single node. Run it on your allocation before choosing
num_streams -- you need at least num_streams + 1.
"""

import asyncio
import sys
import time

import dragon  # noqa: F401

N = 6


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def forever(ser, marker):
    from dragon.data.ddict import DDict

    DDict.attach(ser)[marker] = "up"
    for _ in range(3000):
        await asyncio.sleep(0.1)


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

    for i in range(N):
        flow.function_task(service=True)(forever)(ser, f"s{i}", task_description={})
    say(f"submitted {N} never-returning service tasks")

    best = 0
    for t in range(12):
        await asyncio.sleep(3)
        up = sum(1 for i in range(N) if _has(d, f"s{i}"))
        best = max(best, up)
        say(f"  t+{3*(t+1)}s  running {up}/{N}")
        if up == N:
            break

    say("")
    say(f"ceiling observed: {best}/{N} concurrent service tasks")
    say(f"os.cpu_count(): {__import__('os').cpu_count()}")
    d.destroy()
    return 0


def _has(d, k):
    try:
        d[k]
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
