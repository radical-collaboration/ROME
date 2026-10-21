"""An executable task finishes, its side effect is on disk, its future hangs.

    dragon -s tests/dragon/test_executable_result_hang_dragon.py

A minimal, self-contained reproducer for an upstream report: no ROME, just
``dragon`` + ``rhapsody`` (``DragonExecutionBackend``) + ``radical.asyncflow``.
It shows that while a long-lived **service** task is running, an ordinary
**executable** task submitted afterwards runs to completion — the shell command
executes and writes its output file — yet the ``asyncio.Future`` asyncflow
returned for it never resolves. ``await`` on that future blocks forever.

Both tasks here are plain shell commands, so there is no function-vs-executable
variable: the *service* is ``sh -c 'touch marker; sleep …'`` submitted with
``service=True``, and the *probe* is ``sh -c 'echo ran > file'``. A file on disk
is proof a process ran and exited 0; a future still PENDING long after is proof
the completion was never delivered back to the driver.

The suspected mechanism (see ``test_result_delivery_dragon.py`` for the probe):
rhapsody's monitor sweeps outstanding tasks in order and reads each one's result
key; a still-running service task's key read *blocks* rather than raising, so
every result behind it — including the finished executable task's — is never
delivered. The Dragon backend has no service handling at all (unlike the
RADICAL-Pilot backend, which resolves a service future when it reaches RUNNING),
so ``service=True`` is a no-op there and the service is just a task that never
terminates.

Measured on a 4-CPU single node:

    scenario        task ran (file)   future resolved
    no service      True              True
    idle service    True              False    <- ran, never resolved

Assumes the executable and the driver share a filesystem (true on one node); the
tasks write markers to a tempdir the driver then stats. Exit status is non-zero
if any task ran without its future resolving. Follow a Dragon run with
``dragon-cleanup-deprecated``.
"""

import asyncio
import os
import shlex
import sys
import tempfile
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def main():
    from radical.asyncflow import WorkflowEngine
    try:
        from rhapsody.backends import DragonExecutionBackend
    except ImportError:
        from rhapsody.backends import DragonExecutionBackendV3 as DragonExecutionBackend

    backend = await DragonExecutionBackend({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)

    workdir = tempfile.mkdtemp(prefix="exec_hang_")
    wait_for = float(os.environ.get("RESOLVE_TIMEOUT", 45))
    idle_secs = int(os.environ.get("SERVICE_SECS", 600))

    # An executable task: the body returns the shell command asyncflow runs, and
    # calling the decorated task returns a future for its completion.
    @flow.executable_task
    async def shell(command, task_description={}):  # noqa: B006 - asyncflow reads this default
        return command

    # The same, but marked a service: it touches a marker file so the driver
    # knows it is up, then idles for the rest of the run without terminating.
    @flow.executable_task(service=True)
    async def idle_service(marker, task_description={}):  # noqa: B006
        return (
            f"/bin/sh -c {shlex.quote(f'touch {shlex.quote(marker)}; sleep {idle_secs}')}"
        )

    async def file_appears(path, secs=30):
        for _ in range(int(secs / 0.25)):
            if os.path.isfile(path):
                return True
            await asyncio.sleep(0.25)
        return os.path.isfile(path)

    async def probe(label, name):
        """Submit an executable task that writes a file; report ran vs resolved."""
        out = os.path.join(workdir, f"{name}.out")
        # A plain shell command that provably completes and leaves a file.
        cmd = f"/bin/sh -c {shlex.quote(f'echo ran > {shlex.quote(out)}')}"
        fut = shell(cmd, task_description={})
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=wait_for)
            resolved = True
        except asyncio.TimeoutError:
            resolved = False
        except Exception as ex:
            say(f"   {label}: future raised {type(ex).__name__}: {ex}")
            resolved = True
        # The file is written the instant the command exits; if it is absent
        # even now, the command genuinely did not run.
        ran = await file_appears(out, secs=5)
        say(f"   {label}: ran(file on disk)={ran} future_resolved={resolved}")
        return ran, resolved

    results = {}

    say("1. executable task, nothing else running")
    results["no service"] = await probe("no service", "w1")

    say("2. start an IDLE executable service (touch + sleep), then the same task")
    marker = os.path.join(workdir, "svc_idle.up")
    idle_service(marker, task_description={})
    say(f"   idle service up: {await file_appears(marker)}")
    results["idle service"] = await probe("idle service", "w2")

    say("")
    say(f"{'scenario':<16}{'ran(file)':<12}{'resolved':<10}")
    for name, (r, s) in results.items():
        say(f"{name:<16}{str(r):<12}{str(s):<10}")
    say("")
    broken = [n for n, (r, s) in results.items() if r and not s]
    if broken:
        say(f"RAN BUT FUTURE NEVER RESOLVED: {broken}")
        say("The executable command completed (its output file is on disk) but "
            "asyncflow's future for it stayed PENDING — the completion was never "
            "delivered back to the driver.")
    else:
        say("every executable task that ran also resolved its future")

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
