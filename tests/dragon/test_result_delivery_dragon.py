"""A round runs and its future never resolves. Where did its result go?

    dragon -s tests/dragon/test_result_delivery_dragon.py

Run this on the allocation that shows the stall. It reproduces a training round
stalling with streams up, then reports, for that allocation:

  * whether the round's result reached the backend's results dictionary at all,
    and in which manager shard
  * whether the value has the shape the monitor unpacks
  * whether the monitor thread is alive
  * how long it takes to read the key of a task that is *still running* -- the
    suspected mechanism, since the monitor sweeps outstanding tasks in order and
    a blocked read on one stops every result behind it

Measured on a 4-CPU single node, the running task's key never returned while the
finished round's key read in 0.12s. That has not been confirmed on a large
allocation; if this prints a fast result for both, the mechanism there is
something else and the output says what.

Prints "published -- no stall to inspect" and exits 0 if the round completes
normally, which is the expected outcome once result_fallback_seconds is on --
set ROME_FALLBACK=none to disable it and observe the raw stall.
"""

import asyncio
import os
import sys
import tempfile
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def main():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    try:
        from rhapsody.backends import DragonExecutionBackend
    except ImportError:
        from rhapsody.backends import DragonExecutionBackendV3 as DragonExecutionBackend

    import rome
    from rome.dummy import DummyTrainer, dummy_infer, dummy_load

    backend = await DragonExecutionBackend({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)
    d = DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2)
    ckpt = tempfile.mkdtemp(prefix="rome_sh_")

    manager = rome.Manager(
        flow, ddict=d,
        data_config=rome.DataConfig(min_samples=4),
        trainer_config=rome.TrainerConfig(
            trainer=DummyTrainer(train_seconds=0.5, gpus=0),
            checkpoint_dir=ckpt, poll_interval=0.2,
            result_fallback_seconds=(
                None if os.environ.get("ROME_FALLBACK") == "none" else 60.0)),
        stream_configs=[rome.StreamConfig(
            name="infer", load_func=dummy_load, process_func=dummy_infer,
            num_streams=1, batch_size=2, poll_interval=0.05, num_gpus=0)],
    )
    await manager.start()
    for i in range(8):
        manager.add_training_data(sequence=f"s{i}", score=90.0 + i)

    say("waiting for the round to run and stall...")
    for _ in range(30):
        await asyncio.sleep(2)
        if manager.model_version >= 1:
            say("published -- no stall to inspect")
            d.destroy()
            return 0
        if any(files for _r, _dirs, files in os.walk(ckpt)):
            break
    say(f"round ran (disk has files), version={manager.model_version}, inspecting")

    # -- what the backend thinks is still outstanding ------------------------
    monitored = getattr(backend, "_monitored_batches", {})
    say(f"monitored batches still outstanding: {len(monitored)}")
    for tuid, entry in list(monitored.items())[:10]:
        batch_task, uid = entry
        midx = getattr(getattr(batch_task, "core", None), "manager_idx", "<none>")
        shard = 0 if midx == -1 else midx
        say(f"  tuid={tuid} uid={uid} manager_idx={midx} -> shard {shard}")

    # -- is the thread that delivers results still alive? --------------------
    import threading
    mt = getattr(backend, "_batch_monitor_thread", None)
    say(f"monitor thread: {mt!r} alive={getattr(mt, 'is_alive', lambda: '?')()}")
    say(f"live threads: {sorted(t.name for t in threading.enumerate())}")
    loop = getattr(backend, "_loop", None)
    say(f"backend._loop running={getattr(loop, 'is_running', lambda: '?')()} "
        f"is_driver_loop={loop is asyncio.get_running_loop()}")

    # -- what is actually in the results dictionary? -------------------------
    rd = backend.batch.results_ddict
    say(f"results_ddict type={type(rd).__name__}")
    for attr in ("n_managers", "num_managers", "manager_nodes", "_num_managers"):
        say(f"  {attr} = {getattr(rd, attr, '<absent>')!r}")
    try:
        keys = list(rd.keys())
        say(f"  whole-dict keys ({len(keys)}): {keys[:20]}")
    except Exception as ex:
        say(f"  whole-dict keys failed: {type(ex).__name__}: {ex}")

    for shard in range(4):
        try:
            mk = list(rd.manager(shard).keys())
            say(f"  shard {shard} keys ({len(mk)}): {mk[:20]}")
        except Exception as ex:
            say(f"  shard {shard}: {type(ex).__name__}: {str(ex)[:80]}")

    # THE question: the monitor sweeps tuids in order and 0-0 (the service,
    # still running) comes first. If reading its key blocks rather than raising
    # KeyError, the sweep never reaches the finished round sitting behind it.
    import threading as _th
    for tuid, _entry in list(monitored.items()):
        box = {}

        def _read(t=tuid):
            t0 = time.monotonic()
            try:
                rd.manager(0)[t]
                box["r"] = f"returned in {time.monotonic()-t0:.2f}s"
            except KeyError:
                box["r"] = f"KeyError in {time.monotonic()-t0:.2f}s"
            except Exception as ex:
                box["r"] = f"{type(ex).__name__} in {time.monotonic()-t0:.2f}s"

        th = _th.Thread(target=_read, daemon=True)
        th.start()
        th.join(timeout=20)
        say(f"  manager(0)[{tuid!r}] -> {box.get('r', 'STILL BLOCKED after 20s')}")

    # The monitor does exactly this, then unpacks into 5 fields. A wrong arity
    # is a ValueError, not a KeyError -- so it escapes the monitor's
    # `except KeyError: continue`, the tuid is never popped, and it retries
    # forever. (Use the manager-scoped API: rd[missing] on the whole dict hangs.)
    for shard_key in list(rd.manager(0).keys()):
        try:
            val = rd.manager(0)[shard_key]
        except Exception as ex:
            say(f"  read {shard_key!r} failed: {type(ex).__name__}")
            continue
        say(f"  {shard_key!r}: type={type(val).__name__} "
            f"len={len(val) if hasattr(val, '__len__') else 'n/a'}")
        say(f"    value={str(val)[:200]}")
        try:
            a, b, c, dd, e = val
            say("    unpacks into 5 fields OK")
        except ValueError as ex:
            say(f"    *** UNPACK FAILS: {ex} -- monitor raises ValueError here,")
            say("        never pops the task, and retries it forever ***")

    d.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
