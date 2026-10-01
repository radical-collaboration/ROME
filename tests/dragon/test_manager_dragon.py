"""Dragon smoke test: the whole ROME loop on a real DDict.

    dragon -s tests/dragon/test_manager_dragon.py

Not a pytest module — the Dragon launcher runs a script, not a test session, so
this is a plain script that exits non-zero on failure. The pytest suite covers
the same logic against a plain dict; what only Dragon can answer is whether the
concurrency assumptions hold against a real distributed dictionary:

* many threads hitting one dictionary (each needs its own client handle, which
  :func:`rome.utils.thread_handle` provides — sharing one corrupts its channel);
* ``pop`` as an exactly-once claim under contention;
* records surviving concurrent writers;
* a prefix scan running while another thread deletes the keys it is streaming.

Always follow a Dragon run with ``dragon-cleanup-deprecated``.
"""

import os
import sys

# `dragon -s` puts this script's directory on sys.path, and the Dragon
# execution backend re-runs the script from a different working directory, so
# neither cwd nor sys.path[0] can be relied on to find the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import tempfile
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"ok    {name}")



def _checkpoint_tree(root):
    """What a round actually left on disk, as {relative path: bytes}."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                out[os.path.relpath(full, root)] = os.path.getsize(full)
            except OSError:
                pass
    return out or "<no files>"


def _future_state(fut):
    if fut is None:
        return "<never submitted>"
    try:
        if not fut.done():
            return "PENDING"
        exc = fut.exception()
        return f"RAISED {type(exc).__name__}: {exc}" if exc else f"done -> {fut.result()!r}"
    except Exception as ex:  # cancelled, or not an asyncio future
        return f"<uninspectable: {type(ex).__name__}>"


async def settle(predicate, timeout=60.0, interval=0.25):
    """Wait on wall-clock time, not iteration count.

    A prefix scan against a dictionary that stream replicas are popping from
    takes far longer than the poll interval, so counting iterations makes the
    timeout effectively unbounded and a failure looks like a hang.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


async def run():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    try:
        from rhapsody.backends import DragonExecutionBackend
    except ImportError:
        from rhapsody.backends import DragonExecutionBackendV3 as DragonExecutionBackend

    import rome
    from rome.dummy import DummyTrainer, dummy_infer, dummy_load

    backend = await DragonExecutionBackend(
        {"results_ddict_mem": int(os.environ.get("ROME_RESULTS_MEM", 512 * 1024 ** 2))}
    )
    flow = await WorkflowEngine.create(backend=backend)

    # Shared with a "host workflow" key, to prove ROME stays in its namespace.
    ddict = DDict(managers_per_node=1, n_nodes=1,
                  total_mem=int(os.environ.get("ROME_DDICT_MEM", 256 * 1024 ** 2)))
    ddict["host_workflow_state"] = {"campaign": "smoke"}

    checkpoint_dir = tempfile.mkdtemp(prefix="rome_dragon_")
    trainer = DummyTrainer(train_seconds=0.5,
                           gpus=int(os.environ.get("ROME_TRAIN_GPUS", 0)))

    manager = rome.Manager(
        flow,
        ddict=ddict,
        data_config=rome.DataConfig(min_samples=8),
        trainer_config=rome.TrainerConfig(
            trainer=trainer, checkpoint_dir=checkpoint_dir, poll_interval=0.2
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer",
                load_func=dummy_load,
                process_func=dummy_infer,
                # A stream replica is a *service* task that holds its execution
                # slot for the whole run, and the training round needs a slot of
                # its own. So the loop needs num_streams + 1 concurrent slots; on
                # the 2-service allocation measured by
                # test_task_capacity_dragon.py, that means num_streams=1 (one
                # replica + the round). Default to 1 so this whole-loop smoke
                # test completes on the minimum allocation instead of the round
                # starving forever behind the streams (an empty checkpoint dir
                # with the future PENDING — the fallback publishes from disk, but
                # only for a round that actually ran). Raise ROME_STREAM_REPLICAS
                # on a bigger allocation to exercise multi-replica spread.
                num_streams=int(os.environ.get("ROME_STREAM_REPLICAS", 1)),
                batch_size=2,
                poll_interval=0.02,
                # This box has no GPUs; the default num_gpus=1 puts
                # gpus_per_rank into the task description.
                num_gpus=int(os.environ.get("ROME_STREAM_GPUS", 0)),
            )
        ],
    )

    try:
        await manager.start()

        # -- streams under contention: 4 replicas, one shared dictionary ----
        n_req = int(os.environ.get("ROME_STREAM_REQUESTS", 40))
        request_ids = manager.stream.submit_batch([f"p{i}" for i in range(n_req)])

        # Wait for the requests to be claimed *first*. Reading results while
        # replicas are still popping requests means scanning a dictionary that
        # is being deleted from, and Dragon's keys() silently truncates under
        # that -- so a complete result set reads as a partial one. Once the
        # request queue is empty nothing pops, and the scan is exact.
        drained = await settle(lambda: manager.stream.pending() == 0, timeout=120)
        got = drained and await settle(
            lambda: len(manager.stream.get_outputs(consume=False)) == n_req, timeout=120
        )
        records = manager.stream.get_outputs()

        def every_request_answered_exactly_once():
            statuses = {t.index: t.status.name for t in manager.stream.stream_tasks}
            assert drained, (f"requests never drained "
                             f"({manager.stream.pending()} left), streams={statuses}")
            assert got, f"only {len(records)}/{n_req} results arrived"
            assert len(records) == n_req, len(records)
            returned = [r["request_id"] for r in records]
            assert len(set(returned)) == n_req, "a request was answered twice"
            assert set(returned) == set(request_ids), "result ids do not match requests"

        def work_reached_several_replicas():
            replicas = {r["stream_index"] for r in records}
            n_streams = len(manager.stream.stream_tasks)
            if n_streams < 2:
                # The backend caps concurrent long-lived tasks; on a small
                # allocation there is only one replica to spread over.
                assert replicas == {0}, replicas
                return
            assert len(replicas) > 1, f"only replica {replicas} did any work"

        def outputs_are_distinct():
            results = [r["result"] for r in records]
            assert len(set(results)) == n_req, "a stream echoed a cached result"

        check("every request answered exactly once", every_request_answered_exactly_once)
        check("work spread over replicas", work_reached_several_replicas)
        check("outputs are distinct", outputs_are_distinct)

        # -- concurrent writers into the corpus -----------------------------
        def concurrent_adds_all_survive():
            """Threads adding at once is the case key-per-record exists for."""
            before = manager.data.total_count
            errors = []

            def writer(tag):
                try:
                    for i in range(25):
                        manager.add_training_data(completion=f"{tag}-{i}", score=1.0)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    traceback.print_exc()

            threads = [threading.Thread(target=writer, args=(f"w{t}",)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, errors[0]
            added = manager.data.total_count - before
            assert added == 100, f"{added}/100 records survived concurrent writers"

        check("concurrent writers lose nothing", concurrent_adds_all_survive)

        # -- the closed loop -------------------------------------------------
        train_wait = float(os.environ.get("ROME_TRAIN_WAIT", 180))
        reached = await settle(lambda: manager.model_version >= 1,
                               timeout=train_wait)

        def training_fired_and_published():
            # A bare "no round completed" says nothing about which link broke,
            # and the counts here are prefix scans that Dragon can truncate --
            # so report them rather than leave it to guesswork.
            data = manager.data
            assert reached, (
                "no training round completed | "
                f"status={manager.get_training_status().name} "
                f"total={data.total_count} consumed={data.consumed_count} "
                f"unconsumed={data.unconsumed_count} "
                f"min_samples={data.config.min_samples} "
                f"ready={data.ready_to_train()} "
                f"version={manager.model_version} "
                f"checkpoint={manager.get_current_model()!r} | "
                # The driver creates the round's output directory before
                # submitting, but only the task body writes a file into it. So
                # files on disk mean the round ran and its result never came
                # back; an empty tree means it never ran at all.
                f"disk={_checkpoint_tree(checkpoint_dir)} | "
                f"future={_future_state(manager.trainer._round_fut)} "
                f"waited={train_wait}s"
            )
            published = manager.get_current_model()
            assert published, "no checkpoint was published"
            # NOT trainer.rounds: the round runs in another process, so what it
            # records on the trainer object never comes back to the driver.
            # The checkpoint on disk is the only evidence that crosses.
            assert _checkpoint_tree(checkpoint_dir) != "<no files>", \
                f"published {published!r} but nothing was written under it"

        check("training fired and published", training_fired_and_published)

        version = manager.model_version
        after = await manager.stream.get_output(
            manager.stream.submit("after-training"), timeout=60
        )

        def streams_swapped_onto_the_new_checkpoint():
            assert after is not None, "stream stopped answering after a round"
            assert after["model_version"] >= version, (
                f"stream still on v{after['model_version']}, published v{version}"
            )

        check("streams swapped onto the checkpoint", streams_swapped_onto_the_new_checkpoint)

        # -- namespacing -----------------------------------------------------
        def host_workflow_keys_untouched():
            assert ddict["host_workflow_state"] == {"campaign": "smoke"}
            stray = [
                k for k in ddict.keys()
                if isinstance(k, str) and not k.startswith("rome|")
                and k != "host_workflow_state"
            ]
            assert not stray, f"ROME wrote outside its namespace: {stray[:5]}"

        check("host workflow keys untouched", host_workflow_keys_untouched)

        await manager.stop()
    finally:
        try:
            await flow.shutdown()
        except Exception:
            traceback.print_exc()
        ddict.destroy()


def main():
    try:
        asyncio.run(run())
    except Exception:
        traceback.print_exc()
        FAILURES.append("run() raised")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("ROME works on Dragon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
