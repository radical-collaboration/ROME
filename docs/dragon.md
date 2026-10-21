# ROME on Dragon

Status: **verified working** on Dragon 0.14.1, single node.

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop
dragon -s examples/agnostic/dummy_loop.py         # the worked example

dragon-cleanup-deprecated                         # after every Dragon run
```

`test_manager_dragon.py` brings up four inference replicas and a trainer over
real DDicts — one for the manager, one for the stream group — and asserts: 40
requests answered exactly once, work spread over replicas, 100 records surviving
four concurrent writer threads, a training round firing and publishing, the
running streams swapping onto the new checkpoint, and ROME staying inside its
key namespace in a dictionary shared with the host workflow. All pass.

These are scripts rather than pytest modules because the Dragon launcher runs a
script, not a test session. They exit non-zero on failure.

---

## What was broken, and why

### A DDict client handle is not safe to share between threads

This was a hard failure, not a slow path. ROME ran the manager's event loop
and its stream workers in one process against one `DDict` object, and the run
died partway through with:

```
dragon.fli.DragonFLIEOT: DragonFLIEOT: End of Transmission
```

raised from an unrelated later operation, which is what makes it awkward to
diagnose — the corruption and the symptom are not the same call.

Reduced to a probe (four threads, 200 operations each):

| handle strategy | result |
|---|---|
| one handle shared by all threads | 3 threads failed with `DragonFLIEOT` |
| `DDict.attach(d.serialize())` per thread | 0 errors |

A handle multiplexes an FLI channel, so concurrent use corrupts it. The fix is
`rome.utils.thread_handle`: every `Namespace` operation goes through a handle
owned by the calling thread, attached once and cached. Plain-mapping backings
(the unit-test path) are returned unchanged, so nothing else moved.

Worth being clear about the scope, since Dragon *does* handle transport for
you: passing a DDict object into a task on another node attaches the receiving
**process** automatically, and that has always worked. Threads inside one
process are the gap.

### The test suite broke as soon as Dragon was installed

`conftest.py` stubbed Dragon only when the import failed. But Dragon's API
imports fine without a runtime and then asserts on launch parameters the moment
you construct anything:

```
AssertionError: Launch parameter not initialized: GS_CD
```

So installing Dragon made plain `pytest` fail, which is exactly backwards. The
check is now whether the runtime is *up* (`DRAGON_GS_CD` in the environment),
not whether the package is importable.

---

## What was confirmed rather than fixed

### `pop` is an exactly-once claim

ROME's claim protocol is "whoever pops the key owns the item". Four threads
racing over 120 keys: **120 claimed, 120 unique, 0 duplicates.** The losers get
`DDictKeyError`, which subclasses `KeyError`, so ROME's existing handling was
already correct.

`Namespace.pop` now delegates to the backing's `pop` instead of doing
read-then-delete: one round trip instead of two, and the contention is resolved
manager-side rather than in a window between two calls. Note that
`DDict.pop(key, default)` raises regardless of the default, unlike
`dict.pop` — so the single-argument form plus `except KeyError` is the only
spelling that behaves identically on both backings.

---

## Prefix-scan cost, and why each stream group owns a dictionary

`Namespace` implements prefix scans by listing *every* key in the dictionary and
filtering in Python, because a DDict has no server-side prefix query. Measured:

| keys in the dictionary | one full `keys()` scan |
|---|---|
| 100 | 3.4 ms |
| 1 000 | 18.2 ms |
| 5 000 | 63.2 ms |

Linear, as expected. The problem was never the scan itself but what shared it:
with the corpus and the stream queues in one dictionary, every replica's poll
paid for the corpus, so **corpus growth slowed inference polling** — two rates
that have nothing to do with each other.

So each stream group now gets its own DDict, allocated by the stream manager and
released by `Stream.close()`. A group's dictionary holds only requests in flight
and results not yet collected, which does not accumulate. Measured with a real
Manager, timing a request round trip as the corpus grows:

| corpus records | keys in the group's dict | median request RTT |
|---|---|---|
| 0 | 2 | 54.8 ms |
| 500 | 2 | 58.3 ms |
| 2 000 | 2 | 58.0 ms |

Flat, and the group's dictionary stays at two keys throughout. The manager's own
dictionary is still where the corpus and the published checkpoint live; streams
read it only to notice a new checkpoint.

Supplying `StreamConfig.ddict` shares a dictionary you already own instead, in
which case ROME will not destroy it.

### What is still O(corpus)

`DataManager.total_count` counts by scanning, and `ready_to_train()` calls it on
every training-manager poll. That is far cooler than a stream poll — the trainer
polls on the order of seconds, not milliseconds — but it does grow with the
corpus, and it is why populating a corpus record-by-record while reading the
count back is quadratic.

`DDict.fetch_add` is the natural fix: an atomic distributed counter would make
this O(1) and remove the scan entirely. It is not done here because it has a
sharp edge worth deciding on deliberately — a `fetch_add` value cannot be read
back with ordinary `get`/`[]`, only with `fetch_add(key, 0)`, so that key stops
behaving like the rest of the dictionary. It also moves the corpus count from
derived-from-truth to separately-maintained, which then has to stay right across
eviction and `clear()`. The scan is slower but self-correcting, which for the
value that decides when training fires is the safer default until someone
chooses otherwise.

### `keys()` truncates silently while another client pops

Upgraded from "loud, not wrong" after measuring it properly. It is wrong.

A scan running while another client deletes keys produces manager-side
tracebacks:

```
dragon.fli.DragonFLIError: Failed to send memory over stream channel
  ... managed_memory.c: dragon_memory_get_size() :: DRAGON_MAP_KEY_NOT_FOUND
```

**The cause.** `KeysOp.perform()` snapshots the key list, then hands it to a
*daemon thread* to stream out, and the send takes no ownership
(`send_mem(key, transfer_ownership=False)`). Any `pop` that frees one of those
keys before the thread reaches it leaves a dangling descriptor. The send thread
dies, the stream terminates early, and the client's receive loop treats early
termination as `EOFError -> done` — a normal end of stream.

**So the caller gets a short list and no error.** Measured with 400 stable keys
that nobody touched, scanned while three processes popped 400 other keys
(`tests/dragon/test_keys_race_dragon.py`):

```
scans           : 21
keys() raised   : 0
scans missing a stable key : 11
worst stable-key count     : 39/400
quiesced scan   : 400/400
```

A scan returned **39 of 400** keys and reported success. Nothing was lost from
the dictionary — a quiesced scan sees all 400 — only the *view* is truncated.

**Retrying does not fix it.** Truncation only ever omits, so unioning repeated
scans is monotonically safer, but it does not converge:

| passes | complete scans | worst view |
|---|---|---|
| 1 | 3/12 | 78/400 |
| 2 | 6/12 | 175/400 |
| 3 | 6/12 | 127/400 |
| 4 | 2/4 | 229/400 |

**Prefix namespacing does not protect you either.** `Namespace.keys(prefix)`
calls `self._target.keys()` for the whole dictionary and filters in Python, so
scanning one namespace still streams every key in the dictionary, including the
ones another namespace is popping. Only a *separate DDict* isolates a scan from
a pop.

#### What this means for ROME

| dictionary | pops during a run | scans | exposure |
|---|---|---|---|
| manager (corpus, checkpoint) | none, with `max_records=None` (the default) | `get_records`, `total_count`, `unconsumed_count` | **none** |
| stream group | constant — claim-by-pop | pending counts, result drains | truncation, self-healing |

The corpus is the part that matters, because `get_records()` builds the training
shard and nothing re-checks its length — a truncated scan would silently train on
a fraction of the corpus. It is safe **only because nothing pops from that
dictionary**: eviction is the sole popper and `max_records` defaults to `None`.

> **Setting `max_records` turns on eviction, which pops from the corpus
> dictionary, which exposes every corpus scan to silent truncation.** Leave it
> unset unless you have measured that you need it.

Stream groups pop constantly, so their scans do truncate — but every stream
protocol re-polls, and a claim is verified by the `pop` itself, so a short scan
costs a poll interval rather than a result. That is why
`test_manager_dragon.py` passes all seven checks while printing these
tracebacks.

The real fix is structural: give the request queue and the result queue separate
dictionaries, so a scan of one never streams keys the other is popping. Not yet
done.

## Operational notes

- **Always run `dragon-cleanup-deprecated` after a Dragon program**, including
  after a crash or a timeout. Leftover processes stop the next run from
  starting.
- **Size the DDicts.** Dragon's default `total_mem` is 3 MiB, which a campaign
  corpus will exhaust with an allocation error from inside a manager rather than
  anything ROME can explain. `Manager` now requests 1 GiB by default and
  accepts `ddict_kwargs` to override; pass `ddict=` to share the host
  workflow's dictionary instead.
- **ROME stays in its namespace.** Every key it writes is prefixed `rome|`,
  asserted by the Dragon test against a shared dictionary.

### A never-completing task blocks result delivery for everything behind it

The last thing standing between ROME and a working loop on the multi-process
Dragon backend, and it is a backend defect rather than anything ROME does.

**The measurement.** With a stream running and a training round finished:

```
manager(0)['0-0'] -> STILL BLOCKED after 20s   # the stream service, still running
manager(0)['0-1'] -> returned in 0.12s         # the finished round's result
```

Dragon pre-registers a running task's result key, so *reading* that key blocks
until a value arrives instead of raising `KeyError`. rhapsody's monitor sweeps
its outstanding tasks in insertion order:

```python
for tuid in list(self._monitored_batches.keys()):
    ...
    try:
        result, tb, raised, stdout, stderr = self.batch.results_ddict.manager(idx)[tuid]
    except KeyError:
        continue
```

A ROME stream is a service task that never completes, so its key is
permanently pending, its read never returns, and the sweep never reaches
anything behind it. The monitor thread stays *alive* the whole time, which is
what made this hard to see.

**What it looked like.** The round ran and its checkpoint was on disk, while the
driver's future stayed pending forever and `TrainerStatus` never left `RUNNING`:

```
status=RUNNING total=100 unconsumed=100 ready=True version=0 checkpoint=None
disk={'dummy/v1/checkpoint.json': 72} | future=PENDING waited=180.0s
```

Everything else was eliminated first, each with its own probe: capacity, task
picklability, event-loop starvation (4 ms worst lag), stream poll frequency,
dataset size, the trainer in isolation, which shard the result landed in (the
right one), the stored value's shape (a clean 5-tuple), whether the monitor
thread was alive (it was), and whether missing-key reads are slow (they are not
— 0.4 ms; it is specifically a *pending* key that blocks).

**The workaround.** `TrainerConfig.result_fallback_seconds` (default 60s). A
round whose output is already on disk is treated as finished if the backend has
still not delivered its result after the grace period. The round itself was
never the problem — only the notification — so the disk is the sounder signal.
Set it to `None` to disable and wait on the backend forever.

How "on disk" is detected differs by task type:

* An **executable** round (the ProteinMPNN trainer — submitted as a command)
  writes a per-round completion marker, `train_complete`
  (`rome.trainer.TRAIN_COMPLETE_MARKER`), into its `output_dir` as its *final*
  action. The trainer polls for that marker every couple of seconds, so it
  detects completion within seconds of the round finishing rather than waiting
  out the full grace — the future never resolving costs almost nothing. The
  marker, not the checkpoint, is the signal on purpose: with
  `publish_into_repo` the checkpoint is a stable path that already exists from
  the previous round (or the initial weights), so its existence proves nothing.
  Any `as_command` trainer must write this marker last; see
  `TrainTask.as_command`.
* A **function** round has no such marker, and its future's return value is the
  real checkpoint path, so the trainer waits the full grace for the backend to
  deliver it before falling back to the output directory.

With that in place the whole loop passes on `DragonExecutionBackend`:

```
ok    every request answered exactly once
ok    work spread over replicas
ok    outputs are distinct
ok    concurrent writers lose nothing
ok    training fired and published
ok    streams swapped onto the checkpoint
ok    host workflow keys untouched
ROME works on Dragon
```

**Confirmed where.** The *stall* is confirmed on both a 4-CPU node and an NCSA
Delta GPU node — on Delta the round wrote `dummy/v1/checkpoint.json` while its
future was still pending after 180s. The *mechanism* above (the blocked read on
a pending key) has only been measured on the small node:
`tests/dragon/test_service_blocks_results_dragon.py` reduces it to plain Dragon
and rhapsody, and on Delta every task in it resolves. So the reduced repro is
not proof for a large allocation, and whether the blocked-key read is the cause
there is open.

`tests/dragon/test_result_delivery_dragon.py` answers that on whatever
allocation it is run on: it reproduces the stall (`ROME_FALLBACK=none`) and then
reports which shard the result reached, the value's shape, monitor-thread
liveness, and how long it takes to read a still-running task's key. If that read
returns quickly on your allocation, the mechanism there is something else and
the output says what.

`tests/dragon/test_executable_result_hang_dragon.py` is the minimal,
report-ready reproducer — no ROME, just Dragon + rhapsody + asyncflow. It
submits a trivial **executable** task (`sh -c 'echo … > file'`) with an idle
service running and shows the command completes (its output file lands on disk)
while asyncflow's future for it stays PENDING. The executable form makes the
defect unambiguous for an upstream issue: a shell command that exited 0 whose
completion was never delivered. `test_service_blocks_results_dragon.py` shows
the same for a `function_task`, and tabulates no-service / idle / busy.

If the mechanism does hold, the fix belongs upstream in the monitor, which
should test for a key's presence before reading it, or skip tasks it knows
cannot have completed.

**Telling a rescue from a normal completion.** The fallback logs
`publishing from disk` at WARNING when it fires. If a run passes without that
line, the backend delivered the result normally and the fallback was not
involved.

**A related trap.** `d[missing]` on a whole DDict hangs; `d.manager(i)[missing]`
raises `KeyError` promptly. Prefer the manager-scoped form when a miss is
expected.

