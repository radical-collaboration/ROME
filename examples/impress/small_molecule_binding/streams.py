"""The three surrogates, served as resident streams, and the client end of the same wire.

**One wire format, imported by both sides.** `asteer.contracts.surrogate` defines
`SurrogateRequest` and `SurrogateResponse`; impress-a's `preorg_score` and `affdelta_score`
clients import it and so does this module, so the two ends agree by construction rather than
through two hand-written adapters that drift one field at a time. EI.10 wrote those clients
against a fake honouring this contract. **Nothing in them changes when this lands, and if
something did, the contract was not frozen.**

**CON-5 is sourced here.** A `SurrogateResponse` cannot be constructed without
`(model_id, version)`, and `serve_requests` builds every response from the `ServedModel` the
loader resolved — so there is no code path that produces an attributed-to-nothing score.
impress-a's `surrogate_version_present` gate enforces over the payload what the stream already
states; it is the second line of defence, not the first.

**The loader reads the promoted pointer and ignores the path ROME hands it, and that is the
whole of CON-6 here.** ROME's reload is driven by publication: `Manager` wires
`trainer.on_checkpoint(stream.on_checkpoint)`, which stamps the newest *published* checkpoint
into the shared state and makes `maybe_reload` fire. Publication is not promotion. So
`load_promoted` takes the pointer as its authority and the handed path as a trigger only —
a round that publishes an unpromoted version re-resolves to the *same* promoted version and
changes nothing anybody is served. What does change it is the controller promoting and then
calling `Stream.reload_model()`, which is the one event that should.

Two consequences worth stating rather than discovering:

* **A stream with nothing promoted serves nothing and says so.** `ModelRegistry.promoted`
  already writes the rule down — *"`None` is not an invitation to fall back to the newest
  one"* — and a scoring client that received a default on an outage would put a number into
  the design tree that looks exactly like a score, gets ranked on, and trains the next round
  of the model that was supposed to produce it. Every absence here becomes an error payload,
  which the client turns into `StreamUnavailable`.
* **`ctx.model_version` and the served version are different numbers and will disagree.**
  ROME's is the newest published checkpoint this task has seen; ours is the promoted version
  being served. The gap between them *is* CON-6, so the response carries ours and the record
  carries ROME's, and a test asserts the response's rather than that they match.

**`on_checkpoint` records a path and a version and does no work.** ROME calls checkpoint
callbacks inline on the manager's event loop inside a bare `except Exception:
traceback.print_exc()`, so work there is both blocking and silently swallowed. The
interface-targeting example takes the same restraint for the same reason.
"""
from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import rome
from asteer.contracts import SurrogateRequest, SurrogateResponse, from_mapping, to_mapping
from asteer.registry import ModelRegistry

__all__ = [
    "MODEL_A", "MODEL_B", "MODEL_C", "RESIDENT_FROM",
    "Predict", "ServedModel", "Unavailable",
    "load_promoted", "serve_requests", "surrogate_stream",
    "anchor_stream", "preorg_stream", "affdelta_stream",
    "RomeSurrogateStream", "checkpoint_recorder",
]

MODEL_A = "lig_anchor"
MODEL_B = "pocket_preorg"
MODEL_C = "aff_delta"

#: Which milestone each surrogate is *resident* from. Residency costs a concurrent-task slot
#: permanently (item 18 measured the ceiling at 32 with 28 free), and only B is called once
#: per particle per resampling event. A and C are scored per design, which a task can afford
#: until their call rate justifies the slot. This is a dial — `num_streams` is config — and
#: it is recorded here so the default is a decision rather than an accident.
RESIDENT_FROM = {MODEL_B: "M2", MODEL_A: "M4", MODEL_C: "M4"}

#: `features` in, `(values, sigmas)` out. The three surrogates take different inputs and
#: report different metrics, so the stream layer pins the envelope and nothing else.
Predict = Callable[[Mapping[str, Any]], Tuple[Dict[str, float], Dict[str, float]]]

#: `(version_dir, manifest) -> Predict`. Required rather than defaulted: a stream that
#: silently acquired a stand-in predictor would serve plausible numbers from no model at all.
Build = Callable[[str, Any], Predict]


@dataclass(frozen=True)
class ServedModel:
    """What one stream task currently serves, and the carrier of CON-5."""

    model_id: str
    version: int
    predict: Predict
    version_dir: str = ""

    def respond(self, request: SurrogateRequest) -> SurrogateResponse:
        values, sigmas = self.predict(request.features)
        return SurrogateResponse(
            request_id=request.request_id,
            model_id=self.model_id,
            version=self.version,
            values={k: float(v) for k, v in values.items()},
            sigmas={k: float(v) for k, v in (sigmas or {}).items()},
            served_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class Unavailable:
    """Why this task is serving nothing. Held rather than raised, so the loop stays up.

    A stream that died on an unpromoted registry would have to be restarted by hand once the
    controller promoted. Holding the reason means the next reload can simply succeed.
    """

    model_id: str
    reason: str


def load_promoted(model_path: Optional[str], ctx: Any, *, model_id: str,
                  registry_root: str, build: Build) -> Any:
    """Resolve what should be served, from the pointer rather than from `model_path`.

    `model_path` is the newest **published** checkpoint, which is deliberately not what this
    reads: publication is a trainer saying *"here is a version"* and promotion is the
    controller saying *"serve it"*. The argument is kept in the signature because ROME's
    reload path passes it, and because a reload firing on publication is exactly the trigger
    that re-checks the pointer.
    """
    registry = ModelRegistry(registry_root)
    promoted = registry.promoted(model_id)
    if promoted is None:
        return Unavailable(model_id, (
            f"no promoted version of {model_id!r} in {registry_root}: "
            f"{len(registry.versions(model_id))} published. A published version is a "
            "trainer's claim; only the controller promotes (CON-6), and serving the newest "
            "instead would put an unvalidated model's scores into the design tree."
        ))
    version_dir = str(registry.version_dir(model_id, promoted.version))
    manifest = registry.read_manifest(model_id, promoted.version)
    try:
        predict = build(version_dir, manifest)
    except Exception as exc:  # noqa: BLE001 -- an unloadable version is an outage, not a crash
        return Unavailable(model_id, (
            f"{model_id} v{promoted.version} is promoted but did not load from "
            f"{version_dir}: {type(exc).__name__}: {exc}"
        ))
    return ServedModel(model_id=model_id, version=promoted.version,
                       predict=predict, version_dir=version_dir)


def serve_requests(payloads: List[Any], ctx: Any) -> List[Dict[str, Any]]:
    """One batch: a `SurrogateRequest` mapping in, a `SurrogateResponse` mapping out.

    Per-request failure rather than per-batch: one malformed payload must not cost the other
    seven their answers, and ROME's own `_process_batch` takes the same line one level up.
    """
    served = getattr(ctx, "model", None)
    out: List[Dict[str, Any]] = []
    for payload in payloads:
        out.append(_serve_one(served, payload))
    return out


def _serve_one(served: Any, payload: Any) -> Dict[str, Any]:
    try:
        request = from_mapping(SurrogateRequest, dict(payload))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"not a SurrogateRequest mapping: {type(exc).__name__}: {exc}"}
    if isinstance(served, Unavailable):
        return {"error": served.reason, "request_id": request.request_id}
    if served is None:
        return {"error": "this stream task has loaded no model at all; its `model_path` is "
                         "unset, so ROME never called `load_func`",
                "request_id": request.request_id}
    if request.model_id != served.model_id:
        # Routed to the wrong group. Refused here as well as in the client, because the two
        # checks catch different mistakes: a mistyped `model_id` and a mis-bound stream.
        return {"error": f"asked for {request.model_id!r}; this stream serves "
                         f"{served.model_id!r} (CON-5)",
                "request_id": request.request_id}
    try:
        return to_mapping(served.respond(request))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{served.model_id} v{served.version} failed to score "
                         f"{request.request_id}: {type(exc).__name__}: {exc}",
                "request_id": request.request_id}


def surrogate_stream(model_id: str, build: Build, *, registry_root: str,
                     **over: Any) -> "rome.StreamConfig":
    """One `StreamConfig` serving one surrogate from its promoted version.

    `model_path` is set to the registry root, which is not a checkpoint. ROME calls
    `load_func` only when it has a path (`stream.py:714`, `:365`), so a `None` here would
    mean the loader never ran and the stream served nothing forever. The loader does not
    read the value; it needs it to be present.

    `load_func` and `process_func` are module-level functions rather than closures on
    purpose: a stream task crossing a process boundary has to import them by name.
    """
    config: Dict[str, Any] = dict(
        name=model_id,
        kind=rome.StreamKind.INFERENCE,
        load_func=load_promoted,
        process_func=serve_requests,
        load_kwargs={"model_id": model_id, "registry_root": str(registry_root),
                     "build": build},
        model_path=str(registry_root),
        num_streams=1,
        num_gpus=0,
        batch_size=8,
        poll_interval=0.05,
    )
    config.update(over)
    return rome.StreamConfig(**config)


def anchor_stream(build: Build, *, registry_root: str, **over: Any) -> "rome.StreamConfig":
    """Model A. Resident from M4; until then A is scored inside a task."""
    return surrogate_stream(MODEL_A, build, registry_root=registry_root, **over)


def preorg_stream(build: Build, *, registry_root: str, **over: Any) -> "rome.StreamConfig":
    """Model B, the FK potential. The one surrogate that must be resident at M2."""
    return surrogate_stream(MODEL_B, build, registry_root=registry_root, **over)


def affdelta_stream(build: Build, *, registry_root: str, **over: Any) -> "rome.StreamConfig":
    """Model C. Resident from M4; a GP posterior update takes seconds."""
    return surrogate_stream(MODEL_C, build, registry_root=registry_root, **over)


# -- the client end ----------------------------------------------------------


class RomeSurrogateStream:
    """impress-a's `SurrogateStream` protocol, over a running `rome.Stream`.

    The protocol is one **synchronous** method, `score(request) -> SurrogateResponse`, and
    ROME's per-request read (`Stream.get_output`) is a coroutine. So this polls the group's
    synchronous drain instead and buffers what it did not ask for: `get_outputs(consume=True)`
    takes *every* finished record in the group, so two tool threads scoring at once would
    otherwise eat each other's answers. One lock, one buffer, per bound model.

    Blocking is correct here. impress-a runs tool bodies on a thread pool in the engine's own
    process, so a blocked `score` costs one worker and nothing else -- and the alternative, a
    default score on a slow stream, is what the whole client refuses to produce.

    **Never call this on the event loop that drives the stream.** The stream task is a
    coroutine on that loop, so a synchronous wait there blocks the very thing it is waiting
    for and the only symptom is the timeout above. A thread pool is both what the deployment
    uses and what a test has to use; `tests/unit/test_rome_streams.py` reaches it through
    `asyncio.to_thread`, which is how this was found.
    """

    def __init__(self, stream: Any, name: str, *, timeout: float = 30.0,
                 poll_interval: float = 0.02) -> None:
        self.stream = stream
        self.name = name
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self._lock = threading.Lock()
        self._buffer: Dict[str, Dict[str, Any]] = {}

    def score(self, request: SurrogateRequest) -> SurrogateResponse:
        request_id = self.stream.submit(to_mapping(request), stream=self.name,
                                        request_id=request.request_id)
        deadline = time.monotonic() + self.timeout
        while True:
            record = self._take(request_id)
            if record is not None:
                return self._decode(record, request)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"the {self.name!r} stream did not answer {request_id} within "
                    f"{self.timeout}s. The client turns this into StreamUnavailable rather "
                    "than a default score, which is the point"
                )
            time.sleep(self.poll_interval)

    def _take(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            found = self._buffer.pop(request_id, None)
            if found is not None:
                return found
            for record in self.stream.get_outputs(stream=self.name, consume=True):
                self._buffer[str(record.get("request_id"))] = record
            return self._buffer.pop(request_id, None)

    @staticmethod
    def _decode(record: Dict[str, Any], request: SurrogateRequest) -> SurrogateResponse:
        result = record.get("result")
        if isinstance(result, dict) and "error" in result:
            # Raised, never returned. A scoring client that answered an outage with a number
            # would write something into the design tree that nothing downstream can fault.
            raise RuntimeError(f"the {request.model_id!r} stream refused "
                               f"{request.request_id}: {result['error']}")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"the {request.model_id!r} stream returned {type(result).__name__}, not a "
                "SurrogateResponse mapping: both ends must import asteer.contracts.surrogate"
            )
        return from_mapping(SurrogateResponse, result)


# -- the checkpoint hook, which does nothing on purpose ----------------------


@dataclass
class checkpoint_recorder:  # noqa: N801 -- a recorder, used as a value not a type
    """`(checkpoint_path, version) -> None`, recording both and doing no work.

    ROME runs checkpoint callbacks inline on the manager's event loop inside a bare
    `except Exception: traceback.print_exc()`, so anything slow here blocks every task
    submission and anything that raises here disappears into a printed traceback while every
    log still looks healthy. The promotion decision, which is the work this tempts you to do,
    belongs to the controller and happens off this loop.
    """

    artifact: Optional[str] = None
    version: int = 0
    history: List[Tuple[str, int]] = field(default_factory=list)

    def __call__(self, checkpoint_path: str, version: int) -> None:
        self.artifact = checkpoint_path
        self.version = int(version)
        self.history.append((checkpoint_path, int(version)))
