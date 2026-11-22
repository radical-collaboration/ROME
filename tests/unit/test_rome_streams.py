"""The three surrogate streams, and the client end of the same wire.

Split in three. The serving functions -- `load_promoted` and `serve_requests` -- are plain
callables over a registry on disk, so most of this runs with no ROME runtime at all. The
client adapter is tested against a fake stream manager, because what is worth testing there
is the buffering, not ROME. And one test drives the real thing: a live `rome.Manager` on the
local backend, serving across a promotion, to assert the swap happens **in place**.

Skipped without `asteer`: a sibling checkout rather than a declared ROME dependency.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from asteer.contracts import (  # noqa: E402
    ModelManifest,
    PromotionDecision,
    PromotionOutcome,
    SurrogateRequest,
    SurrogateResponse,
    to_mapping,
)
from asteer.registry import ModelRegistry  # noqa: E402

from examples.impress.small_molecule_binding.streams import (  # noqa: E402
    MODEL_B,
    RESIDENT_FROM,
    RomeSurrogateStream,
    ServedModel,
    Unavailable,
    checkpoint_recorder,
    load_promoted,
    preorg_stream,
    serve_requests,
    surrogate_stream,
)

VALUES = {"preorg_rmsd": 1.25, "induced_fit_risk": 0.4}
SIGMAS = {"preorg_rmsd": 0.3}


# -- fixtures ----------------------------------------------------------------


def _publish(root, model_id: str, version: int) -> ModelRegistry:
    registry = ModelRegistry(root)
    registry.publish(ModelManifest(
        model_id=model_id, version=version, trained_at="2026-11-21T00:00:00Z",
        corpus_size=20, git_sha="", nll=1.0, logo_r2=0.4, holdout_n=5,
        min_holdout_n=4, min_effect_size=0.1))
    return registry


def _promote(root, model_id: str, version: int) -> None:
    ModelRegistry(root).promote(PromotionDecision(
        decision_id=f"dec-{version}", model_id=model_id, to_version=version,
        outcome=PromotionOutcome.PROMOTE, decided_at="2026-11-21T00:00:00Z",
        rationale="the test's controller"))


def _build(version_dir: str, manifest: Any):
    """A predictor that reports which version produced it, so a swap is observable."""
    def predict(features):
        return ({**VALUES, "served_version": float(manifest.version)}, SIGMAS)
    return predict


def _request(model_id: str = MODEL_B, request_id: str = "req-1") -> SurrogateRequest:
    return SurrogateRequest(request_id=request_id, model_id=model_id,
                            features={"structure": "/tmp/d.pdb", "pocket_selection": "A10-A20"},
                            design_campaign_id="camp-1", node_id="nd-7")


# -- the promoted pointer is the authority -----------------------------------


def test_the_promoted_version_is_what_gets_served(tmp_path):
    _publish(tmp_path, MODEL_B, 0)
    _publish(tmp_path, MODEL_B, 1)
    _promote(tmp_path, MODEL_B, 1)
    served = load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                           build=_build)
    assert isinstance(served, ServedModel) and served.version == 1


def test_publishing_a_newer_version_changes_nothing_until_it_is_promoted(tmp_path):
    """The gap between publish and promote is CON-6, and this is it from the stream's side."""
    _publish(tmp_path, MODEL_B, 0)
    _promote(tmp_path, MODEL_B, 0)
    first = load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                          build=_build)
    _publish(tmp_path, MODEL_B, 1)
    # A publication is exactly what makes ROME call `load_func` again, which is why the
    # re-resolve matters: the trigger fires and the answer does not move.
    second = load_promoted("/some/checkpoint/v1", None, model_id=MODEL_B,
                           registry_root=str(tmp_path), build=_build)
    assert first.version == second.version == 0


def test_the_handed_checkpoint_path_is_not_what_decides(tmp_path):
    """ROME hands the newest published checkpoint. Publication is not promotion."""
    _publish(tmp_path, MODEL_B, 0)
    _publish(tmp_path, MODEL_B, 1)
    _promote(tmp_path, MODEL_B, 0)
    served = load_promoted(str(tmp_path / MODEL_B / "v1"), None, model_id=MODEL_B,
                           registry_root=str(tmp_path), build=_build)
    assert served.version == 0


def test_nothing_promoted_serves_nothing_and_says_so(tmp_path):
    _publish(tmp_path, MODEL_B, 0)
    served = load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                           build=_build)
    assert isinstance(served, Unavailable)
    assert "only the controller promotes" in served.reason
    payload = serve_requests([to_mapping(_request())], _Ctx(served))[0]
    assert "error" in payload and "preorg_rmsd" not in payload


def test_a_promoted_version_that_will_not_load_is_an_outage_not_a_crash(tmp_path):
    _publish(tmp_path, MODEL_B, 0)
    _promote(tmp_path, MODEL_B, 0)

    def broken(version_dir, manifest):
        raise FileNotFoundError(f"no weights in {version_dir}")

    served = load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                           build=broken)
    assert isinstance(served, Unavailable)
    assert "is promoted but did not load" in served.reason


# -- CON-5, by construction --------------------------------------------------


class _Ctx:
    def __init__(self, model: Any) -> None:
        self.model = model


def _served(tmp_path, version: int = 0) -> ServedModel:
    _publish(tmp_path, MODEL_B, version)
    _promote(tmp_path, MODEL_B, version)
    return load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                         build=_build)


def test_every_response_carries_the_model_and_the_version(tmp_path):
    payload = serve_requests([to_mapping(_request())], _Ctx(_served(tmp_path)))[0]
    assert payload["model_id"] == MODEL_B and payload["version"] == 0
    assert payload["values"]["preorg_rmsd"] == 1.25
    assert payload["sigmas"] == {"preorg_rmsd": 0.3}


def test_a_response_cannot_be_built_without_attribution():
    """CON-5 'by construction', asserted by trying. The contract is the only constructor
    `ServedModel.respond` uses, so what it refuses, the stream cannot emit."""
    with pytest.raises(ValueError, match="untraceable"):
        SurrogateResponse(request_id="r", model_id="", version=0)
    with pytest.raises(ValueError, match="unversioned"):
        SurrogateResponse(request_id="r", model_id=MODEL_B, version=-1)


def test_a_request_for_another_model_is_refused_rather_than_answered(tmp_path):
    payload = serve_requests([to_mapping(_request(model_id="aff_delta"))],
                             _Ctx(_served(tmp_path)))[0]
    assert "error" in payload and "CON-5" in payload["error"]


def test_a_malformed_payload_costs_one_request_and_not_the_batch(tmp_path):
    ctx = _Ctx(_served(tmp_path))
    payloads = [to_mapping(_request(request_id="good-1")), {"nonsense": True},
                to_mapping(_request(request_id="good-2"))]
    results = serve_requests(payloads, ctx)
    assert results[0]["model_id"] == MODEL_B and results[2]["model_id"] == MODEL_B
    assert "error" in results[1]


def test_a_predictor_that_raises_costs_one_request(tmp_path):
    def exploding(version_dir, manifest):
        def predict(features):
            raise RuntimeError("the pocket has no atoms")
        return predict

    _publish(tmp_path, MODEL_B, 0)
    _promote(tmp_path, MODEL_B, 0)
    served = load_promoted(None, None, model_id=MODEL_B, registry_root=str(tmp_path),
                           build=exploding)
    payload = serve_requests([to_mapping(_request())], _Ctx(served))[0]
    assert "error" in payload and "the pocket has no atoms" in payload["error"]


def test_a_task_that_never_loaded_anything_refuses(tmp_path):
    payload = serve_requests([to_mapping(_request())], _Ctx(None))[0]
    assert "error" in payload and "loaded no model" in payload["error"]


# -- the config --------------------------------------------------------------


def test_the_stream_config_names_functions_a_remote_task_can_import(tmp_path):
    """A closure would work on the local backend and silently fail to cross a process."""
    config = preorg_stream(_build, registry_root=str(tmp_path))
    for func in (config.load_func, config.process_func):
        assert func.__name__ != "<lambda>"
        assert getattr(__import__(func.__module__, fromlist=["_"]), func.__name__) is func


def test_the_registry_root_is_the_model_path_because_rome_needs_one(tmp_path):
    """`rome/stream.py:714` and `:365` call `load_func` only when a path is present, so a
    `None` here would mean the loader never ran and the stream served nothing forever."""
    config = preorg_stream(_build, registry_root=str(tmp_path))
    assert config.model_path == str(tmp_path)
    assert config.name == MODEL_B and config.num_gpus == 0
    config.validate()


def test_residency_is_a_recorded_default_rather_than_an_accident():
    assert RESIDENT_FROM[MODEL_B] == "M2", "B is called once per particle per resampling event"
    assert set(RESIDENT_FROM.values()) == {"M2", "M4"}


def test_an_override_reaches_the_config(tmp_path):
    config = surrogate_stream(MODEL_B, _build, registry_root=str(tmp_path), num_streams=3)
    assert config.num_streams == 3


# -- the checkpoint hook -----------------------------------------------------


def test_the_checkpoint_recorder_records_and_does_nothing_else():
    """ROME runs checkpoint callbacks inline on the manager's event loop inside a bare
    `except Exception: traceback.print_exc()`, so work here is blocking and swallowed."""
    recorder = checkpoint_recorder()
    recorder("/no/such/path/v1", 1)
    recorder("/no/such/path/v2", 2)
    assert (recorder.artifact, recorder.version) == ("/no/such/path/v2", 2)
    assert recorder.history == [("/no/such/path/v1", 1), ("/no/such/path/v2", 2)]


# -- the client adapter ------------------------------------------------------


class _FakeStreamManager:
    """`submit` and `get_outputs`, which is all `RomeSurrogateStream` uses."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.pending: list = []
        self.lock = threading.Lock()

    def submit(self, payload, stream=None, request_id=None):
        with self.lock:
            self.pending.append((request_id, payload))
        return request_id

    def get_outputs(self, stream=None, consume=True):
        with self.lock:
            batch, self.pending = self.pending, []
        return [{"request_id": rid, "result": self.responder(payload), "stream": stream,
                 "model_version": 0}
                for rid, payload in batch]


def _responder(payload):
    return to_mapping(SurrogateResponse(
        request_id=payload["request_id"], model_id=payload["model_id"], version=2,
        values=VALUES, sigmas=SIGMAS))


def test_the_client_decodes_one_response(tmp_path):
    client = RomeSurrogateStream(_FakeStreamManager(_responder), MODEL_B)
    response = client.score(_request())
    assert isinstance(response, SurrogateResponse)
    assert response.attribution == (MODEL_B, 2)


def test_two_threads_scoring_at_once_each_get_their_own_answer():
    """`get_outputs(consume=True)` drains the whole group, so without the buffer one caller
    would eat the other's record and then time out waiting for it."""
    client = RomeSurrogateStream(_FakeStreamManager(_responder), MODEL_B, timeout=5.0)
    got: dict = {}

    def run(n: int) -> None:
        got[n] = client.score(_request(request_id=f"req-{n}"))

    threads = [threading.Thread(target=run, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert len(got) == 6
    assert {r.request_id for r in got.values()} == {f"req-{n}" for n in range(6)}


def test_an_error_payload_becomes_a_raise_and_never_a_number():
    def refuse(payload):
        return {"error": "no promoted version", "request_id": payload["request_id"]}

    client = RomeSurrogateStream(_FakeStreamManager(refuse), MODEL_B)
    with pytest.raises(RuntimeError, match="no promoted version"):
        client.score(_request())


def test_a_silent_stream_times_out_rather_than_defaulting():
    class _Silent(_FakeStreamManager):
        def get_outputs(self, stream=None, consume=True):
            return []

    client = RomeSurrogateStream(_Silent(_responder), MODEL_B, timeout=0.2,
                                 poll_interval=0.02)
    with pytest.raises(TimeoutError, match="did not answer"):
        client.score(_request())


# -- both ends of the wire ---------------------------------------------------


def test_impress_as_client_scores_against_this_stream_unchanged(tmp_path):
    """EI.10's clients were written against a fake honouring the frozen contract. This is
    the other end of the same wire, and nothing in `asteer_design_agents` changes for it."""
    pytest.importorskip("impress_a", reason="impress-a is a sibling checkout")
    from impress_a.tools import asteer_design_agents as clients

    client = RomeSurrogateStream(_FakeStreamManager(_responder), MODEL_B)
    with clients.bound_stream(clients.MODEL_B, client):
        response = clients._ask(clients.MODEL_B, {"structure": "/tmp/d.pdb"}, None, "camp-1")
    assert response.attribution == (MODEL_B, 2)

    payload = clients._payload("preorg_score", response, {"preorg_rmsd": 1.25}, "/tmp/d.pdb")
    assert payload["model_versions"] == {"preorg_rmsd": f"{MODEL_B}@v2"}


def test_an_outage_reaches_impress_a_as_stream_unavailable(tmp_path):
    pytest.importorskip("impress_a", reason="impress-a is a sibling checkout")
    from impress_a.tools import asteer_design_agents as clients

    def refuse(payload):
        return {"error": "no promoted version", "request_id": payload["request_id"]}

    client = RomeSurrogateStream(_FakeStreamManager(refuse), MODEL_B)
    with clients.bound_stream(clients.MODEL_B, client):
        with pytest.raises(clients.StreamUnavailable):
            clients._ask(clients.MODEL_B, {"structure": "/tmp/d.pdb"}, None, None)


# -- the swap, in place, against real ROME -----------------------------------

#: Strong references, for the reason `asteer/tests/test_rome_reload.py` gives: `id()` is only
#: unique among live objects, so a freed context and a new one at the same address would read
#: as the same one -- which is precisely the conclusion drawn from it.
LOAD_CONTEXTS: list = []


def recording_load(model_path, ctx, **kwargs):
    """`load_promoted`, with the context it was handed recorded.

    Module-level because `load_func` must be importable by name; it records only because the
    local backend runs the stream task in this process.
    """
    LOAD_CONTEXTS.append(ctx)
    return load_promoted(model_path, ctx, **kwargs)


async def _drive(root) -> dict:
    from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

    import rome

    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)
    manager = rome.Manager(
        flow,
        # `min_samples` far above anything added, so no training round ever runs: this test
        # is about the promotion path, and a publish firing would confuse which trigger
        # moved the version.
        data_config=rome.DataConfig(min_samples=10**6),
        stream_configs=[preorg_stream(_build, registry_root=str(root),
                                      load_func=recording_load, poll_interval=0.02)],
    )
    await manager.start()
    seen = {}
    try:
        client = RomeSurrogateStream(manager.stream, MODEL_B, timeout=20.0)
        # In a thread, not on this loop. `score` blocks, and the stream task it is waiting
        # for is a coroutine on this same loop -- calling it here deadlocks, which is how
        # this test first failed. impress-a runs tool bodies on a thread pool, so a thread
        # is also what the deployment actually does.
        seen["before"] = (await asyncio.to_thread(
            client.score, _request(request_id="before"))).version
        seen["task_before"] = manager.stream.tasks_for(MODEL_B)[0]
        _promote(root, MODEL_B, 1)
        await manager.stream.reload_model(stream=MODEL_B)
        seen["after"] = (await asyncio.to_thread(
            client.score, _request(request_id="after"))).version
        seen["task_after"] = manager.stream.tasks_for(MODEL_B)[0]
        seen["status"] = manager.stream.tasks_for(MODEL_B)[0].status.name
    finally:
        await manager.stop()
        await flow.shutdown()
    return seen


@pytest.fixture(scope="module")
def swap(tmp_path_factory) -> dict:
    pytest.importorskip("radical.asyncflow", reason="radical.asyncflow is not installed")
    root = tmp_path_factory.mktemp("registry")
    _publish(root, MODEL_B, 0)
    _publish(root, MODEL_B, 1)
    _promote(root, MODEL_B, 0)
    LOAD_CONTEXTS.clear()
    return asyncio.run(_drive(root))


def test_the_served_version_climbs_when_the_controller_promotes(swap):
    assert swap["before"] == 0
    assert swap["after"] == 1, (
        "the promotion is the event that changes what is served; nothing was published "
        "between the two scores"
    )


def test_the_swap_happens_in_place(swap):
    """One `StreamContext` across both loads, and the task still RUNNING.

    ROME's reload path assigns the new model to `ctx.model` on the *existing* context, so one
    context holding two successive models is exactly what 'in place' means -- a restart would
    necessarily have produced a second context and a second task.
    """
    assert len(LOAD_CONTEXTS) >= 2, "the loader ran at startup and again on the reload"
    assert len({id(c) for c in LOAD_CONTEXTS}) == 1
    assert swap["task_before"] is swap["task_after"]
    assert swap["status"] == "RUNNING"


def test_the_response_version_is_the_promoted_one_not_romes_published_count(swap):
    """`ctx.model_version` is the newest published checkpoint this task has seen; the
    response carries the promoted version. The gap between them is CON-6."""
    assert swap["before"] != swap["after"]
    assert {swap["before"], swap["after"]} == {0, 1}
