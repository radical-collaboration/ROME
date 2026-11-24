"""The ASD surrogates runner, driven on a laptop with no GPU and no Dragon runtime.

Three properties, which are the three the episode plan asks for: it constructs; the engine is
built once and handed over, so no manager builds its own; and the node inventory comes from
the backend rather than being written here a third time.

Nothing in this file needs a device, an allocation or a Dragon launch. `rome._substrate`
returns a plain `dict` for the shared store outside a runtime, which is what makes
constructing three managers on a laptop possible at all.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from examples.impress.small_molecule_binding import run_asd_surrogates_rome as runner
from examples.impress.small_molecule_binding import streams

RUNNER_SOURCE = Path(runner.__file__)


# -- the inventory -----------------------------------------------------------


class _Backend:
    """A backend that reports a topology, in rhapsody's own shape."""

    def __init__(self, topology):
        self._topology = topology

    def topology(self):
        if isinstance(self._topology, Exception):
            raise self._topology
        return self._topology


def test_the_pool_is_derived_from_the_backends_topology():
    specs = runner.node_specs(_Backend([
        {"id": "gpua001", "cores": 32, "gpus": 4},
        {"id": "gpua002", "cores": 32, "gpus": 4},
    ]))
    assert specs == [
        {"name": "gpua001", "cpus": 32.0, "gpus": 4},
        {"name": "gpua002", "cpus": 32.0, "gpus": 4},
    ]


def test_no_topology_is_reported_as_no_inventory_rather_than_filled_in():
    """The whole of item 59: a Pool invented here is the node count's third spelling."""
    assert runner.node_specs(_Backend([])) == []
    assert runner.node_specs(_Backend(None)) == []
    # A backend that cannot answer is still not a reason to make one up.
    assert runner.node_specs(_Backend(RuntimeError("no batch"))) == []


def test_a_node_with_no_workers_is_not_a_node():
    assert runner.node_specs(_Backend([{"id": "n1", "cores": 0, "gpus": 4}])) == []


@pytest.mark.parametrize("visible, probed, expected", [
    (None, 4, 4),          # unset: no restriction
    ("", 4, 0),            # empty: CUDA's own reading is "no devices"
    ("0", 4, 1),
    ("2,3", 4, 2),
    ("0,1,2,3,4,5", 4, 4), # cannot widen past what was probed
    (",,", 4, 0),
])
def test_the_gpu_count_is_narrowed_by_the_devices_the_job_was_given(visible, probed, expected):
    """NVML counts the machine's devices, not the job's: it does not honour the variable.

    A pool claiming devices the job did not get does not fail admission -- it places a task
    onto a device that is not there.
    """
    assert runner.visible_gpus(probed, visible) == expected


def test_a_released_middleware_gets_no_policy_and_says_which_half_is_missing():
    """Not a hedge: `policy=` is on an unmerged branch, and 0.7.0 is what the siblings pin."""
    policy, why = runner.resource_policy(_Backend([{"id": "n1", "cores": 8, "gpus": 0}]))
    import inspect

    from radical.asyncflow import WorkflowEngine
    has_kwarg = "policy" in inspect.signature(WorkflowEngine.create).parameters
    try:
        import radical.asyncflow.policies  # noqa: F401
        has_module = True
    except ImportError:
        has_module = False

    if has_module and has_kwarg:
        assert policy is not None, why
        assert "ResourcePolicy over 1 node" in why
    else:
        assert policy is None
        assert "policies" in why or "policy" in why
        assert "feature/policy-resource" in why or "TypeError" in why


# -- construction ------------------------------------------------------------


def _settings(tmp_path: Path) -> runner.Settings:
    root = tmp_path / "campaign"
    return runner.Settings(
        output_dir=str(root),
        label_root=str(root / "labels"),
        registry_root=str(root / "models"),
        checkpoint_root=str(root / "rome_checkpoints"),
        buffer_path=str(root / "replay.jsonl"),
        min_samples={streams.MODEL_B: 20, streams.MODEL_C: 5, streams.MODEL_A: 8},
        trainer_gpus={streams.MODEL_B: 1, streams.MODEL_C: 0, streams.MODEL_A: 1},
    )


class _Flow:
    """Stands in for the engine. Only its identity is under test."""


def test_three_managers_three_trainers_three_streams_on_a_laptop(tmp_path):
    surrogates = runner.build_surrogates(_Flow(), _settings(tmp_path))
    assert [s.model_id for s in surrogates] == [streams.MODEL_B, streams.MODEL_C,
                                                streams.MODEL_A]
    for s in surrogates:
        assert s.manager.trainer.config.trainer is not None, s.model_id
        assert len(s.manager._stream_configs) == 1
        assert s.manager._stream_configs[0].name == s.model_id


def test_every_manager_is_handed_the_one_engine_so_none_builds_its_own(tmp_path):
    """`manager.py:205` builds an engine only when nobody handed one over. CON-1."""
    flow = _Flow()
    surrogates = runner.build_surrogates(flow, _settings(tmp_path))
    assert {id(s.manager.asyncflow) for s in surrogates} == {id(flow)}
    assert all(s.manager._owns_asyncflow is False for s in surrogates)


def test_the_engine_is_constructed_in_exactly_one_place():
    """A structural check, because the property is "once in this file", not "once per call".

    Three managers over one allocation is three schedulers competing for the same nodes, and
    the resource policy attaches to an engine -- so a second `create` would also be a second
    inventory.
    """
    source = RUNNER_SOURCE.read_text()
    assert source.count("WorkflowEngine.create(") == 1, (
        "more than one engine is constructed in the runner"
    )


def test_a_manager_without_an_engine_is_refused_rather_than_given_one(tmp_path):
    with pytest.raises(ValueError, match="needs the host engine"):
        runner.build_surrogates(None, _settings(tmp_path))


def test_the_demand_is_derived_from_the_trainers_own_gpu_count(tmp_path):
    """One quantity, written once: a demand with its own literal would drift from `gpus`."""
    surrogates = {s.model_id: s for s in runner.build_surrogates(_Flow(), _settings(tmp_path))}
    for model_id, expect in ((streams.MODEL_B, 1), (streams.MODEL_C, 0), (streams.MODEL_A, 1)):
        config = surrogates[model_id].manager.trainer.config
        assert config.trainer.gpus == expect
        assert config.task_description == {"demand": {"cpus": 1.0, "gpus": expect}}


def test_no_line_in_the_runner_names_a_device_or_a_host():
    """The placement determination, as a check rather than a paragraph.

    An application declares a quantity; the middleware chooses the device. Every mention of
    a host name, a device index or a GPU affinity in this file is in prose explaining why it
    is not in code -- so the check is over code lines only.
    """
    import io
    import tokenize

    named = ("gpu_affinity", "host_name", "HOST_NAME", "infrastructure.policy",
             "placement")
    source = RUNNER_SOURCE.read_text()
    offenders = []
    # Tokenised rather than line-filtered, because every one of these words appears in this
    # file's prose explaining why it is not in its code -- and a check that cannot tell the
    # two apart is a check that would be deleted the first time it fired.
    # FSTRING_MIDDLE carries the literal text of an f-string on 3.12+, where a plain
    # `STRING` check would let `f"... placement: {why}"` through as code.
    literal = {tokenize.STRING, tokenize.COMMENT} | {
        getattr(tokenize, name) for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in literal:
            continue
        if any(word in token.string for word in named):
            offenders.append(f"{token.start[0]}: {token.string}")
    assert not offenders, offenders


# -- the two streams that refuse --------------------------------------------


def test_the_default_property_of_every_trainer_is_a_property_and_not_a_metric():
    """Item 66, fixed at the source by J6 rather than worked around by the runner.

    `affdelta.py` and `liganchor.py` both defaulted to `dg_abfe`, which is what ATM
    *reports*. A trainer filters a corpus on `LabelRecord.property_name`, which carries
    `dg_corrected`. Against a real label store the defaults selected **zero** records and
    refused every round, while every test passed -- because every test wrote its own corpus
    under the same wrong name. The runner was already passing the property explicitly, which
    is right for a runner and left the defaults wrong for everybody else.

    **A test and not an import.** These drivers are constructed in the campaign process and
    import nothing from `asteer` (CON-4); asserting the literal against the contract here
    gives the same guarantee for nothing. If `PROPERTY_FOR_METRIC` ever changes, this fails
    rather than the drivers silently disagreeing with it.
    """
    from asteer.labels_workflow import PROPERTY_FOR_METRIC

    from examples.impress.small_molecule_binding import affdelta, liganchor, preorg

    assert affdelta.DEFAULT_PROPERTY == PROPERTY_FOR_METRIC["dg_abfe"][0] == "dg_corrected"
    assert liganchor.DEFAULT_PROPERTY == PROPERTY_FOR_METRIC["dg_abfe"][0]
    assert preorg.DEFAULT_PROPERTY == PROPERTY_FOR_METRIC["preorg_rmsd"][0] == "preorg_rmsd"

    metrics = set(PROPERTY_FOR_METRIC)
    for name, driver in (("affdelta", affdelta), ("liganchor", liganchor),
                         ("preorg", preorg)):
        assert driver.DEFAULT_PROPERTY not in (metrics - {"preorg_rmsd"}), (
            f"{name} defaults to a metric name; the label store writes property names"
        )


def test_the_property_each_trainer_filters_on_is_the_one_the_label_store_writes(tmp_path):
    """The runner's half of item 66: it passes the property explicitly either way.

    `PROPERTY_FOR_METRIC` maps `dg_abfe` (what ATM reports) to `dg_corrected` (what the
    design tree holds and what a `LabelRecord` carries). The runner reads the mapping rather
    than restating it, so it cannot be a fourth place the correspondence is written -- and
    since J6 the drivers' own defaults agree with it too.
    """
    surrogates = {s.model_id: s for s in runner.build_surrogates(_Flow(), _settings(tmp_path))}
    assert runner.ABFE_PROPERTY == "dg_corrected"
    for model_id in (streams.MODEL_C, streams.MODEL_A):
        trainer = surrogates[model_id].manager.trainer.config.trainer
        assert trainer.config.property_name == runner.ABFE_PROPERTY, model_id
    b = surrogates[streams.MODEL_B].manager.trainer.config.trainer
    # Model B names all three since J12, and the runner resolves each through the same
    # mapping rather than writing any of them out -- so a change to the naming contract
    # reaches B's round the same way it reaches C's.
    assert b.config.properties == runner.PREORG_PROPERTIES
    assert b.config.properties == ("preorg_rmsd", "rotamer_entropy", "ligand_rmsf")


@pytest.mark.parametrize("build, model_id", [
    (runner.anchor_build, streams.MODEL_A),
])
def test_a_surrogate_with_no_serving_envelope_refuses_instead_of_guessing(build, model_id,
                                                                         tmp_path):
    """**Model A only, since J11 gave Model C its envelope.**

    A's refusal is permanent and J18 records why: its head is a scorer over bags of atoms
    whose output is comparable within a group only, so a single request's score is not a
    quantity, and it reports no sigma -- which `acquisition.rank` refuses a candidate for
    lacking. C's refusal was never permanent, only unimplemented: it named the composition it
    needed, and `affdelta_predictor` is that composition.
    """
    with pytest.raises(NotImplementedError) as caught:
        build(str(tmp_path))
    assert model_id in str(caught.value)
    assert "RESIDENT_FROM" in str(caught.value)


def test_model_c_now_has_an_envelope_and_refuses_an_empty_version_by_name():
    """The refusal it does keep is about the *artifact*, not about the composition.

    `affdelta_build` no longer raises `NotImplementedError`; it loads, and a version
    directory with no weights or no residual is refused for that, by name.
    """
    import pytest as _pytest

    with _pytest.raises((FileNotFoundError, ValueError)) as caught:
        runner.affdelta_build("/nonexistent/version/dir")
    assert "RESIDENT_FROM" not in str(caught.value), (
        "C's envelope exists; a missing artifact is not a missing envelope"
    )


def test_the_refusal_reaches_the_client_as_an_outage_not_a_score(tmp_path):
    """The refusal is a mechanism: `load_promoted` turns it into an `Unavailable`.

    And it cannot fire until somebody promotes a version for A, which is exactly the moment a
    guess would otherwise be served.
    """
    from asteer.contracts import ModelManifest, PromotionDecision, PromotionOutcome
    from asteer.registry import ModelRegistry

    root = tmp_path / "models"
    registry = ModelRegistry(root)
    registry.publish(ModelManifest(
        model_id=streams.MODEL_A, version=1, trained_at="2026-11-22T00:00:00Z",
        corpus_size=12, git_sha="", nll=1.0, logo_r2=0.3, holdout_n=5,
        min_holdout_n=4, min_effect_size=0.1))
    registry.promote(PromotionDecision(
        decision_id="dec-1", model_id=streams.MODEL_A, to_version=1,
        outcome=PromotionOutcome.PROMOTE, decided_at="2026-11-22T00:00:00Z",
        rationale="the test's controller"))

    served = streams.load_promoted(str(root), None, model_id=streams.MODEL_A,
                                   registry_root=str(root), build=runner.anchor_build)
    assert isinstance(served, streams.Unavailable)
    assert "did not load" in served.reason
    assert "NotImplementedError" in served.reason


# -- CON-4 -------------------------------------------------------------------


def test_importing_the_runner_pulls_in_no_torch():
    """The runner is the driver process. A torch import here keeps a CUDA context resident.

    numpy is *not* asserted against, and that is a measurement rather than a concession:
    `import rome` reaches `rome._substrate`, which imports Dragon, which imports numpy
    (backlog item 61). torch is the one that costs a device.
    """
    code = (
        "import sys; "
        "import examples.impress.small_molecule_binding.run_asd_surrogates_rome as r; "
        "print(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'torch', 'botorch', 'gpytorch', 'itgt', 'sklearn'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(RUNNER_SOURCE.parents[3]))
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", f"the runner pulled in {out.stdout.strip()}"


# -- the pump ----------------------------------------------------------------


def _vector(width: int) -> list[float]:
    """A synthetic vector of the featuriser's declared width."""
    return [0.5] + [1.0] * (width - 1)


def _label(label_id: str, prop: str, value: float, kind=None):
    from asteer.contracts import LabelKind, LabelRecord
    from asteer.features import embedding_metadata, featuriser_for_property

    return LabelRecord(
        label_id=label_id,
        kind=kind or LabelKind.MD,
        design_campaign_id="camp-1",
        node_id=f"nd-{label_id}",
        property_name=prop,
        value=value,
        produced_at="2026-11-22T00:00:00Z",
        units="angstrom",
        sigma=0.2,
        replicates=1,
        protocol="md-500ns",
        source_job_id="job-1",
        # Through the featuriser, not written by hand: the corpus row and a score request
        # agree because they share `asteer.features.embed` (D3) -- and since asteer's J4,
        # because they resolve the *same* featuriser for this property rather than sharing a
        # module-level default. `embed` has no default any more, on purpose.
        #
        # **A vector rather than a structure path, since asteer's J9.** This test is about the
        # pump -- which labels are owed, taken and retired -- and `pocket-geom-1` reads the
        # file a payload names. The fixture used to pass `/designs/{label_id}.pdb`, which was
        # never on disk and worked only because `standin-1` hashed the string. The width comes
        # from the featuriser so this keeps working when a model's featuriser changes again.
        metadata=embedding_metadata(
            {"embedding": _vector(featuriser_for_property(prop).width)},
            featuriser=featuriser_for_property(prop),
        ),
    )


def test_the_pump_moves_owed_labels_into_the_corpus_and_retires_them(tmp_path):
    from asteer.labels import LabelQueue

    settings = _settings(tmp_path)
    queue = LabelQueue(settings.label_root)
    for i in range(4):
        queue.write(_label(f"lab-{i}", "preorg_rmsd", 1.0 + i * 0.1))

    surrogates = {s.model_id: s for s in runner.build_surrogates(_Flow(), settings)}
    b = surrogates[streams.MODEL_B]

    first = runner.pump(b)
    assert first["owed"] == 4 and first["added"] == 4
    assert b.manager.data.total_count == 4

    # Retired: the same pump again is a no-op, which is what stops a round re-selecting
    # records it has already trained on.
    assert runner.pump(b) == {"model_id": streams.MODEL_B, "owed": 0, "added": 0}
    assert b.manager.data.total_count == 4


def test_each_model_is_owed_the_same_store_independently(tmp_path):
    """Three consumers, three `Consumed` sets: B taking a label does not retire it for C."""
    from asteer.contracts import LabelKind
    from asteer.labels import LabelQueue

    settings = _settings(tmp_path)
    queue = LabelQueue(settings.label_root)
    queue.write(_label("lab-b", "preorg_rmsd", 1.1))
    queue.write(_label("lab-c", runner.ABFE_PROPERTY, -9.4, kind=LabelKind.ABFE))

    surrogates = {s.model_id: s for s in runner.build_surrogates(_Flow(), settings)}
    assert runner.pump(surrogates[streams.MODEL_B])["owed"] == 2
    # C has taken nothing yet, so both rows are still owed to it.
    assert runner.pump(surrogates[streams.MODEL_C])["owed"] == 2


def test_a_crash_between_take_and_commit_leaves_every_record_owed(tmp_path):
    """The reason `take` and `commit` are two calls, asserted rather than described.

    20 to 40 physics labels is a corpus where losing one is a measurable loss, so a round
    that dies after selecting must find its records still there.
    """
    from asteer.labels import LabelQueue

    settings = _settings(tmp_path)
    queue = LabelQueue(settings.label_root)
    for i in range(3):
        queue.write(_label(f"lab-{i}", "preorg_rmsd", 1.0 + i))

    b = {s.model_id: s for s in runner.build_surrogates(_Flow(), settings)}[streams.MODEL_B]
    taken = b.corpus.take()          # selected, nothing marked
    assert len(taken) == 3
    assert len(b.corpus.take()) == 3, "a record was retired by a take that never committed"
    b.corpus.commit(taken)
    assert len(b.corpus.take()) == 0
