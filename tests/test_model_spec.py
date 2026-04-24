"""Tests for InferenceModelSpec and ModelDeployment."""

from __future__ import annotations

import pytest

from openg2g.datacenter.config import InferenceModelSpec, ModelDeployment


def _make_spec(**overrides) -> InferenceModelSpec:
    payload = {
        "model_id": "test/Model",
        "model_label": "TestModel",
        "gpu_model": "H100",
        "task": "lm-arena-chat",
        "gpus_per_replica": 1,
        "itl_deadline_s": 0.1,
        "batch_sizes": (64, 128),
        "feasible_batch_sizes": (64, 128),
    }
    payload.update(overrides)
    return InferenceModelSpec(**payload)


class TestInferenceModelSpec:
    def test_basic(self) -> None:
        """Constructor should store label and GPUs per replica."""
        m = _make_spec(gpus_per_replica=4)
        assert m.model_label == "TestModel"
        assert m.gpus_per_replica == 4

    def test_feasible_batch_sizes_required(self) -> None:
        """feasible_batch_sizes must not be empty."""
        with pytest.raises(ValueError, match="feasible_batch_sizes must not be empty"):
            _make_spec(model_label="M", feasible_batch_sizes=())

    def test_batch_sizes_required(self) -> None:
        """batch_sizes must not be empty."""
        with pytest.raises(ValueError, match="batch_sizes must not be empty"):
            _make_spec(model_label="M", batch_sizes=(), feasible_batch_sizes=(128,))

    def test_feasible_batch_sizes_must_be_subset(self) -> None:
        """feasible_batch_sizes must be a subset of batch_sizes."""
        with pytest.raises(ValueError, match="must be a subset of batch_sizes"):
            _make_spec(model_label="M", batch_sizes=(64, 128), feasible_batch_sizes=(64, 256))

    def test_feasible_defaults_to_batch_sizes(self) -> None:
        """Omitting feasible_batch_sizes (or passing None) copies batch_sizes."""
        m = _make_spec(model_label="M", batch_sizes=(8, 16, 32), feasible_batch_sizes=None)
        assert m.feasible_batch_sizes == (8, 16, 32)

    def test_itl_deadline(self) -> None:
        """ITL deadline should be stored as given."""
        m = _make_spec(model_label="M", itl_deadline_s=0.08, batch_sizes=(128,), feasible_batch_sizes=(128,))
        assert m.itl_deadline_s == 0.08

    def test_zero_gpus_per_replica_raises(self) -> None:
        """Zero gpus_per_replica should raise ValueError."""
        with pytest.raises(ValueError, match="gpus_per_replica must be >= 1"):
            _make_spec(model_label="M", gpus_per_replica=0, batch_sizes=(128,), feasible_batch_sizes=(128,))

    def test_zero_tensor_parallel_raises(self) -> None:
        """Zero tensor_parallel should raise ValueError."""
        with pytest.raises(ValueError, match="tensor_parallel must be >= 1"):
            _make_spec(model_label="M", tensor_parallel=0, batch_sizes=(128,), feasible_batch_sizes=(128,))

    def test_zero_expert_parallel_raises(self) -> None:
        """Zero expert_parallel should raise ValueError."""
        with pytest.raises(ValueError, match="expert_parallel must be >= 1"):
            _make_spec(model_label="M", expert_parallel=0, batch_sizes=(128,), feasible_batch_sizes=(128,))

    def test_zero_itl_deadline_raises(self) -> None:
        """Zero itl_deadline_s should raise ValueError."""
        with pytest.raises(ValueError, match="itl_deadline_s must be > 0"):
            _make_spec(model_label="M", itl_deadline_s=0.0, batch_sizes=(128,), feasible_batch_sizes=(128,))

    def test_fit_exclude_batch_sizes_stored(self) -> None:
        """fit_exclude_batch_sizes should be stored as given."""
        m = _make_spec(
            model_label="M",
            batch_sizes=(8, 16, 32, 64),
            feasible_batch_sizes=(8, 16, 32, 64),
            fit_exclude_batch_sizes=(16, 32),
        )
        assert m.fit_exclude_batch_sizes == (16, 32)


class TestCacheHash:
    def test_hash_is_deterministic(self) -> None:
        """Two specs with identical measurement-relevant fields hash equal."""
        a = _make_spec(model_label="A")
        b = _make_spec(model_label="A")
        assert a.cache_hash() == b.cache_hash()

    def test_hash_ignores_simulation_time_knobs(self) -> None:
        """model_label, itl_deadline_s, and feasible_batch_sizes do not
        invalidate the on-disk measurement cache."""
        base = _make_spec(
            model_label="A", itl_deadline_s=0.1, batch_sizes=(8, 16, 32), feasible_batch_sizes=(8, 16, 32)
        )
        relabeled = _make_spec(
            model_label="B", itl_deadline_s=0.1, batch_sizes=(8, 16, 32), feasible_batch_sizes=(8, 16, 32)
        )
        re_deadlined = _make_spec(
            model_label="A", itl_deadline_s=0.2, batch_sizes=(8, 16, 32), feasible_batch_sizes=(8, 16, 32)
        )
        re_feasible = _make_spec(
            model_label="A", itl_deadline_s=0.1, batch_sizes=(8, 16, 32), feasible_batch_sizes=(16, 32)
        )
        assert relabeled.cache_hash() == base.cache_hash()
        assert re_deadlined.cache_hash() == base.cache_hash()
        assert re_feasible.cache_hash() == base.cache_hash()

    def test_hash_changes_on_measurement_fields(self) -> None:
        """Fields that drive the v3 query change the cache hash."""
        base = _make_spec(model_label="A")
        assert _make_spec(model_label="A", model_id="test/Other").cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", gpu_model="B200").cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", task="gpqa").cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", precision="fp8").cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", gpus_per_replica=2, tensor_parallel=2).cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", tensor_parallel=2).cache_hash() != base.cache_hash()
        assert _make_spec(model_label="A", expert_parallel=2).cache_hash() != base.cache_hash()
        assert (
            _make_spec(model_label="A", batch_sizes=(64, 128, 256), feasible_batch_sizes=(64, 128)).cache_hash()
            != base.cache_hash()
        )
        assert _make_spec(model_label="A", fit_exclude_batch_sizes=(64,)).cache_hash() != base.cache_hash()


class TestModelDeployment:
    def test_basic(self) -> None:
        spec = _make_spec(model_label="M")
        d = ModelDeployment(spec=spec, initial_batch_size=128)
        assert d.spec is spec
        assert d.initial_batch_size == 128

    def test_zero_batch_size_raises(self) -> None:
        spec = _make_spec(model_label="M", batch_sizes=(128,), feasible_batch_sizes=(128,))
        with pytest.raises(ValueError, match="initial_batch_size must be > 0"):
            ModelDeployment(spec=spec, initial_batch_size=0)

    def test_batch_size_not_in_feasible_raises(self) -> None:
        spec = _make_spec(model_label="M", batch_sizes=(8, 16, 32, 64), feasible_batch_sizes=(8, 16, 32, 64))
        with pytest.raises(ValueError, match=r"initial_batch_size.*must be in.*feasible_batch_sizes"):
            ModelDeployment(spec=spec, initial_batch_size=128)
