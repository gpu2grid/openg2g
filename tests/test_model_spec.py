"""Tests for openg2g.models.spec: LLMInferenceModelSpec, LLMInferenceWorkload."""

from __future__ import annotations

import pytest

from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload


class TestLLMInferenceModelSpec:
    def test_basic(self) -> None:
        """Constructor should store label, replica count, and GPUs per replica."""
        m = LLMInferenceModelSpec(
            "TestModel", num_replicas=10, gpus_per_replica=4, initial_batch_size=128, itl_deadline_s=0.1
        )
        assert m.model_label == "TestModel"
        assert m.num_replicas == 10
        assert m.gpus_per_replica == 4

    def test_default_batch_sizes(self) -> None:
        """Default feasible_batch_sizes should be (128,)."""
        m = LLMInferenceModelSpec("M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1)
        assert m.feasible_batch_sizes == (128,)

    def test_custom_batch_sizes(self) -> None:
        m = LLMInferenceModelSpec(
            "M",
            num_replicas=1,
            gpus_per_replica=1,
            initial_batch_size=16,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(8, 16, 32, 64),
        )
        assert m.initial_batch_size == 16

    def test_itl_deadline(self) -> None:
        """ITL deadline should be stored as given."""
        m = LLMInferenceModelSpec("M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.08)
        assert m.itl_deadline_s == 0.08

    def test_empty_batch_sizes_raises(self) -> None:
        """Empty feasible_batch_sizes tuple should raise ValueError."""
        with pytest.raises(ValueError, match="feasible_batch_sizes must not be empty"):
            LLMInferenceModelSpec(
                "M",
                num_replicas=1,
                gpus_per_replica=1,
                initial_batch_size=128,
                itl_deadline_s=0.1,
                feasible_batch_sizes=(),
            )

    def test_negative_replicas_raises(self) -> None:
        """Negative num_replicas should raise ValueError."""
        with pytest.raises(ValueError, match="num_replicas must be >= 0"):
            LLMInferenceModelSpec("M", num_replicas=-1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1)

    def test_zero_gpus_per_replica_raises(self) -> None:
        """Zero gpus_per_replica should raise ValueError."""
        with pytest.raises(ValueError, match="gpus_per_replica must be >= 1"):
            LLMInferenceModelSpec("M", num_replicas=1, gpus_per_replica=0, initial_batch_size=128, itl_deadline_s=0.1)

    def test_zero_initial_batch_raises(self) -> None:
        """Zero initial_batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="initial_batch_size must be > 0"):
            LLMInferenceModelSpec("M", num_replicas=1, gpus_per_replica=1, initial_batch_size=0, itl_deadline_s=0.1)

    def test_zero_itl_deadline_raises(self) -> None:
        """Zero itl_deadline_s should raise ValueError."""
        with pytest.raises(ValueError, match="itl_deadline_s must be > 0"):
            LLMInferenceModelSpec("M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.0)


class TestLLMInferenceWorkload:
    def _make_models(self) -> tuple[LLMInferenceModelSpec, ...]:
        return (
            LLMInferenceModelSpec(
                "M1",
                num_replicas=10,
                gpus_per_replica=1,
                initial_batch_size=32,
                feasible_batch_sizes=(8, 16, 32),
                itl_deadline_s=0.08,
            ),
            LLMInferenceModelSpec(
                "M2",
                num_replicas=20,
                gpus_per_replica=4,
                initial_batch_size=64,
                feasible_batch_sizes=(16, 32, 64),
                itl_deadline_s=0.10,
            ),
        )

    def test_model_labels(self) -> None:
        """model_labels should return labels in declaration order."""
        w = LLMInferenceWorkload(models=self._make_models())
        assert w.model_labels == ["M1", "M2"]

    def test_total_gpus(self) -> None:
        """total_gpus should sum num_replicas * gpus_per_replica across all models."""
        w = LLMInferenceWorkload(models=self._make_models())
        assert w.total_gpus == 10 * 1 + 20 * 4

    def test_initial_batch_size_by_model(self) -> None:
        w = LLMInferenceWorkload(models=self._make_models())
        init = w.initial_batch_size_by_model
        assert init["M1"] == 32
        assert init["M2"] == 64

    def test_itl_deadline_by_model(self) -> None:
        """itl_deadline_by_model should map labels to their deadlines."""
        w = LLMInferenceWorkload(models=self._make_models())
        deadlines = w.itl_deadline_by_model
        assert deadlines == {"M1": 0.08, "M2": 0.10}

    def test_required_measured_gpus(self) -> None:
        """required_measured_gpus should map labels to gpus_per_replica."""
        w = LLMInferenceWorkload(models=self._make_models())
        rmg = w.required_measured_gpus
        assert rmg == {"M1": 1, "M2": 4}

    def test_feasible_batch_sizes_union(self) -> None:
        """feasible_batch_sizes_union should be the sorted union of all
        models' feasible batch sizes."""
        w = LLMInferenceWorkload(models=self._make_models())
        union = w.feasible_batch_sizes_union
        assert union == [8, 16, 32, 64]

    def test_empty_models_raises(self) -> None:
        """An empty models tuple should raise ValueError."""
        with pytest.raises(ValueError, match="at least one model"):
            LLMInferenceWorkload(models=())

    def test_duplicate_labels_raises(self) -> None:
        """Duplicate model labels should raise ValueError."""
        models = (
            LLMInferenceModelSpec("M1", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1),
            LLMInferenceModelSpec("M1", num_replicas=2, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1),
        )
        with pytest.raises(ValueError, match="Duplicate model labels"):
            LLMInferenceWorkload(models=models)
