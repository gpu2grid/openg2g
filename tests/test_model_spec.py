"""Tests for InferenceModelSpec and ModelDeployment."""

from __future__ import annotations

import pytest

from openg2g.datacenter.config import InferenceModelSpec, ModelDeployment


class TestInferenceModelSpec:
    def test_basic(self) -> None:
        """Constructor should store label and GPUs per replica."""
        m = InferenceModelSpec(
            model_label="TestModel",
            gpus_per_replica=4,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(64, 128),
        )
        assert m.model_label == "TestModel"
        assert m.gpus_per_replica == 4

    def test_feasible_batch_sizes_required(self) -> None:
        """feasible_batch_sizes must not be empty."""
        with pytest.raises(ValueError, match="feasible_batch_sizes must not be empty"):
            InferenceModelSpec(
                model_label="M",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=(),
            )

    def test_itl_deadline(self) -> None:
        """ITL deadline should be stored as given."""
        m = InferenceModelSpec(
            model_label="M",
            gpus_per_replica=1,
            itl_deadline_s=0.08,
            feasible_batch_sizes=(128,),
        )
        assert m.itl_deadline_s == 0.08

    def test_zero_gpus_per_replica_raises(self) -> None:
        """Zero gpus_per_replica should raise ValueError."""
        with pytest.raises(ValueError, match="gpus_per_replica must be >= 1"):
            InferenceModelSpec(
                model_label="M",
                gpus_per_replica=0,
                itl_deadline_s=0.1,
                feasible_batch_sizes=(128,),
            )

    def test_zero_itl_deadline_raises(self) -> None:
        """Zero itl_deadline_s should raise ValueError."""
        with pytest.raises(ValueError, match="itl_deadline_s must be > 0"):
            InferenceModelSpec(
                model_label="M",
                gpus_per_replica=1,
                itl_deadline_s=0.0,
                feasible_batch_sizes=(128,),
            )


class TestModelDeployment:
    def test_basic(self) -> None:
        spec = InferenceModelSpec(
            model_label="M",
            gpus_per_replica=1,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(64, 128),
        )
        d = ModelDeployment(spec=spec, num_replicas=10, initial_batch_size=128)
        assert d.spec is spec
        assert d.num_replicas == 10
        assert d.initial_batch_size == 128

    def test_negative_replicas_raises(self) -> None:
        spec = InferenceModelSpec(
            model_label="M",
            gpus_per_replica=1,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(128,),
        )
        with pytest.raises(ValueError, match="num_replicas must be >= 0"):
            ModelDeployment(spec=spec, num_replicas=-1, initial_batch_size=128)

    def test_zero_batch_size_raises(self) -> None:
        spec = InferenceModelSpec(
            model_label="M",
            gpus_per_replica=1,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(128,),
        )
        with pytest.raises(ValueError, match="initial_batch_size must be > 0"):
            ModelDeployment(spec=spec, num_replicas=10, initial_batch_size=0)

    def test_batch_size_not_in_feasible_raises(self) -> None:
        spec = InferenceModelSpec(
            model_label="M",
            gpus_per_replica=1,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(8, 16, 32, 64),
        )
        with pytest.raises(ValueError, match=r"initial_batch_size.*must be in.*feasible_batch_sizes"):
            ModelDeployment(spec=spec, num_replicas=10, initial_batch_size=128)
