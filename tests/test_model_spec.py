"""Tests for InferenceModelSpec."""

from __future__ import annotations

import pytest

from openg2g.datacenter.config import InferenceModelSpec


class TestInferenceModelSpec:
    def test_basic(self) -> None:
        """Constructor should store label, replica count, and GPUs per replica."""
        m = InferenceModelSpec(
            model_label="TestModel", num_replicas=10, gpus_per_replica=4, initial_batch_size=128, itl_deadline_s=0.1
        )
        assert m.model_label == "TestModel"
        assert m.num_replicas == 10
        assert m.gpus_per_replica == 4

    def test_default_batch_sizes(self) -> None:
        """Default feasible_batch_sizes should be (initial_batch_size,)."""
        m = InferenceModelSpec(
            model_label="M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1
        )
        assert m.feasible_batch_sizes == (128,)
        m2 = InferenceModelSpec(
            model_label="M", num_replicas=1, gpus_per_replica=1, initial_batch_size=64, itl_deadline_s=0.1
        )
        assert m2.feasible_batch_sizes == (64,)

    def test_custom_batch_sizes(self) -> None:
        m = InferenceModelSpec(
            model_label="M",
            num_replicas=1,
            gpus_per_replica=1,
            initial_batch_size=16,
            itl_deadline_s=0.1,
            feasible_batch_sizes=(8, 16, 32, 64),
        )
        assert m.initial_batch_size == 16

    def test_itl_deadline(self) -> None:
        """ITL deadline should be stored as given."""
        m = InferenceModelSpec(
            model_label="M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.08
        )
        assert m.itl_deadline_s == 0.08

    def test_initial_batch_not_in_feasible_raises(self) -> None:
        """initial_batch_size must be in feasible_batch_sizes when specified."""
        with pytest.raises(ValueError, match=r"initial_batch_size.*must be in.*feasible_batch_sizes"):
            InferenceModelSpec(
                model_label="M",
                num_replicas=1,
                gpus_per_replica=1,
                initial_batch_size=128,
                itl_deadline_s=0.1,
                feasible_batch_sizes=(8, 16, 32, 64),
            )

    def test_negative_replicas_raises(self) -> None:
        """Negative num_replicas should raise ValueError."""
        with pytest.raises(ValueError, match="num_replicas must be >= 0"):
            InferenceModelSpec(
                model_label="M", num_replicas=-1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1
            )

    def test_zero_gpus_per_replica_raises(self) -> None:
        """Zero gpus_per_replica should raise ValueError."""
        with pytest.raises(ValueError, match="gpus_per_replica must be >= 1"):
            InferenceModelSpec(
                model_label="M", num_replicas=1, gpus_per_replica=0, initial_batch_size=128, itl_deadline_s=0.1
            )

    def test_zero_initial_batch_raises(self) -> None:
        """Zero initial_batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="initial_batch_size must be > 0"):
            InferenceModelSpec(
                model_label="M", num_replicas=1, gpus_per_replica=1, initial_batch_size=0, itl_deadline_s=0.1
            )

    def test_zero_itl_deadline_raises(self) -> None:
        """Zero itl_deadline_s should raise ValueError."""
        with pytest.raises(ValueError, match="itl_deadline_s must be > 0"):
            InferenceModelSpec(
                model_label="M", num_replicas=1, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.0
            )
