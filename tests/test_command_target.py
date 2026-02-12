from __future__ import annotations

import pytest

from openg2g.types import Command, CommandTarget


def test_command_target_accepts_string_and_enum() -> None:
    c1 = Command(target="datacenter", kind="set_batch_size")
    c2 = Command(target=CommandTarget.DATACENTER, kind="set_batch_size")
    assert c1.target == CommandTarget.DATACENTER
    assert c2.target == CommandTarget.DATACENTER


def test_command_target_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        Command(target="unknown", kind="noop")
