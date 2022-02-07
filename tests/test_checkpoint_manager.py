# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from textless.checkpoint_manager import CHECKPOINT_MANAGER
import tempfile
import pathlib
import pytest


def test_checkpoint_manager():
    codes = CHECKPOINT_MANAGER.get_by_name(
        "hubert-base-ls960-kmeans-50-tacotron-codes", download_if_needed=True
    )
    assert pathlib.Path(codes).exists()

    with pytest.raises(KeyError):
        codes = CHECKPOINT_MANAGER.get_by_name("123", download_if_needed=True)


def test_changing_root():
    name = "hubert-base-ls960-kmeans-50-tacotron-codes"

    with tempfile.TemporaryDirectory() as tmpdir:
        CHECKPOINT_MANAGER.set_root(tmpdir)
        with pytest.raises(FileNotFoundError):
            CHECKPOINT_MANAGER.get_by_name(name, download_if_needed=False)

        CHECKPOINT_MANAGER.get_by_name(name, download_if_needed=True)
        assert (pathlib.Path(tmpdir) / CHECKPOINT_MANAGER.storage[name].fname).exists()
