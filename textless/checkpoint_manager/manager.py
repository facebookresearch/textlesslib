# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from dataclasses import dataclass
from torch.hub import download_url_to_file
import pathlib


@dataclass
class Checkpoint:
    name: str
    remote_path: str
    fname: str
    sha256: str


class CheckpointManager:
    def __init__(self, disk_root: Union[str, pathlib.Path] = "~/.textless/"):
        self.disk_root = pathlib.Path(disk_root).expanduser().resolve()
        if not self.disk_root.exists():
            self.disk_root.mkdir()

        self.storage: dict[str, Checkpoint] = {}

    def add_checkpoint(self, checkpoint: Checkpoint) -> None:
        name = checkpoint.name
        assert name not in self.storage
        self.storage[name] = checkpoint

    def download_by_name(self, name: str) -> None:
        checkpoint = self.storage[name]
        download_url_to_file(
            url=checkpoint.remote_path,
            dst=(self.disk_root / checkpoint.fname),
            hash_prefix=checkpoint.sha256,
            progress=True,
        )

    def get_by_name(self, name: str, download_if_needed: bool = True) -> pathlib.Path:
        checkpoint = self.storage[name]
        disk_name = self.disk_root / checkpoint.fname

        if not disk_name.exists():
            if download_if_needed:
                self.download_by_name(name)
            else:
                raise FileNotFoundError(
                    f"Checkpoint {checkpoint} was not found locally at {disk_name}, please set `allow_download` flag"
                )
        return disk_name

    def set_root(self, new_root: Union[str, pathlib.Path]) -> None:
        self.disk_root = pathlib.Path(new_root)
