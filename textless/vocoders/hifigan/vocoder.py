# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from .generator import CodeGenerator

import json
from typing import Union
from pathlib import Path

from textless.checkpoint_manager import CHECKPOINT_MANAGER


class CodeHiFiGANVocoder(nn.Module):
    def __init__(
        self,
        hifigan_model_path: str,
        hifigan_config_path: str,
        hifigan_speaker_path: str = None,
        hifigan_style_path: str = None,
        fp16: str = False,
    ) -> None:
        super().__init__()

        # Load hifigan config
        with open(hifigan_config_path) as f:
            cfg = json.load(f)
        self.cfg = cfg

        # Load hifigan model
        self.model = CodeGenerator(cfg)
        state_dict = torch.load(hifigan_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        print("CodeHiFiGAN model loaded!")

        # Load hifigan metadata (if exists)
        self.speakers, self.styles = load_vocoder_meta(
            speakers_path=hifigan_speaker_path, styles_path=hifigan_style_path
        )

        # Useful for detecting device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def forward(
        self,
        code: torch.Tensor,
        speaker_id: int = 0,
        style_id: int = 0,
        f0: torch.Tensor = None,
        dur_prediction: bool = False,
    ) -> torch.Tensor:
        x = dict()
        x["code"] = code
        if self.model.multispkr:
            x["spkr"] = torch.LongTensor([speaker_id]).view(1, 1).to(code.device)
        if self.model.multistyle:
            x["style"] = torch.LongTensor([style_id]).view(1, 1).to(code.device)
        if self.model.f0:
            assert f0 is not None
            x["f0"] = f0.to(code.device)

        x["dur_prediction"] = dur_prediction
        if dur_prediction:
            assert (
                self.model.dur_predictor is not None
            ), "This CodeHiFiGAN model doesn't support duration prediction!"

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        return self.model(**x).detach().squeeze()

    @property
    def device(self) -> torch.device:
        return self._float_tensor.device

    @property
    def output_sample_rate(self) -> int:
        return self.cfg.get("sampling_rate", 16_000)

    @classmethod
    def by_name(
        cls,
        dense_model_name: str,
        quantizer_model_name: str,
        vocab_size: int,
        vocoder_suffix: str = None,
        speaker_meta: bool = False,
        style_meta: bool = False,
    ):

        # Get hifigan checkpoint name and path
        hifigan_checkpoint_name = (
            f"{dense_model_name}-{quantizer_model_name}-{vocab_size}-hifigan"
        )
        if vocoder_suffix is not None:
            hifigan_checkpoint_name += "-" + vocoder_suffix
        hifigan_checkpoint_path = CHECKPOINT_MANAGER.get_by_name(
            hifigan_checkpoint_name
        )

        # Get hifigan config name and path
        hifigan_config_name = f"{hifigan_checkpoint_name}-config"
        hifigan_config_path = CHECKPOINT_MANAGER.get_by_name(hifigan_config_name)

        # Get hifigan speaker metadata name and path
        hifigan_speaker_path = None
        if speaker_meta:
            hifigan_speaker_name = f"{hifigan_checkpoint_name}-speakers"
            hifigan_speaker_path = CHECKPOINT_MANAGER.get_by_name(hifigan_speaker_name)

        # Get hifigan style metadata name and path
        hifigan_style_path = None
        if style_meta:
            hifigan_style_name = f"{hifigan_checkpoint_name}-styles"
            hifigan_style_path = CHECKPOINT_MANAGER.get_by_name(hifigan_style_name)

        return cls(
            hifigan_checkpoint_path,
            hifigan_config_path,
            hifigan_speaker_path,
            hifigan_style_path,
        )


def load_vocoder_meta(speakers_path=None, styles_path=None):
    """
    Load speakers and styles of a vocoder from text files.
    """
    speakers, styles = None, None

    if speakers_path is not None and Path(speakers_path).exists():
        with open(speakers_path) as f:
            speakers = [line.strip() for line in f]
        print(f"Loaded {len(speakers)} speakers. First few speakers: {speakers[:10]}")

    if styles_path is not None and Path(styles_path).exists():
        with open(styles_path) as f:
            styles = [line.strip() for line in f]
        print(f"Loaded {len(styles)} styles. First few styles: {styles[:10]}")

    return speakers, styles


def preprocess_code(
    code: Union[str, list, torch.Tensor], deduplicate_code: bool = False
) -> torch.Tensor:
    """
    Convert the code to a long tensor.
    The code can be one of the following forms:
        - String of integers: "1 2 3"
        - List/Array of (string of) integers: [1, 2, 3]
    """
    if isinstance(code, str):
        code = code.split()
    if isinstance(code, list):
        code = list(map(int, code))
        code = torch.tensor(code)
    elif isinstance(code, np.ndarray):
        code = torch.from_numpy(code)
    code = code.long()
    if deduplicate_code:
        code = torch.unique_consecutive(code)
    return code.view(1, -1)
