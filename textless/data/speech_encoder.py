# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn
import torchaudio
from typing import Optional, Callable, Dict
from textless import dispatch_dense_model, dispatch_quantizer
from textless import dispatch_dense_model
from textless.data.f0_preprocess import get_f0, align_f0_to_durations, F0_FRAME_SPACE
from .collater_utils import wrap_bos_eos


def get_streams(
    waveform,
    speaker,
    dense_model,
    quantizer_model,
    f0_normalizer,
    f0_quantizer,
    need_f0,
    deduplicate,
    f0_code_ratio,
):
    if waveform.ndim > 1:
        waveform = waveform.mean(0)

    dense_features = dense_model(waveform)
    units = quantizer_model(dense_features)

    if deduplicate:
        units, durations = torch.unique_consecutive(units, return_counts=True)
    else:
        durations = torch.ones_like(units)

    if need_f0:
        f0 = get_f0(waveform.cpu().numpy())
        f0 = torch.from_numpy(f0).float()

        if f0_normalizer:
            f0 = f0_normalizer(f0, speaker)
        tol = 5 * f0_code_ratio
        f0 = align_f0_to_durations(f0, durations, f0_code_ratio, tol)
        if f0_quantizer:
            f0 = f0_quantizer(f0)
    else:
        f0 = None

    return units, durations, f0, dense_features


class SpeechEncoder(torch.nn.Module):
    def __init__(
        self,
        dense_model: torch.nn.Module,
        quantizer_model: torch.nn.Module,
        deduplicate: bool,
        download: Optional[bool] = False,
        add_bos_eos: bool = False,
        need_f0: bool = True,
        f0_normalizer: Optional[Callable] = None,
        f0_quantizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.dense_model = dense_model
        self.quantizer_model = quantizer_model

        self.deduplicate = deduplicate
        self.sampling_rate = 16_000
        self.add_bos_eos = add_bos_eos
        self.need_f0 = need_f0
        self.f0_normalizer = f0_normalizer
        self.f0_quantizer = f0_quantizer

        self.download = download

        self.unit_vocab_size = self.quantizer_model.vocab_size

        self.register_buffer(
            "bos", torch.tensor([self.unit_vocab_size], dtype=torch.int)
        )
        self.register_buffer(
            "eos", torch.tensor([self.unit_vocab_size + 1], dtype=torch.int)
        )

        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @classmethod
    def by_name(
        cls,
        dense_model_name: str,
        quantizer_model_name: str,
        vocab_size: int,
        deduplicate: bool,
        download: Optional[bool] = False,
        add_bos_eos: bool = False,
        need_f0: bool = True,
        f0_normalizer: Optional[Callable] = None,
        f0_quantizer: Optional[Callable] = None,
    ) -> "SpeechEncoder":
        dense_model = dispatch_dense_model(dense_model_name)
        quantizer_model = dispatch_quantizer(
            dense_model_name, quantizer_model_name, vocab_size
        )

        return cls(
            dense_model,
            quantizer_model,
            deduplicate,
            download,
            add_bos_eos,
            need_f0,
            f0_normalizer,
            f0_quantizer,
        )

    @property
    def device(self):
        return self._float_tensor.device

    @property
    def vocab_size(self):
        return self.quantizer_model.vocab_size

    @property
    def f0_code_ratio(self):
        return self.code_hop_size / self.sampling_rate / F0_FRAME_SPACE

    @property
    def code_hop_size(self) -> int:
        return self.dense_model.code_hop_size

    @property
    def expected_sample_rate(self) -> int:
        return self.dense_model.expected_sample_rate

    def maybe_resample(
        self, waveform: torch.Tensor, input_sample_rate: int
    ) -> torch.Tensor:
        if input_sample_rate == self.expected_sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform, input_sample_rate, self.expected_sample_rate
        )

    def forward(
        self, waveform: torch.Tensor, speaker: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        units, durations, f0, dense_features = get_streams(
            waveform,
            speaker,
            self.dense_model,
            self.quantizer_model,
            self.f0_normalizer,
            self.f0_quantizer,
            self.need_f0,
            self.deduplicate,
            self.f0_code_ratio,
        )

        if self.add_bos_eos:
            units, durations, f0, dense_features = wrap_bos_eos(
                units, durations, f0, dense_features, self.bos, self.eos
            )

        item = {
            "units": units.to(self.device),
            "durations": durations.to(self.device),
            "dense": dense_features,
        }
        if f0 is not None:
            item["f0"] = f0

        return item
