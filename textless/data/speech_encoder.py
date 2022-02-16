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
    """SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations,
    and, optionally, F0.
    """

    def __init__(
        self,
        dense_model: torch.nn.Module,
        quantizer_model: torch.nn.Module,
        deduplicate: bool,
        add_bos_eos: bool = False,
        need_f0: bool = True,
        f0_normalizer: Optional[Callable] = None,
        f0_quantizer: Optional[Callable] = None,
    ):
        """Builds a SpeechEncoder instance. SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations,
        and, optionally, F0.

        Args:
            dense_model (torch.nn.Module): Dense module used to represent the audio
            quantizer_model (torch.nn.Module): Quantize module that converts dense representation into discrete tokens
            deduplicate (bool): if set, run-length encoding is applied so that repeated tokens are deduplicated
                and duration channel contains the number of repeats of the token.
            add_bos_eos (bool, optional): if set, each token sequences will be prepended with a special token (bos)
                and appended with another special token (eos).
            need_f0 (bool, optional): whether F0 stream should be returned. Estimating F0 is computationally heavy,
                consider disabling it if not needed.
            f0_normalizer (Optional[Callable], optional): A callback that allows F0 normalization (e.g., per-speaker)
            f0_quantizer (Optional[Callable], optional): F0 quantization module
        """
        super().__init__()
        self.dense_model = dense_model
        self.quantizer_model = quantizer_model

        self.deduplicate = deduplicate
        self.add_bos_eos = add_bos_eos
        self.need_f0 = need_f0
        self.f0_normalizer = f0_normalizer
        self.f0_quantizer = f0_quantizer

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
        add_bos_eos: bool = False,
        need_f0: bool = True,
        f0_normalizer: Optional[Callable] = None,
        f0_quantizer: Optional[Callable] = None,
    ) -> "SpeechEncoder":
        """Builds a SpeechEncoder instance by retrieving pre-trained dense and quantizer models specified by their parameters
        (names and vocabulary size).

        Args:
            dense_model_name (str): Name of the dense module used to represent the audio
            quantizer_model_name (str): Name of the quantizer module that converts dense representation into discrete tokens
            vocab_size (int): Specifies the codebook size
            deduplicate (bool): if set, run-length encoding is applied so that repeated tokens are deduplicated
                and duration channel contains the number of repeats of the token.
            add_bos_eos (bool, optional): if set, each token sequences will be prepended with a special token (bos)
                and appended with another special token (eos).
            need_f0 (bool, optional): whether F0 stream should be returned. Estimating F0 is computationally heavy,
                consider disabling it if not needed.
            f0_normalizer (Optional[Callable], optional): A callback that allows F0 normalization (e.g., per-speaker)
            f0_quantizer (Optional[Callable], optional): F0 quantization module
        """
        dense_model = dispatch_dense_model(dense_model_name)
        quantizer_model = dispatch_quantizer(
            dense_model_name, quantizer_model_name, vocab_size
        )

        return cls(
            dense_model,
            quantizer_model,
            deduplicate,
            add_bos_eos,
            need_f0,
            f0_normalizer,
            f0_quantizer,
        )

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: device where the speech encoder resides
        """
        return self._float_tensor.device

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: vocabulary size used for the unit stream (NB: not counting bos/eos/pad tokens)
        """
        return self.quantizer_model.vocab_size

    @property
    def f0_code_ratio(self) -> float:
        """
        Returns:
            float: F0 frames per unit frame
        """
        return self.code_hop_size / self.expected_sample_rate / F0_FRAME_SPACE

    @property
    def code_hop_size(self) -> int:
        """
        Returns:
            int: hop step size of the dense model
        """
        return self.dense_model.code_hop_size

    @property
    def expected_sample_rate(self) -> int:
        """
        int: sample rate expected by the underlying dense model
        """
        return self.dense_model.expected_sample_rate

    def maybe_resample(
        self, waveform: torch.Tensor, input_sample_rate: int
    ) -> torch.Tensor:
        """
        Takes a waveform and input rate and resamples it into the
        sample rate expected by the encoder (and underlying dense model). Does nothing
        if the sample rates coincide.
        Args:
            waveform (torch.Tensor): audio stream
            input_sample_rate (int): sample rate of the original audio

        Returns:
            torch.Tensor: audio, potentially resampled to match the expected
            sample rate of the encoder
        """
        if input_sample_rate == self.expected_sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform, input_sample_rate, self.expected_sample_rate
        )

    def forward(
        self, waveform: torch.Tensor, speaker: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Encodes a raw waveform tensor into two or three aligned & synchronised streams: pseudo-unit (token),
        duration, and pitch aka F0 streams. F0 is only provided if the SpeechEncoder instance
        was initialized with `requires_f0=True`.

        Args:
            waveform (torch.Tensor): audio to be encoded
            speaker (Optional[str], optional): speaker id to be passed to the F0 normalizer.
            Can be safely ignored if no F0 stream is requested or no per-speaker F0 normalizer
            provided.

        Returns:
            Dict[str, torch.Tensor]: dictionary with the following keys:
             * "units": contains an int tensor with the unit stream,
             * "durations": duration of each unit, measured in frames,
             * "dense": dense encoding of the audio, as provided by the underlying dense model,
             * "f0": F0 stream - only returned if `requires_f0=True` was set in the constructor.
        """
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
