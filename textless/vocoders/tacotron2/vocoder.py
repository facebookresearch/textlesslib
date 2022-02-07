# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch

from .tts_data import TacotronInputDataset
from .model import Tacotron2
from .glow import WaveGlow
from .waveglow_denoiser import Denoiser
from textless.checkpoint_manager import CHECKPOINT_MANAGER

from typing import Union


class TacotronVocoder(nn.Module):
    def __init__(
        self,
        tacotron_model_path: str,
        tacotron_dict_path: str,
        waveglow_path: str,
        max_decoder_steps: int = 2000,
        denoiser_strength: float = 0.1,
    ):
        super().__init__()
        self.max_decoder_steps = max_decoder_steps
        self.denoiser_strength = denoiser_strength
        (
            self.tacotron_model,
            self.tacotron_sample_rate,
            self.tacotron_hparams,
        ) = load_tacotron(
            tacotron_model_path=tacotron_model_path,
            code_dict_path=tacotron_dict_path,
            max_decoder_steps=self.max_decoder_steps,
        )
        self.waveglow_model, self.denoiser_model = load_waveglow_standalone(
            waveglow_path=waveglow_path,
        )
        self.tts_dataset = TacotronInputDataset(self.tacotron_hparams)
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def forward(self, units: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(units, torch.Tensor):
            units_str = " ".join([str(x) for x in units.cpu().tolist()])
        else:
            units_str = units
        tts_input = self.tts_dataset.get_tensor(units_str)
        tts_input = tts_input.to(self.device)
        _, _, aud_dn, _ = synthesize_audio(
            self.tacotron_model,
            self.waveglow_model,
            self.denoiser_model,
            tts_input.unsqueeze(0),
            strength=self.denoiser_strength,
        )
        out_audio = aud_dn[0]
        return out_audio

    @classmethod
    def by_name(
        cls,
        dense_model_name: str,
        quantizer_model_name: str,
        vocab_size: int,
        max_decoder_steps: int = 2000,
        denoiser_strength: float = 0.1,
    ):
        waveglow_path = CHECKPOINT_MANAGER.get_by_name("waveglow")

        tacotron_checkpoint_name = (
            f"{dense_model_name}-{quantizer_model_name}-{vocab_size}-tacotron"
        )
        tacotron_checkpoint_path = CHECKPOINT_MANAGER.get_by_name(
            tacotron_checkpoint_name
        )

        checkpoint_codes_name = f"{tacotron_checkpoint_name}-codes"
        tacotron_codes_path = CHECKPOINT_MANAGER.get_by_name(checkpoint_codes_name)

        return cls(
            tacotron_checkpoint_path,
            tacotron_codes_path,
            waveglow_path,
            max_decoder_steps,
            denoiser_strength,
        )

    @property
    def device(self) -> torch.device:
        return self._float_tensor.device

    @property
    def output_sample_rate(self) -> int:
        return self.tacotron_sample_rate


def synthesize_audio(model, waveglow, denoiser, inp, lab=None, strength=0.0):
    assert inp.size(0) == 1
    if lab is not None:
        lab = torch.LongTensor(1).fill_(lab)

    with torch.inference_mode():
        model_device = next(model.parameters()).device
        _, mel, _, ali, has_eos = model.inference(
            inp.to(model_device),
            lab.to(model_device) if lab is not None else None,
            ret_has_eos=True,
        )
        aud = waveglow.infer(mel.float(), sigma=0.666)
        aud_dn = denoiser(aud.half(), strength=strength).squeeze(1)
    return mel, aud, aud_dn, has_eos


def load_tacotron(tacotron_model_path, code_dict_path, max_decoder_steps):
    ckpt_dict = torch.load(tacotron_model_path, map_location=torch.device("cpu"))
    hparams = ckpt_dict["hparams"]
    hparams.code_dict = code_dict_path
    hparams.max_decoder_steps = max_decoder_steps
    sr = hparams.sampling_rate
    model = Tacotron2(hparams)
    model.load_state_dict(ckpt_dict["model_dict"])
    model = model.half()
    model = model.eval()
    return model, sr, hparams


def load_waveglow_standalone(waveglow_path, device="cpu"):
    ckpt_dict = torch.load(waveglow_path, map_location=torch.device("cpu"))
    hparams = ckpt_dict["hparams"]
    waveglow = WaveGlow(**hparams)
    waveglow.load_state_dict(ckpt_dict["model_dict"])
    waveglow = waveglow.eval()
    waveglow = waveglow.to(device)
    denoiser = Denoiser(waveglow)
    denoiser = denoiser.eval()
    return waveglow, denoiser
