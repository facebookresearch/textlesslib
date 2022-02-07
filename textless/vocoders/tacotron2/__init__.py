# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import textless.vocoders.tacotron2 as tacotron2
import sys
from textless.checkpoint_manager import CHECKPOINT_MANAGER
from .waveglow_denoiser import Denoiser
from .model import Tacotron2
from .tts_data import TacotronInputDataset


def get_waveglow(download_if_needed=True):

    waveglow_path = CHECKPOINT_MANAGER.get_by_name(
        "waveglow", download_if_needed=download_if_needed
    )

    sys.path.append(tacotron2.__path__[0])
    waveglow = torch.load(waveglow_path)["model"]
    sys.path.pop()

    waveglow = waveglow.cuda().eval()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser


def load_tacotron(model_name, max_decoder_steps, download_if_needed=True):
    tacotron_path = CHECKPOINT_MANAGER.get_by_name(
        model_name, download_if_needed=download_if_needed
    )
    ckpt_dict = torch.load(tacotron_path)

    hparams = ckpt_dict["hparams"]
    codes_path = CHECKPOINT_MANAGER.get_by_name(
        f"{model_name}-codes", download_if_needed=download_if_needed
    )
    hparams.code_dict = codes_path

    hparams.max_decoder_steps = max_decoder_steps
    model = Tacotron2(hparams)
    model.load_state_dict(ckpt_dict["model_dict"])
    model = model.cuda().eval().half()

    tts_dataset = TacotronInputDataset(hparams)

    return model, tts_dataset


def synthesize_audio(
    units, model, tts_dataset, waveglow, denoiser, lab=None, denoiser_strength=0.0
):
    quantized_units_str = " ".join(map(str, units.tolist()))
    tokens = tts_dataset.get_tensor(quantized_units_str).cuda().unsqueeze(0)

    if lab is not None:
        lab = torch.LongTensor(1).cuda().fill_(lab)

    with torch.no_grad():
        _, mel, _, ali, has_eos = model.inference(tokens, lab, ret_has_eos=True)
        mel = mel.float()
        audio = waveglow.infer(mel, sigma=0.666)
        denoised_audio = denoiser(audio, strength=denoiser_strength).squeeze(1)
    return mel, audio, denoised_audio, has_eos
