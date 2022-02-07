# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from librosa.util import normalize
import numpy as np
from scipy.interpolate import interp1d

F0_FRAME_SPACE = 0.005  # sec


def get_f0(audio, rate=16_000):
    assert audio.ndim == 1
    frame_length = 20.0  # ms
    to_pad = int(frame_length / 1000 * rate) // 2

    audio = normalize(audio) * 0.95
    audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)
    audio = basic.SignalObj(audio, rate)
    pitch = pYAAPT.yaapt(
        audio,
        frame_length=frame_length,
        frame_space=F0_FRAME_SPACE * 1000,
        nccf_thresh1=0.25,
        tda_frame_length=25.0,
    )
    f0 = pitch.samp_values
    return f0


def align_f0_to_durations(f0, durations, f0_code_ratio, tol=1):
    code_len = durations.sum()
    targ_len = int(f0_code_ratio * code_len)
    diff = f0.size(0) - targ_len
    assert abs(diff) <= tol, (
        f"Cannot subsample F0: |{f0.size(0)} - {f0_code_ratio}*{code_len}|"
        f" > {tol} (dur=\n{durations})"
    )
    if diff > 0:
        f0 = f0[:targ_len]
    elif diff < 0:
        f0 = torch.cat((f0, f0.new_full((-diff,), f0[-1])), 0)

    f0_offset = 0.0
    seg_f0s = []
    for dur in durations:
        f0_dur = dur.item() * f0_code_ratio
        seg_f0 = f0[int(f0_offset) : int(f0_offset + f0_dur)]
        seg_f0 = seg_f0[seg_f0 != 0]
        if len(seg_f0) == 0:
            seg_f0 = torch.tensor(0).type(seg_f0.type())
        else:
            seg_f0 = seg_f0.mean()
        seg_f0s.append(seg_f0)
        f0_offset += f0_dur

    assert int(f0_offset) == f0.size(0), f"{f0_offset} {f0.size()} {durations.sum()}"
    return torch.tensor(seg_f0s)


class SpeakerMeanNormalize:
    def __init__(self, path_to_stats, center=True, scale=False, log=True):
        self.stats = torch.load(path_to_stats)
        self.center = center
        self.scale = scale
        self.log = log

    def __call__(self, f0, speaker):
        f0 = f0.clone()
        mask = f0 != 0.0
        if self.log:
            f0[mask] = f0[mask].log()

        mean = (
            self.stats[speaker]["logf0_mean"]
            if self.log
            else self.stats[speaker]["f0_mean"]
        )
        std = (
            self.stats[speaker]["logf0_std"]
            if self.log
            else self.stats[speaker]["f0_std"]
        )

        if self.center:
            f0[mask] -= mean
        if self.scale:
            f0[mask] /= std

        return f0


class PromptNormalize:
    def __init__(self, center=True, scale=False, log=True):
        self.center = center
        self.scale = scale
        self.log = log

    def __call__(self, f0, _speaker=None):
        f0 = f0.clone()
        mask = f0 != 0.0
        if self.log:
            f0[mask] = f0[mask].log()

        if self.center:
            f0[mask] -= f0[mask].mean()
        if self.scale:
            f0[mask] /= f0[mask].std()

        return f0


class F0BinQuantizer:
    def __init__(self, bins_path):
        self.bins = torch.load(bins_path)

    def __call__(self, f0: torch.Tensor):
        bin_idx = (f0.view(-1, 1) > self.bins.view(1, -1)).long().sum(dim=1)
        return bin_idx


def trailing_silence_mask(f0):
    """
    >>> f0 = torch.tensor([1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    >>> trailing_silence_mask(f0)
    tensor([False, False, False, False,  True,  True,  True])
    """
    assert f0.ndim == 1
    mask = ((f0.flip(0) != 0.0).cumsum(0) == 0).flip(0)
    return mask


def interpolate_f0(f0):
    orig_t = np.arange(f0.shape[0])
    f0_interp = f0[:]
    ii = f0_interp != 0
    if ii.sum() > 1:
        f0_interp = interp1d(
            orig_t[ii], f0_interp[ii], bounds_error=False, kind="linear", fill_value=0
        )(orig_t)
        # f0_interp = torch.Tensor(f0_interp).type_as(f0).to(f0.device)
    return f0_interp
