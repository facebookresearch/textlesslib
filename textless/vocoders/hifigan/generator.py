# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Hifi-gan Generator and CodeGenerator modules
# Adapted from:
# - https://github.com/jik876/hifi-gan
# - https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/text_to_speech
# - https://github.com/facebookresearch/speech-resynthesis/blob/main/examples/speech_to_speech_translation/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm
from .resblock import ResBlock, init_weights, LRELU_SLOPE


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.get("model_in_dim", 80),
                cfg["upsample_initial_channel"],
                7,
                1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])
        ):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        cfg["upsample_initial_channel"] // (2**i),
                        cfg["upsample_initial_channel"] // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(
                cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class CodeGenerator(Generator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dict = nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"])
        self.multispkr = cfg.get("multispkr", None)
        self.embedder = cfg.get("embedder_params", None)

        self.multistyle = cfg.get("multistyle", None)

        if self.multispkr and not self.embedder:
            self.spkr = nn.Embedding(cfg.get("num_speakers", 200), cfg["embedding_dim"])
        elif self.embedder:
            self.spkr = nn.Linear(cfg.get("embedder_dim", 256), cfg["embedding_dim"])

        if self.multistyle:
            self.style = nn.Embedding(cfg.get("num_styles", 100), cfg["embedding_dim"])

        self.dur_predictor = None
        if cfg.get("dur_predictor_params", None):
            self.dur_predictor = VariancePredictor(
                cfg["dur_predictor_params"]["encoder_embed_dim"],
                cfg["dur_predictor_params"]["var_pred_hidden_dim"],
                cfg["dur_predictor_params"]["var_pred_kernel_size"],
                cfg["dur_predictor_params"]["var_pred_dropout"],
            )

        self.f0 = cfg.get("f0", None)
        n_f0_bin = cfg.get("f0_quant_num_bin", 0)
        self.f0_quant_embed = (
            None if n_f0_bin <= 0 else nn.Embedding(n_f0_bin, cfg["embedding_dim"])
        )

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)

        if self.dur_predictor and kwargs.get("dur_prediction", False):
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.f0:
            if self.f0_quant_embed:
                kwargs["f0"] = self.f0_quant_embed(kwargs["f0"].long()).transpose(1, 2)
            else:
                kwargs["f0"] = kwargs["f0"].unsqueeze(1)

            if x.shape[-1] < kwargs["f0"].shape[-1]:
                x = self._upsample(x, kwargs["f0"].shape[-1])
            elif x.shape[-1] > kwargs["f0"].shape[-1]:
                kwargs["f0"] = self._upsample(kwargs["f0"], x.shape[-1])
            x = torch.cat([x, kwargs["f0"]], dim=1)

        if self.multispkr:
            assert (
                "spkr" in kwargs
            ), 'require "spkr" input for multispeaker CodeHiFiGAN vocoder'
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        if self.multistyle:
            assert (
                "style" in kwargs
            ), 'require "style" input for multispeaker CodeHiFiGAN vocoder'
            style = self.style(kwargs["style"]).transpose(1, 2)
            style = self._upsample(style, x.shape[-1])
            x = torch.cat([x, style], dim=1)

        for k, feat in kwargs.items():
            if k in ["spkr", "code", "f0", "dur_prediction", "style"]:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super().forward(x)


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln1(x), p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)
