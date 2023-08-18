# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import fairseq
import torch.nn.functional as F


class HubertFeatureReader(torch.nn.Module):
    def __init__(
        self, checkpoint_path, layer=6, feat_hop_size=320, max_chunk=100 * 16_000, lazy_load=False
    ):
        super().__init__()
        # NB: fairseq doesn't support pathlib.Path
        self.checkpoint_path = str(checkpoint_path)
        self.should_normalize = False
        self.lazy_load = lazy_load
        self.model = None
        self.layer = layer
        self.feat_hop_size = feat_hop_size
        self.max_chunk = max_chunk
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        if not self.lazy_load:
            self.load_checkpoint_()

    @torch.no_grad()  # otherwise some non-leaf nodes appear which breaks serialization
    def load_checkpoint_(self):
        model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.checkpoint_path]
        )
        self.model = model[0].eval()
        self.model = self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.should_normalize = task.cfg.normalize

    @property
    def device(self):
        return self._float_tensor.device

    @property
    def code_hop_size(self) -> int:
        return self.feat_hop_size

    @property
    def expected_sample_rate(self) -> int:
        return 16_000

    def forward(self, x):
        if self.lazy_load and self.model is None:
            self.load_checkpoint_()

        return self.get_features(x)

    @torch.inference_mode()
    def get_features(self, x):
        x = x.to(self.device)
        if self.should_normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )
            feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0).cpu()
