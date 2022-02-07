# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import joblib
import warnings


class KMeansQuantizer(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.kmeans_model = self.load_kmeans_model(checkpoint_path)

    def forward(self, x):
        return torch.from_numpy(self.kmeans_model.predict(x.cpu().numpy())).to(
            self.device
        )

    @property
    def vocab_size(self) -> int:
        return self.kmeans_model.n_clusters

    @property
    def device(self):
        return self._float_tensor.device

    @staticmethod
    def load_kmeans_model(checkpoint_path: str):
        with open(checkpoint_path, "rb") as fd:
            with warnings.catch_warnings():
                # produces lots of version warnings which can be annoying when we have many workers
                warnings.simplefilter("ignore")
                kmeans_model = joblib.load(fd)
                # some of the GSLM checkpoints (CPC) were saved under a different scikit version
                if not hasattr(kmeans_model, "_n_threads"):
                    kmeans_model._n_threads = 40

        kmeans_model.verbose = False
        return kmeans_model
