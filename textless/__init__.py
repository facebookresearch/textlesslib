# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from textless.data.cpc_feature_reader import CpcFeatureReader
from textless.data.hubert_feature_reader import HubertFeatureReader
from textless.data.kmeans_quantizer import KMeansQuantizer
from textless.checkpoint_manager import CHECKPOINT_MANAGER
from textless.vocoders.tacotron2.vocoder import TacotronVocoder

DENSE_MODELS = {
    "hubert-base-ls960": HubertFeatureReader,
    "cpc-big-ll6k": CpcFeatureReader,
}

QUANTIZER_MODELS = {
    "kmeans": KMeansQuantizer,
}


# TODO: add kwargs everywhere
def dispatch_dense_model(name: str, **kwargs):
    model_class = DENSE_MODELS[name]
    checkpoint_path = CHECKPOINT_MANAGER.get_by_name(name)
    return model_class(checkpoint_path, **kwargs)


def dispatch_quantizer(dense_model_name: str, quantizer_name: str, vocab_size: int):
    quantizer_checkpoint_name = f"{dense_model_name}-{quantizer_name}-{vocab_size}"
    checkpoint_path = CHECKPOINT_MANAGER.get_by_name(quantizer_checkpoint_name)
    quantizer = QUANTIZER_MODELS[quantizer_name](checkpoint_path)
    return quantizer


def dispatch_vocoder(
    dense_model_name: str,
    quantizer_name: str,
    vocoder_name: str,
    vocab_size: int,
):
    if vocoder_name == "tacotron":
        vocoder = TacotronVocoder.by_name(
            dense_model_name,
            quantizer_name,
            vocab_size,
        )
    else:
        assert False, "Unsupported vocoder name"
    return vocoder
