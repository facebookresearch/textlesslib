# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from textless.data.quantized_datasets import QuantizedLibriSpeech
from textless.data.speech_encoder import SpeechEncoder


def test_quantized_librispeech():
    url = "dev-clean"
    root = "./data"

    pathlib.Path(root).mkdir(exist_ok=True)

    dense_model_name = "hubert-base-ls960"
    quantizer_name = "kmeans"
    vocab_size = 100

    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        need_f0=True,
        deduplicate=True,
        f0_normalizer=None,
        f0_quantizer=None,
    )

    quantized_dataset = QuantizedLibriSpeech(
        root=root, speech_encoder=encoder, url=url, download=True
    )
    item = quantized_dataset[0]

    # checking a few invariants
    assert item["units"].size(0) == item["durations"].size(0) == item["f0"].size(0)
    assert item["durations"].sum().item() == item["dense"].size(0)
