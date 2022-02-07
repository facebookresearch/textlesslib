# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from textless import dispatch_dense_model, dispatch_quantizer
from textless.data.speech_encoder import SpeechEncoder
import torch
from itertools import product

from textless.vocoders.tacotron2.vocoder import TacotronVocoder


def test_model_dispatch():
    dense_model_name = "hubert-base-ls960"
    quantizer_name = "kmeans"
    vocab_size = 100

    # getting dense model
    dense_model = dispatch_dense_model(dense_model_name)
    assert isinstance(dense_model, torch.nn.Module)

    # getting a quantizer for it
    assert (
        dispatch_quantizer(dense_model_name, quantizer_name, vocab_size=vocab_size)
        is not None
    )

    with pytest.raises(KeyError):
        assert dispatch_quantizer(dense_model_name, quantizer_name, vocab_size=101)

    # getting a vocoder for it
    assert (
        TacotronVocoder.by_name(
            dense_model_name=dense_model_name,
            quantizer_model_name=quantizer_name,
            vocab_size=vocab_size,
        )
        is not None
    )


densename_vocabsize = list(product(["hubert-base-ls960", "cpc-big-ll6k"], [50, 100, 200]))


@pytest.mark.parametrize("dense_name,vocab_size", densename_vocabsize)
def test_speech_encoder(dense_name, vocab_size):
    quantizer_name = "kmeans"

    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        need_f0=False,
        deduplicate=True,
        f0_normalizer=None,
        f0_quantizer=None,
    )

    assert encoder is not None

    # let's pass 0.5s of silence thru it
    waveform = torch.zeros(encoder.expected_sample_rate // 2)
    encoded = encoder(waveform)

    assert encoded


@pytest.mark.parametrize("dense_name,vocab_size", densename_vocabsize)
def test_vocoder_lookup(dense_name, vocab_size):
    quantizer_name = "kmeans"

    vocoder = TacotronVocoder.by_name(
        dense_model_name=dense_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
    )
    assert vocoder is not None
