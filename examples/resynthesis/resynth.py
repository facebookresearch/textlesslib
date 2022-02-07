# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchaudio
from textless import dispatch_dense_model, dispatch_quantizer
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense_model_name",
        type=str,
        default="hubert-base-ls960",
        choices=["hubert-base-ls960", "cpc-big-ll6k"],
        help="Dense representation model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50,
        help="Vocabulary size used for resynthesis",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output audio file.",
    )
    parser.add_argument(
        "--decoder_steps",
        type=int,
        default=100,
        help="Maximal number of decoder steps",
    )

    args = parser.parse_args()
    return args


def get_compression_rate(dense_model, units, wave, vocab_size, sample_rate):
    import numpy as np

    assert units.ndim == 1
    assert wave.ndim == 1

    time_in_seconds = wave.numel() / sample_rate

    uniform_token_entropy = np.log2(vocab_size)
    # calculated on LL-6k train
    unigram_token_entropy = {
        "hubert-base-ls960": {
            50: 5.458528917634601,
            100: 6.44513268276806,
            200: 7.477069233162813,
        },
        "cpc-big-ll6k": {
            50: 5.428271158461133,
            100: 6.413083187885448,
            200: 7.44253841579776,
        },
    }[dense_model][vocab_size]

    uniform_bps = uniform_token_entropy * units.size(0) / time_in_seconds
    unigram_entropy = unigram_token_entropy * units.size(0) / time_in_seconds

    return uniform_bps, unigram_entropy


def main(args):
    dense_model_name = args.dense_model_name
    quantizer_name = "kmeans"

    # We can build a speech encoder module using names of pre-trained dense and quantizer models.
    # The call below will download appropriate checkpoints as needed behind the scenes
    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=args.vocab_size,
        need_f0=False,
        deduplicate=True,
        f0_normalizer=None,
        f0_quantizer=None,
    ).cuda()

    # Alternatively, we can pass dense/quantizer models directly.
    # Here, we'll look up the same models as above, but generally those
    # could be any other models.
    dense_model = dispatch_dense_model(dense_model_name)
    quantizer_model = dispatch_quantizer(
        dense_model_name, quantizer_name, args.vocab_size
    )

    # .. and use them when initializing the encoder. Same constructor can be used to when we want
    # to use models other than pre-defined.
    encoder = SpeechEncoder(
        dense_model=dense_model,
        quantizer_model=quantizer_model,
        need_f0=False,
        deduplicate=True,
        f0_normalizer=None,
        f0_quantizer=None,
    ).cuda()

    # now let's load an audio example
    waveform, input_sample_rate = torchaudio.load(args.input)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = encoder.maybe_resample(waveform, input_sample_rate)

    # now and convert it in a stream of deduplicated units (as in GSLM)
    encoded = encoder(waveform.cuda())
    # encoded is a dict with keys ('dense', 'units', 'durations'). It can also contain 'f0' if SpeechEncoder
    # was initialized with need_f0=True flag.
    units = encoded[
        "units"
    ]  # tensor([71, 12, 57, 12, 57, 12, 57, 12, ...], device='cuda:0', dtype=torch.int32)

    # as with encoder, we can setup vocoder by specifying names of pretrained models
    # or by passing checkpoint paths directly. The dense/quantizer models are not invokes,
    # we just use their names as an index.
    vocoder = TacotronVocoder.by_name(
        dense_model_name,
        quantizer_name,
        args.vocab_size,
    ).cuda()

    # now we turn those units back into the audio.
    audio = vocoder(units)

    # save the audio
    torchaudio.save(
        args.output, audio.cpu().float().unsqueeze(0), vocoder.output_sample_rate
    )

    uniform_bps, learned_bps = get_compression_rate(
        dense_model_name, units, waveform, args.vocab_size, encoder.expected_sample_rate
    )

    print(
        f"Audio of length {round(waveform.size(0) / 16_000, 1)} seconds represented as {units.numel()} tokens"
    )
    print(
        f"\tAssuming uniform token distribution: {round(uniform_bps, 1)} bits per second"
    )
    print(
        f"\tAssuming unigram token distribution estimated on LL-6K train: {round(learned_bps, 1)} bits per second"
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
