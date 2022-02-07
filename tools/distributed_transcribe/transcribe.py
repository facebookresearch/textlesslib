# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.distributed as distr
import torch
import pathlib
from data_handler import ManifestDataset
from distributed import init_distributed_context

import logging

from textless.data.speech_encoder import SpeechEncoder

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_size",
        default=100,
        type=int,
        help="Quantization codebook vocabulary size",
    )
    parser.add_argument(
        "--dense_model", default="hubert-base-ls960", help="Dense model to be used"
    )
    parser.add_argument(
        "--quantizer_model", default="kmeans", help="Quantizer model to be used"
    )

    parser.add_argument(
        "--manifest", required=True, help="Path to the dataset manifest file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output files. Pseudo-units and duration (if requested) streams will be stored in files with .units and .durations suffixes, respectively",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="if set, consecutive repeats of the same pseudo-unit are collapsed ('1 2 2 2 3' becomes '1 2 3')",
    )
    parser.add_argument(
        "--durations",
        action="store_true",
        help="if set, the token durations stream is produced",
    )
    parser.add_argument(
        "--f0s",
        action="store_true",
        help="if set, the F0 stream is produced",
    )
    parser.add_argument(
        "--preserve_name",
        action="store_true",
        help="If set, the transcript contains names of the audio files",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Separator between pseudo-unit tokens",
    )

    parser.add_argument("--distributed_port", type=int, default=58554)

    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args


def worker_shard_path(fname, suffix, worker_id) -> pathlib.Path:
    return pathlib.Path(fname).with_suffix(f".{suffix}_partial_{worker_id}")


def transcribe(args, rank, world_size):
    dataset = ManifestDataset(args.manifest)

    speech_encoder = SpeechEncoder.by_name(
        dense_model_name=args.dense_model,
        quantizer_model_name=args.quantizer_model,
        vocab_size=args.vocab_size,
        deduplicate=args.deduplicate,
        need_f0=args.f0s,
    ).cuda()

    output_files = {
        "units": open(worker_shard_path(args.output, "units", rank), "w"),
        "durations": None
        if not args.durations
        else open(worker_shard_path(args.output, "durations", rank), "w"),
        "f0s": None
        if not args.f0s
        else open(worker_shard_path(args.output, "f0s", rank), "w"),
    }

    # DistributedSampler will pad the dataloader to be divisible
    # by the number of workers, which we do not want so we iterate directly
    for i in range(rank, len(dataset), world_size):
        waveform, name = dataset[i]
        encoded = speech_encoder(waveform)

        stream_names = ["units", "durations"]
        if args.f0s:
            stream_names += ["f0s"]

        for stream_name in stream_names:
            stream = encoded[stream_name]
            stream = [str(int(x)) for x in stream.tolist()]
            stream = args.separator.join(stream)

            stream = f"{name}\t{stream}" if args.preserve_name else stream
            print(stream, file=output_files[stream_name])

    for fout in output_files.values():
        if fout:
            fout.close()


def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        transcribe(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()

    if context.is_leader:
        generated_streams = ["units"]
        if args.durations:
            generated_streams += ["durations"]
        if args.f0s:
            generated_streams += ["f0s"]

        for stream_name in generated_streams:
            merge_files(args.output, stream_name, context.world_size)


def merge_files(full_output, suffix, n_workers):
    output = full_output + f".{suffix}"
    with open(output, "w") as full:
        for worker_id in range(n_workers):
            partial_path = worker_shard_path(full_output, suffix, worker_id)
            partial = open(partial_path, "r")
            for line in partial:
                print(line.strip(), file=full)
            partial_path.unlink()


if __name__ == "__main__":
    args = get_args()
    main(args)
