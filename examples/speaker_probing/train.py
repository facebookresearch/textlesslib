# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import argparse
from textless.data.quantized_datasets import QuantizedLibriSpeech
from torch.utils.data import DataLoader
import torch.nn.functional as F
from probes import ContinuousClassifier, DiscreteClassifier, ConstantBaseline
from textless.data.speech_encoder import SpeechEncoder
from textless import dispatch_dense_model, dispatch_quantizer


def set_seed_(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def move_to(x, device: torch.device):
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict):
        return {k: move_to(v, device) for k, v in x.items()}
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense_model_name",
        type=str,
        help="Dense model to be used",
        default="hubert-base-ls960",
        choices=["hubert-base-ls960", "cpc-big-ll6k"],
    )
    parser.add_argument("--vocab_size", type=int, help="Unit vocab size", default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for K-means training", default=32
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--model_type",
        choices=["baseline", "discrete", "continuous"],
        default="baseline",
    )

    args = parser.parse_args()

    return args


def train(model, train_dataloader, valid_dataloader, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        train_epoch(model, train_dataloader, optimizer, epoch)
        evaluate_model(model, valid_dataloader)


def train_epoch(model, dataloader, optimizer, e):
    model.train()
    n_examples = 0.0
    accumulated = torch.zeros(1, dtype=torch.float64).cuda()

    for batch in dataloader:
        batch = move_to(batch, torch.cuda.current_device())
        speakers = torch.tensor(batch["rest"][2]).cuda()

        speaker_logprobs = model(batch)
        loss = F.nll_loss(speaker_logprobs, speakers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated += loss.detach().sum()
        n_examples += speakers.size(0)

    train_loss = (accumulated / n_examples).item()
    print(f"Epoch {e} | sliding mean train loss {train_loss}")


@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()
    n_examples = 0
    accumulated_loss = torch.zeros(1, dtype=torch.float64).cuda()
    accuracy = torch.zeros(1, dtype=torch.float64).cuda()

    for batch in dataloader:
        batch = move_to(batch, torch.cuda.current_device())
        speakers = torch.tensor(batch["rest"][2]).cuda()

        speaker_logprobs = model(batch)
        loss = F.nll_loss(speaker_logprobs, speakers)
        accumulated_loss += loss

        accuracy += (speaker_logprobs.argmax(dim=-1) == speakers).sum()
        n_examples += speakers.size(0)

    accumulated_loss /= n_examples
    accuracy /= n_examples

    print(f"Valid loss: {accumulated_loss.item()}, accuracy: {accuracy.item()}")


class SpeakerDatasetWrapper:
    def __init__(self, quantized_data, speaker_mapping=None):
        self.quantized_data = quantized_data
        self.speaker_mapping = (
            speaker_mapping
            if speaker_mapping is not None
            else self.get_speaker_ids(quantized_data.dataset._walker)
        )
        self.collater = self.quantized_data.collater
        self.max_length = (
            10 * 16_000 // self.quantized_data.speech_encoder.code_hop_size
        )

    @staticmethod
    def get_speaker_ids(walker):
        speaker_mapping = {}
        for fileid in walker:
            speaker_id, *_ = fileid.split("-")
            speaker_id = int(speaker_id)
            if speaker_id not in speaker_mapping:
                speaker_mapping[speaker_id] = len(speaker_mapping)
        return speaker_mapping

    def __getitem__(self, k):
        item = self.quantized_data[k]
        speaker = item["rest"][2]
        item["rest"][2] = self.speaker_mapping[speaker]

        if self.max_length < item["dense"].size(0):
            item["dense"] = item["dense"][: self.max_length, :]
            item["units"] = item["units"][: self.max_length]
            item["durations"] = item["durations"][: self.max_length]

        return item

    def __len__(self):
        return len(self.quantized_data)


def main():
    args = get_args()
    set_seed_(args.seed)

    dense_model_name = args.dense_model_name
    quantizer_model_name = "kmeans"
    vocab_size = args.vocab_size

    # NB: Hubert is not serializable as-is, so to have a multi-worker dataloader
    # we have a worker-around: load the actual checkpoint on the first call - which
    # will happen in a worker process already. This behavior is enabled with
    # the `lazy_load` flag.
    dense_model = dispatch_dense_model(dense_model_name, lazy_load=True)
    quantizer_model = dispatch_quantizer(
        dense_model_name, quantizer_model_name, vocab_size
    )

    speech_encoder = SpeechEncoder(
        dense_model,
        quantizer_model,
        deduplicate=False,
        need_f0=False,
        add_bos_eos=True,
    )

    dataset = QuantizedLibriSpeech(
        speech_encoder,
        root="datasets",
        url="dev-clean",
        download=True,
        device="auto"
        # when we set `device` to auto, the dataset instance will check if it is
        # running within a worker process of a dataloader. If it is the case,
        # it will move SpeechEncoder to one of the available GPUs, depending on the
        # worker id. This way we can pack quite a few (GPU-hungry) Hubert instances running across
        # all GPUs in parallel, within the same standard DataLoader.
    )

    speaker_mapping = SpeakerDatasetWrapper.get_speaker_ids(dataset.dataset._walker)
    max_speaker_id = max(speaker_mapping.values())
    dataset = SpeakerDatasetWrapper(dataset, speaker_mapping)

    valid_size = int(0.1 * len(dataset))
    train_size = len(dataset) - valid_size
    train_data, valid_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collater,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collater,
        num_workers=4,
    )

    if args.model_type == "baseline":
        model = ConstantBaseline(total_speakers=max_speaker_id + 1)
    elif args.model_type == "discrete":
        model = DiscreteClassifier(
            vocab_size=args.vocab_size + 3,  # accounting for bos, pad, eos
            embedding_size=32,
            n_heads=4,
            hidden_size=128,
            n_layers=2,
            dropout=0.1,
            pad_value=dataset.quantized_data.unit_pad,
            total_speakers=max_speaker_id + 1,
        )
    elif args.model_type == "continuous":
        input_size = {
            "hubert-base-ls960": 768,
            "cpc-big-ll6k": 512,
        }[dense_model_name]

        model = ContinuousClassifier(
            embedding_size=32,
            input_size=input_size,
            n_heads=4,
            hidden_size=128,
            n_layers=2,
            dropout=0.1,
            pad_value=dataset.quantized_data.unit_pad,
            total_speakers=max_speaker_id + 1,
        )
    else:
        assert False, "unknown model type"

    model.cuda()
    train(model, train_loader, valid_loader, args)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method

    set_start_method("spawn", force=True)
    main()
