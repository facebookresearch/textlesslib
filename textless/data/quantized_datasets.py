# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A `textless` wrapper around datasets (Quantize dataset) and a set of pre-wrapped standard torchaudio datasets
(https://pytorch.org/audio/stable/datasets.html).
"""

from typing import Callable, Union, Optional, Any, Dict, List
import pathlib
import torch
import logging


from torchaudio.datasets import LIBRISPEECH, LJSPEECH, COMMONVOICE, VCTK_092, YESNO
from torch.utils.data import Dataset

from textless.data.speech_encoder import SpeechEncoder
from textless.data.collater_utils import collate_tensors

logger = logging.getLogger(__name__)


def QuantizedLibriSpeech(
    speech_encoder: Callable,
    root: Union[str, pathlib.Path],
    url: str = "train-clean-100",
    folder_in_archive: str = "LibriSpeech",
    download: bool = False,
    device: Optional[str] = None,
):
    dataset = LIBRISPEECH(root, url, folder_in_archive, download)
    return QuantizeDataset(
        dataset, speech_encoder, device, speaker_extractor=default_speaker_ls
    )


def default_speaker_ls(rest):
    return str(rest[2])


def QuantizedLjSpeech(
    speech_encoder: Callable,
    root: Union[str, pathlib.Path],
    url: str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    folder_in_archive: str = "wavs",
    download: bool = False,
    device: Optional[str] = None,
):
    dataset = LJSPEECH(root, url, folder_in_archive, download)
    return QuantizeDataset(dataset, speech_encoder, device)


# NB: no direct download, see https://github.com/pytorch/audio/pull/1018
def QuantizedCommonVoice(
    speech_encoder: Callable,
    root: Union[str, pathlib.Path],
    tsv: Optional[str] = "train.tsv",
    device: Optional[str] = None,
):
    dataset = COMMONVOICE(root, tsv)
    return QuantizeDataset(
        dataset, speech_encoder, device, speaker_extractor=default_speaker_commonvoice
    )


def default_speaker_commonvoice(rest):
    return rest["client_id"]


def QuantizedVCTK_092(
    speech_encoder: Callable,
    root: Union[str, pathlib.Path],
    mic_id: Optional[str] = "mic2",
    download: bool = False,
    url: Optional[
        str
    ] = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
    audio_ext=".flac",
    device: Optional[str] = None,
):
    dataset = VCTK_092(root, mic_id, download, url, audio_ext)
    return QuantizeDataset(
        dataset, speech_encoder, device, speaker_extractor=default_speaker_vctk
    )


def default_speaker_vctk(rest):
    return rest[2]


def QuantizedYesNo(
    speech_encoder: Callable,
    root: Union[str, pathlib.Path],
    url: str = "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
    folder_in_archive: str = "waves_yesno",
    download: bool = False,
    device: Optional[str] = None,
):
    dataset = YESNO(root, url, folder_in_archive, download)
    return QuantizeDataset(
        dataset, speech_encoder, device, speaker_extractor=no_speaker
    )


def no_speaker(_):
    """
    The dataset doesn't provide speaker information; please
    use speaker-independent F0 normalization.
    """

    return None


class QuantizeDataset:
    def __init__(
        self,
        dataset: Dataset,
        speech_encoder: Callable,
        device: Optional[Union[str, torch.torch.device]] = None,
        speaker_extractor: Optional[Callable] = None,
    ):
        """Wraps an arbitrary index-style PyTorch dataset and provides a "textless" quantized view of it.
        It makes the following assumptions about `dataset[i]`:
        * it returns a tuple where the first element is a raw waveform,
        * all waveforms have the same sample rate as speech_encoder expects.

        Args:
            dataset (Dataset): Dataset to be wrapped
            speech_encoder (Callable): SpeechEncoder to encode the audio
            device (Optional[Union[str, torch.torch.device]], optional):
                Sets a device to be used for encoding speech. If is set to None, torch.cuda.current_device() will be used.
                When used in a dataloader with multiple workers, it might be useful to set `device` to "auto".
                This way, on the first call of __getitem__(), the QuantizeDataset checks if it runs in a dataloader
                worker. If this is the case, it will grab one of the available GPUs and place its copy of
                SpeechEncoder there. This could be useful as SpeechEncoder is typically GPU-intensive and it is a good idea to parallelize
                across multiple GPUs. Defaults to None.
            speaker_extractor (Optional[Callable], optional): An optional callable that extracts speaker id from the
            dataset[i] output. Only needed if speech_encoder uses per-speaker F0 normalization. Defaults to None.
        """

        self.dataset = dataset
        self.speech_encoder = speech_encoder

        self.randomize_device_on_next_call = False
        # TODO: allow specifying a list of allowed devices
        # this way we can preprocess and train a model on different GPUs.
        if device == "auto":
            self.device = None
            self.randomize_device_on_next_call = True
        elif device is None:
            self.device = torch.cuda.current_device()
            self.speech_encoder.to(self.device)
        else:
            self.device = torch.device(device)
            self.speech_encoder.to(self.device)

        self.speaker_extractor = (
            speaker_extractor if speaker_extractor is not None else default_speaker_ls
        )

        self.unit_vocab_size = torch.tensor([self.speech_encoder.vocab_size])

        self.unit_pad = 1 + max(
            self.unit_vocab_size - 1,
            self.speech_encoder.bos.item(),
            self.speech_encoder.eos.item(),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def select_worker_gpu(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # not a worker process
            self.device = torch.cuda.current_device()
            self.speech_encoder.to(self.device)
        else:
            # we are in a worker process, we'll assign ourselves
            # to a random GPU among available
            worker_id = worker_info.id
            device_id = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            self.device = torch.cuda.current_device()
            self.speech_encoder.cuda()

    def __getitem__(self, k: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """Returns "texless" representation of the k-th element of the wrapped dataset in
        a form of aligned unit, durarion, and (optionally) F0 streams.
        Args:
            k (int): example index.
        Returns:
            Dict[str, Union[torch.Tensor, Any]]: A dict that has contains
            speech_encoder(waveform) outputs and the remainder of what
            the underlying dataset returned under the "rest" key.
            All tensors are placed on CPU.
        """
        waveform, *rest = self.dataset[k]

        if self.randomize_device_on_next_call and self.device is None:
            self.select_worker_gpu()
            self.randomize_device_on_next_call = False

        speaker = self.speaker_extractor(rest)
        encoded = self.speech_encoder(waveform.to(self.device), speaker)
        # dataset
        for k, v in encoded.items():
            if isinstance(v, torch.Tensor):
                encoded[k] = v.cpu()
        encoded["rest"] = rest

        return encoded

    def collater(self, samples: List[Any]):
        """Collater utility for using QuantizeDataset within a DataLoader.
        Args:
            samples (_type_): Samples to collate.

        Returns:
            Dict[str, Union[Any, torch.Tensor]: Per-stream collated samples.
            The unit stream is padded with QuantizeDataset.unit_pad while other streams are padded
            with zeros. The non-audio parts of the samples (i.e. those that are in
            sample["rest"]) are not collated and are returned as-is under the "rest" key.
        """
        units = collate_tensors([s["units"] for s in samples], pad=self.unit_pad)
        if "f0" in samples[0]:
            f0 = collate_tensors(
                [s["f0"] for s in samples], pad=torch.zeros_like(samples[0]["f0"][0])
            )
        else:
            f0 = None
        durations = collate_tensors(
            [s["durations"] for s in samples],
            pad=torch.zeros_like(samples[0]["durations"][0]),
        )

        bsz = len(samples)
        dense_dim = samples[0]["dense"].size(1)
        max_len = max(s["dense"].size(0) for s in samples)

        dense = torch.zeros((bsz, max_len, dense_dim))
        for i, s in enumerate(samples):
            l = s["dense"].size(0)
            dense[i, :l, :] = s["dense"]

        n_rest = len(samples[0]["rest"])
        _slice = lambda i: [s["rest"][i] for s in samples]

        rest = [_slice(i) for i in range(n_rest)]

        result = {"units": units, "durations": durations, "dense": dense, "rest": rest}
        if f0 is not None:
            result["f0"] = (f0,)
        return result
