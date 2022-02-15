# textlesslib

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Textless NLP is an active area of research that aims to extend NLP techniques to work directly on spoken language. By using self-supervisedly
learnt discrete speech representations, the area promises to unlock interesting NLP applications on languages without written form or on facets of spoken 
language that are unaccessable for text-based approaches, e.g. prosody. To learn more, please check some of the [papers](https://speechbot.github.io/).

**textlesslib** is a library aimed to facilitate research in Textless NLP. The goal of the library is to speed up the research cycle and
lower the learning curve for those who want to start. We provide highly configurable, off-the-shelf available tools to encode speech
as sequences of discrete values and tools to decode such streams back into the audio domain.


Table of Contents
=================

   * [Installation](#installation)
   * [Usage examples](#usage-examples)
      * [Encoding speech](#encoding-speech)
      * [Dataset helpers](#dataset-helpers)
      * [Data preprocessing](#data-preprocessing)
   * [Testing](#testing)


## Installation
```bash
git clone git@github.com:facebookresearch/textlesslib.git
cd textlesslib
pip install -e .
pip install git+git://github.com:pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8
```

## Usage examples
We include a set of examples in the [examples](./examples) folder:
*  [Discrete speech resynthesis (& compression)](./examples/resynthesis/) 
*  [Probing for speaker information in the representations](./examples/speaker_probing/)
*  [Generative Spoken Language Modeling (aka Speech Continuation)](./examples/gslm/)

There is also a [[Jupyter notebook]](./examples/resynthesis_and_continuation.ipynb) and a [[Google Colab]](https://colab.research.google.com/github/facebookresearch/textlesslib/blob/main/examples/resynthesis_and_continuation.ipynb) that combine discrete resynthesis and speech continuation examples in a step-by-step mini-tutorial.

We believe those examples can serve both as illustrations for the provided components and provide 
a starting point for tinkering in interesting directions.

### Encoding speech
Below is an example on loading an audio example and encoding it as a sequence of HuBERT-based discrete tokens (aka pseudo-units).
Downloading of the required checkpoints is handled by textlesslib itself (by default they are stored in `~/.textless`):

```python
import torchaudio
from textless.data.speech_encoder import SpeechEncoder

dense_model_name = "hubert-base-ls960"
quantizer_name, vocab_size = "kmeans", 100
input_file = "input.wav"

# now let's load an audio example
waveform, sample_rate = torchaudio.load(input_file)

# We can build a speech encoder module using names of pre-trained
# dense and quantizer models.  The call below will download
# appropriate checkpoints as needed behind the scenes. We can
# also construct an encoder by directly passing model instances
encoder = SpeechEncoder.by_name(
    dense_model_name=dense_model_name,
    quantizer_model_name=quantizer_name,
    vocab_size=vocab_size,
    deduplicate=True,
).cuda()


# now convert it in a stream of deduplicated units (as in GSLM)
encoded = encoder(waveform.cuda())
# encoded is a dict with keys ('dense', 'units', 'durations').
# It can also contain 'f0' if SpeechEncoder was initialized
# with need_f0=True flag.
units = encoded["units"]  # tensor([71, 12, 57, ...], ...)
```
Now it can be casted back into the audio domain:

```python
# as with encoder, we can setup vocoder by passing checkpoints
# directly or by specifying the expected format by the names
# of dense and quantizer models (these models themselves
# won't be loaded)
vocoder = TacotronVocoder.by_name(
    dense_model_name,
    quantizer_name,
    vocab_size,
).cuda()

# now we turn those units back into the audio.
audio = vocoder(units)

# save the audio
torchaudio.save(output_file, audio.cpu().float().unsqueeze(0), vocoder.output_sample_rate)
```
### Dataset helpers
Below is an example on using `textless` view on the LibriSpeech dataset:
```python
encoder = SpeechEncoder.by_name(
  dense_model_name=dense_model_name,
  quantizer_model_name=quantizer_name,
  vocab_size=vocab_size,
  deduplicate=True,
).cuda()

quantized_dataset = QuantizedLibriSpeech(
  root=existing_root, speech_encoder=encoder, url=url)

datum = quantized_dataset[0]
sample_rate, utterance, speaker_id, chapter_id, utterance_id = datum['rest']
# datum['units'] = tensor([71, 12, 63, ...])
```
In the [probing example](./examples/speaker_probing/) we illustrate how such a dataset
can be used with a standard Pytorch dataloader in a scalable manner.

### Data preprocessing
We also provide a [multi-GPU/multi-node preprocessing tool](tools/distributed_transcribe/)
for the cases where on-the-fly processing of audio should be avoided.

## Testing
We use pytest (`pip install pytest pytest-xdist `). Unit tests are located in the `test` directory:
```bash
cd tests && pytest -n 8
```

## Licence
textlesslib is licensed under MIT, the text of the license can be found [here](LICENSE).
Internally, it uses 
* [WaveGlow](https://github.com/NVIDIA/waveglow) - licensed under BSD-3-Clause license;
* [tacotron implementation](https://github.com/keithito/tacotron) - licensed under MIT license;
* [tacotron2 implementation](https://github.com/NVIDIA/tacotron2) - licensed under BSD-3-Clause license;
* [STFT implementation](https://github.com/pseeth/torch-stft) - licensed under BSD-3-Clause license.
