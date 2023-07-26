# Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis

We introduce Expresso, a high-quality expressive speech dataset that includes both expressively rendered read speech (7 styles) and improvised dialogues rendered (26 styles). The dataset includes 4 speakers (2 males, 2 females), and totals 40 hours. The task of the Expresso Benchmark is to resynthesize the input audio using a low-bitrate discrete code that has been obtained without supervision from text.

## The Expresso dataset
Please go to the [dataset](dataset/) repository to have access to the Expresso dataset.

## The baseline resythesis model
We train unit-based [hifigan](https://arxiv.org/pdf/2010.05646.pdf) vocoders using speech units obtained from [HuBERT](https://arxiv.org/pdf/2106.07447.pdf) model as input. We condition the vocoder with one-hot speaker and style information of the utterance similar to [this work](https://arxiv.org/pdf/2104.00355.pdf).

### Pre-trained model
We share pre-trained hifigan vocoders using HuBERT units on the Expresso dataset, conditioning on one-hot speaker ans expression information.
| Link | | |

The resynthesis can be obtained from `textlesslib` as follows:
```python
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder


# Load speech encoder and vocoder
encoder = SpeechEncoder.by_name(
    dense_model_name = "hubert-base-ls960-layer-9",
    quantizer_model_name = "kmeans",
    vocab_size = 500,
    deduplicate = False, # False if the vocoder doesn't support duration prediction
).cuda()

vocoder = CodeHiFiGANVocoder.by_name(
    dense_model_name = "hubert-base-ls960-layer-9",
    quantizer_model_name = "kmeans",
    vocab_size = 500,
    vocoder_suffix = "expresso",
    speaker_meta = True,
    style_meta = True
).cuda()

# Load the audio
input_file = "path/to/audio.wav"
waveform, sample_rate = torchaudio.load(input_file)

# Convert it to (duplicated) units
encoded = encoder(waveform.cuda())
units = encoded["units"] # torch.Tensor([17, 17, 17, 17, 296, 296,...]

# Convert units back to audio
audio = vocoder(units) # torch.Tensor([-9.9573e-04, -1.7003e-04, -6.8756e-05,...]
```

### Train the hifigan model
You can use the [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis) library to train a speaker- and style-conditioned hifigan model, more information can be found in this [repo](https://github.com/facebookresearch/speech-resynthesis/tree/main/examples/expresso).


## Evaluation metrics
### WER
### Emotion Classification
### F0 Evaluation

## Citation
Please consider citing our work if you find it useful in your research:
```
```