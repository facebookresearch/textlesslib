# The Expresso Dataset
[[paper]](https://arxiv.org/abs/2308.05725) [[demo samples]](https://speechbot.github.io/expresso/)

## Introduction
The Expresso dataset is a high-quality (48kHz) expressive speech dataset that includes both expressively rendered read speech (8 styles, in mono wav format) and improvised dialogues (26 styles, in stereo wav format). The dataset includes 4 speakers (2 males, 2 females), and totals 40 hours (11h read, 30h improvised). The transcriptions of the read speech are also provided.

You can listen to samples from the Expresso Dataset at [this website](https://speechbot.github.io/expresso/).

## Data Statistics
Here are the statistics of Expresso’s expressive styles:

----------------------------------------------------------------
 Style            | Read (min) | Improvised (min) | total (hrs)
------------------|------------|------------------|-------------
 angry            | -          | 82               | 1.4
 animal           | -          | 27               | 0.4
 animal_directed  | -          | 32               | 0.5
 awe              | -          | 92               | 1.5
 bored            | -          | 92               | 1.5
 calm             | -          | 93               | 1.6
 child            | -          | 28               | 0.4
 child_directed   | -          | 38               | 0.6
 confused         | 94         | 66               | 2.7
 default          | 133        | 158              | 4.9
 desire           | -          | 92               | 1.5
 disgusted        | -          | 118              | 2.0
 enunciated       | 116        | 62               | 3.0
 fast             | -          | 98               | 1.6
 fearful          | -          | 98               | 1.6
 happy            | 74         | 92               | 2.8
 laughing         | 94         | 103              | 3.3
 narration        | 21         | 76               | 1.6
 non_verbal       | -          | 32               | 0.5
 projected        | -          | 94               | 1.6
 sad              | 81         | 101              | 3.0
 sarcastic        | -          | 106              | 1.8
 singing*         | -          | 4                | .07
 sleepy           | -          | 93               | 1.5
 sympathetic      | -          | 100              | 1.7
 whisper          | 79         | 86               | 2.8
 **Total**        | **11.5h**  | **34.4h**        | **45.9h**
----------------------------------------------------------------
*singing is the only improvised style that is not in dialogue format.

## Audio Quality
The audio was recorded in a professional recording studio with minimal background noise at 48kHz/24bit. The files for read speech and singing are in a mono wav format; and for the dialog section in stereo (one channel per actor), where the original flow of turn-taking is preserved.

## Downloading the Expresso dataset
### Downloading
The Expresso dataset can be downloaded from the following link:
* [expresso.tar (36GB)](https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar)
    * md5: `6bf580a4cd4392ae2473626147b5307c`
### Directory Structure

The expresso dataset directory has the following structure:
```
expresso/
├───README.txt
├───LICENSE.txt
├───read_transcriptions.txt
├───VAD_segments.txt
├───splits/
│   ├───train.txt
│   ├───dev.txt
│   ├───test.txt
│   └───README
└───audio_48khz/
    ├───conversational/
    │   ├───ex04-ex01/ # speaker pair in {channel1_spk}-{channel2_spk} format
    │   │   ├───animal-animaldir/ # style pair in {channel1_style}-{channel2_style} format
    │   │   │   ├───ex04-ex01_animal-animaldir_005.wav
    │   │   │   ├───ex04-ex01_animal-animaldir_006.wav
    │   │   │   └───...
    │   │   ├───laughing/ # both channels have the same style
    │   │   │   ├───ex04-ex01_laughing_001.wav
    │   │   │   ├───ex04-ex01_laughing_002.wav
    │   │   │   └───...
    │   │   └───...
    │   └───...
    └───read/
        ├───ex03/ # speaker
        │   ├───default/ # style
        │   │   ├───longform/ # recorded in long format
        │   │   │   └───ex03_default_longform_00001.wav
        │   │   └───base/ # recorded in short sentences
        │   │       ├───ex03_default_00003.wav
        │   │       ├───ex03_default_emphasis_00010.wav
        │   │       ├───ex03_default_essentials_00005.wav
        │   │       └───...
        │   ├───happy/
        │   │   └───base/
        │   │       ├───ex04_happy_00085.wav
        │   │       ├───ex04_happy_00091.wav
        │   │       └───...
        │   └───...
        └───...
```

### Conversational Speech
The `conversational` audio directory contains the improvised dialogues (except singing) and is organized with the following structure:

`conversational/{speaker pair}/{styles}/{speaker pair}_{styles}_{id}.wav`

where

- `{speaker pair}`: hyphen-separated pair of speakers in two channels, with the left speaker on first/left channel, and the right speaker on second/right channel. E.g. `ex03-ex01` means `ex03` on channel 1 and `ex01` on channel 2

- `{styles}`: styles of the two channels of the dialogue. If `{styles}` **contains a hyphen**, the scene contains **two styles**. The style on the left corresponds to the channel1/left speaker’s style, and the style on the right corresponds to the channel2/right speaker’s style. If the is **no hyphen**, both speakers are speaking in the same style

- `{id}`: ID of the audio with the given speaker pair and style

### Read Speech
The `read` audio directory contains all the read speech and singing style, with the following structure:

`read/{speaker}/{style}/{corpus}/{speaker}_{style or substyle}_{id}.wav`

where

- `{speaker}`: speaker name

- `{style}`: expressive style of the audio

- `{corpus}`: 'base' or 'longform'

- `{style or substyle}`: same as `{style}`, with the exception of longform speech and default substyles. The exception substyles are: 'narration_longform', 'default_longform', 'default_emphasis', 'default_essentials'

- `{id}`: ID of the audio with the given speaker and style/substyle

## Data preparation for speech resynthesis
We prepare the Expresso dataset for speech synthesis experiments by cutting down to short segments of less than 15 seconds based on VAD information. The processing scripts can be found in the [processing](processing) directory.

## License
The Expresso dataset is distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Reference
For more information, see the paper: [EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis](https://arxiv.org/abs/2308.05725), Tu Anh Nguyen*, Wei-Ning Hsu*, Antony D'Avirro*, Bowen Shi*, Itai Gat, Maryam Fazel-Zarani, Tal Remez, Jade Copet, Gabriel Synnaeve, Michael Hassid, Felix Kreuk, Yossi Adi⁺, Emmanuel Dupoux⁺, INTERSPEECH 2023.