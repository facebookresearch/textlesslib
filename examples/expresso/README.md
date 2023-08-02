# Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis

We introduce Expresso, a high-quality expressive speech dataset that includes both expressively rendered read speech (7 styles) and improvised dialogues rendered (26 styles). The dataset includes 4 speakers (2 males, 2 females), and totals 40 hours. The task of the Expresso Benchmark is to resynthesize the input audio using a low-bitrate discrete code that has been obtained without supervision from text.

## 1. The Expresso dataset
Please go to the [dataset](dataset/) repository to have access to the Expresso dataset.

## 2. The baseline resythesis model
We train unit-based [hifigan](https://arxiv.org/pdf/2010.05646.pdf) vocoders using speech units obtained from [HuBERT](https://arxiv.org/pdf/2106.07447.pdf) model as input. We condition the vocoder with one-hot speaker and style information of the utterance similar to [this work](https://arxiv.org/pdf/2104.00355.pdf).

### 2.1 Pre-trained model
We share pre-trained hifigan vocoders using HuBERT units on the Expresso dataset, conditioning on one-hot speaker ans expression information.
|HuBERT model|Quantizer|Vocoder Data|Vocoder Model|
|---|---|---|---|
|HuBERT base LS960 (Layer9)|Km500 (LS960)|Expresso|[download]()|
|HuBERT base LS960 (Layer9)|Km2000 (Expresso)|Expresso|[download]()|
|HuBERT Mix1 (Layer12)|Km2000 (LS960)|Expresso|[download]()|
|HuBERT Mix1 (Layer12)|Km2000 (Expresso)|Expresso|[download]()|

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
units = encoded["units"] # torch.Tensor([17, 17, 17, 17, 296, 296,...])

# Convert units back to audio
audio = vocoder(units) # torch.Tensor([-9.9573e-04, -1.7003e-04, -6.8756e-05,...])
```
Please note that you'll need a reasonably recent version of fairseq (i.e. [fairseq/tree/100cd91db1](https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189)) in order to load HuBERT Mix1 checkpoint.

### 2.2 Train the hifigan model
You can use the [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis) library to train a speaker- and style-conditioned hifigan model, more information can be found in this [repo](https://github.com/facebookresearch/speech-resynthesis/tree/main/examples/expresso).


## 3. Evaluation metrics
### 3.1 WER
We transcibre the synthesized speech using a pre-trained Automatic Speech Recognition (ASR) model ([wav2vec2]((https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)) pre-trained on Librilight60kh fine-tuned on Librispeech960h, get `wav2vec_vox_960h_pl.pt` checkpoint [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)) and compute the Word Error Rate (WER) between the transcribed text and the true transcription.

#### 3.1.1 Prepare ASR dataset
You'll need to preprare a manifest file along with a transcription to perform the ASR.

The manifest file can be obtain with wav2vec's manifest script [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py), and the transcription is expected to be in letter format.

Here is an example of the manifest file `expresso.tsv`
```
/root/to/synthesized/expresso/audio
ex04_default_00340_gen.wav	30381
ex04_default_00341_gen.wav	29970
...
```
and the transciption file `expresso.ltr`
```
C A N | W E | G O | T H E R E | P L E A S E
C O M P U T E | F O U R | S I X T E E N
...
```

You'll need to download wav2vec dictionary `dict.ltr.txt` and put it into data directory
```bash
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P $DATA_DIR
```

The data directory now should contain the following files
```bash
ls $DATA_DIR
> dict.ltr.txt  expresso.ltr  expresso.tsv
```

#### 3.1.2 Run ASR and obtain WER
The ASR is performed using [fairseq's speech recognition example](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_recognition). Here is the example command:
```bash
SUBSET_NAME="expresso"

cd $FAIRSEQ_ROOT
python examples/speech_recognition/infer.py \
    ${DATA_DIR} --task audio_finetuning \
    --nbest 1 \
    --path path/to/wav2vec_vox_960h_pl.pt \
    --w2l-decoder viterbi \
    --criterion ctc \
    --labels ltr \
    --max-tokens 4000000 \
    --post-process letter \
    --gen-subset ${SUBSET_NAME} \
    --results-path ${OUTPUT_DIR} \
```
Please note that you'll need a reasonably recent version of fairseq (i.e. [fairseq/tree/100cd91db1](https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189)) in order to run the ASR script. You'll also need to install flashlight python bindings, follow the instructions on [this page](https://github.com/flashlight/flashlight/tree/e16682fa32df30cbf675c8fe010f929c61e3b833/bindings/python) to install. Flashlight v0.3.2 must be used to install the bindings:
```
git clone --branch v0.3.2 https://github.com/flashlight/flashlight
```

### 3.2 Emotion Classification
We fine-tune the wav2vec2-base model on audio classification task using [transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) library. We use the expressive style in the Expresso dataset as the labels.

#### 3.2.1 Pre-trained model checkpoint
You can download model checkpoint and config here
|Model Checkpoint|Config|Preprocessor Config|
|---|---|---|
|[pytorch_model.bin]()|[config.json]()|[preprocessor_config.json]()|

#### 3.2.2 Run emotion classification
Use the `classify_audio.py` script to perform prediction and possibly compute the accuracy if the label file is given. Here is an example command
```bash
MANIFEST_FILE=$DATA_DIR/expresso.tsv
LABEL_FILE=$DATA_DIR/expresso.labels # each line contains the true label of the corresponding audio in the manifest file
PREDICTION_FILE=$DATA_DIR/expresso.predictions

python classify_audio.py \
    --model_ckpt $CKPT_DIR \
    --from_tsv $MANIFEST_FILE \
    --label_file $LABEL_FILE \
    --output_file $PREDICTION_FILE
```

### 3.3 F0 Evaluation
We'll use the [F0 evaluation script](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_synthesis/evaluation/eval_f0.py) from [FAIRSEQ S^2](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_synthesis).

#### 3.3.1 Prepare F0 evaluation dataset
For F0 evals, you need to prepare a `generated_data.tsv` file, containing `"ref"` and `"syn"` fields with the ground truth audio and generated audio as follows
```
"ref"   "syn"
/path/to/ex04_default_00340_gt.wav /path/to/ex04_default_00340_gen.wav
/path/to/ex04_default_00341_gt.wav /path/to/ex04_default_00341_gen.wav
...
```

#### 3.3.2 Run F0 evaluation
Then run the following command
```bash
cd $FAIRSEQ_ROOT
python examples/speech_synthesis/evaluation/eval_f0.py \
    generated_data.tsv --ffe
```


## Citation
Please consider citing our work if you find it useful in your research:
```
```