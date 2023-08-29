# Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis

[[paper]](https://arxiv.org/abs/2308.05725) [[demo samples]](https://speechbot.github.io/expresso/) [[dataset]](dataset/)

We introduce Expresso, a high-quality (48kHz) expressive speech dataset that includes both expressively rendered read speech (8 styles, in mono wav format) and improvised dialogues (26 styles, in stereo wav format). The dataset includes 4 speakers (2 males, 2 females), and totals 40 hours (11h read, 30h improvised). The transcriptions of the read speech are also provided. The task of the Expresso Benchmark is to resynthesize the input audio using a low-bitrate discrete code that has been obtained without supervision from text.

## 1. The Expresso dataset
Please go to the [dataset](dataset/) repository to have access to the Expresso dataset.

## 2. The baseline resythesis model
We train unit-based [hifigan](https://arxiv.org/pdf/2010.05646.pdf) vocoders using speech units obtained from [HuBERT](https://arxiv.org/pdf/2106.07447.pdf) model as input. We condition the vocoder with one-hot speaker and style information of the utterance similar to [this work](https://arxiv.org/pdf/2104.00355.pdf).

### 2.1 Pre-trained model
We share pre-trained hifigan vocoders using HuBERT units on Expresso, LJ and VCTK datasets, conditioning on one-hot speaker and expression information (*you don't need to manually download the checkpoints if using `textlesslib` as shown below*).
|HuBERT model|Quantizer|Vocoder Data|HifiGAN Vocoder Model|
|---|---|---|---|
|[HuBERT base LS960](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)|[L9 km500 (LS960)](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin)|Expresso + LJ + VCTK|[generator.pt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/generator.pt) - [config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/config.json) - [speakers.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/speakers.txt) - [styles.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/styles.txt)|
|[HuBERT base LS960](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)|[L9 km2000 (Expresso)](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hubert_base_ls960_L9_km2000_expresso.bin)|Expresso + LJ + VCTK|[generator.pt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/generator.pt) - [config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/config.json) - [speakers.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/speakers.txt) - [styles.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/styles.txt)|
|[HuBERT Mix1 (VP, MLS, CV)](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3.pt)|[L12 km2000 (Mix1)](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3_L12_km2000.bin)|Expresso + LJ + VCTK|[generator.pt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/generator.pt) - [config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/config.json) - [speakers.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/speakers.txt) - [styles.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/styles.txt)|
|[HuBERT Mix1 (VP, MLS, CV)](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3.pt)|[L12 km2000 (Expresso)](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso.bin)|Expresso + LJ + VCTK|[generator.pt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/generator.pt) - [config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/config.json) - [speakers.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/speakers.txt) - [styles.txt](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/styles.txt)|

The resynthesis can be obtained from `textlesslib` as follows:

Please note that you'll need a reasonably recent version of fairseq (i.e. [fairseq/tree/4db264940f](https://github.com/facebookresearch/fairseq/tree/4db264940f281a6f47558d17387b1455d4abd8d9)) in order to load HuBERT Mix1 checkpoint.

```python
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder

# Available models
EXPRESSO_MODELS = [
    ("hubert-base-ls960-layer-9", "kmeans", 500),
    ("hubert-base-ls960-layer-9", "kmeans-expresso", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans-expresso", 2000),
]

# Try one model
dense_model, quantizer_model, vocab = EXPRESSO_MODELS[3]

# Load speech encoder and vocoder
encoder = SpeechEncoder.by_name(
    dense_model_name = dense_model,
    quantizer_model_name = quantizer_model,
    vocab_size = vocab,
    deduplicate = False, # False if the vocoder doesn't support duration prediction
).cuda()

vocoder = CodeHiFiGANVocoder.by_name(
    dense_model_name = dense_model,
    quantizer_model_name = quantizer_model,
    vocab_size = vocab,
    speaker_meta = True,
    style_meta = True
).cuda()
speakers = vocoder.speakers # ['ex01', 'ex02', 'ex03', 'ex04', 'lj', 'vctk_p225', ...]
styles = vocoder.styles # ['read-default', 'read-happy', 'read-sad', 'read-whisper', ...]

# Load the audio
input_file = "path/to/audio.wav"
waveform, sample_rate = torchaudio.load(input_file)

# Convert it to (duplicated) units
encoded = encoder(waveform.cuda())
units = encoded["units"] # torch.Tensor([17, 17, 17, 17, 296, 296,...])

# Convert units back to audio
audio = vocoder(
    units,
    speaker_id=speakers.index('ex01'),
    style_id=styles.index('read-default'),
) # torch.Tensor([-9.9573e-04, -1.7003e-04, -6.8756e-05,...])
```

### 2.2 Train the hifigan model
You can use the [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis) library to train a speaker- and style-conditioned hifigan model, more information can be found in this [repo](https://github.com/facebookresearch/speech-resynthesis/tree/main/examples/expresso).


## 3. Evaluation metrics
### 3.1 WER
We transcibre the synthesized speech using a pre-trained Automatic Speech Recognition (ASR) model ([wav2vec2]((https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)) pre-trained on Librilight60kh fine-tuned on Librispeech960h, get `wav2vec_vox_960h_pl.pt` checkpoint [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)) and compute the Word Error Rate (WER) between the transcribed text and the true transcription.

#### 3.1.1 Prepare ASR dataset
You'll need to prepare a manifest file along with a transcription to perform the ASR.

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
Please note that you'll need a reasonably recent version of fairseq (i.e. [fairseq/tree/4db264940f](https://github.com/facebookresearch/fairseq/tree/4db264940f281a6f47558d17387b1455d4abd8d9)) in order to run the ASR script. You'll also need to install flashlight python bindings, follow the instructions on [this page](https://github.com/flashlight/flashlight/tree/e16682fa32df30cbf675c8fe010f929c61e3b833/bindings/python) to install. Flashlight v0.3.2 must be used to install the bindings:
```
git clone --branch v0.3.2 https://github.com/flashlight/flashlight
```

### 3.2 Emotion Classification
We fine-tune the wav2vec2-base model on audio classification task using [transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) library. We use the expressive style in the Expresso dataset as the labels.

#### 3.2.1 Pre-trained model checkpoint
You can download model checkpoint and config here
|Model Checkpoint|Config|Preprocessor Config|
|---|---|---|
|[pytorch_model.bin](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/wav2vec2_emotion_classification/pytorch_model.bin)|[config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/wav2vec2_emotion_classification/config.json)|[preprocessor_config.json](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/wav2vec2_emotion_classification/preprocessor_config.json)|

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
@misc{nguyen2023expresso,
      title={EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
      author={Tu Anh Nguyen and Wei-Ning Hsu and Antony D'Avirro and Bowen Shi and Itai Gat and Maryam Fazel-Zarani and Tal Remez and Jade Copet and Gabriel Synnaeve and Michael Hassid and Felix Kreuk and Yossi Adi and Emmanuel Dupoux},
      year={2023},
      eprint={2308.05725},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```