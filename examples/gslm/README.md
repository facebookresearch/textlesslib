# Generative Spoken Language Modeling pipeline

## Retrieve a language model

Assume you want to experiment with a pre-trained language model that is trained on HuBERT representations, quantized with a codebook of size 100.
Firstly, you need to download and unpack the model itself:
```bash
mkdir LM/
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz -O LM/hubert100_lm.tgz
cd LM/ && tar -xvf hubert100_lm.tgz
```
(other checkpoints can be found in the [Textless NLP GSLM release](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/ulm).)

## Run Speech Continuation on a file
To run the speech continuation pipeline with the previously downloaded models, you can use the following command:
```bash
python sample.py \
	--language-model-data-dir=LM/hubert100_lm \
	--input-file 174-84280-0004.flac \
	--output-file output_new.wav \
	--prompt-duration-sec=3 \
	--temperature=0.7 \
	--vocab-size=100
```
