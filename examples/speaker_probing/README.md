# Speaker probing example

This directory contains a short example that illustrates the speaker probing task. Specifically, we investigate whether an anonymised speaker id can
be predicted based on their utterances representated as (potentially quantized) HuBERT representations. This example uses LibriSpeech dev-clean as a dataset.

## Running example
To train a simple speaker classifier and get its accuracy on validation data, it is enough to simply run a command:
```python train.py --model_type=discrete --seed=0 --epochs=5 --vocab_size=50```
This will train a small Transformer model on HuBERT representations, quantized into a vocabulary of 50 pseudo-units.

## Command-line arguments
* `--dense_model_name`: dense model to be used. Must be either `hubert-base-ls960` or `cpc-big-ll6k`;
* `--seed`: sets the random seed;
* `--epochs`: sets the number of training epochs;
* `--vocab_size`: sets the size of the codebook. The example uses pre-trained codebooks and support vocabulary sizes of 50, 100, and 200;
* `--model_type`: selects the model/representation to be used. Must be one of [`discrete`, `continuous`, `baseline` (default)].


