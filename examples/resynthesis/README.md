# Discrete Resynthesis example

In `resynth.py` we showcase a simple demonstration of the audio resynthesis done via HuBERT-based discrete pseudo-units. The code closesly
follows the [unit2speech module](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/unit2speech) of GSLM.

# How to run
Below is an example of running the script:
```bash
python resynth.py --input test_input.wav --output=test_output.wav --vocab_size=100 --decoder_steps=500
```

`resynth.py` supports the following command-line arguments:
* `--dense_model_name`: name of the dense representation model to be used (suppported: `hubert-base-ls960` and `cpc-big-ll6k`);
* `--input`: the input audio file (must have the sample rate of 16 KHz);
* `--output`: the output file name;
* `--vocab_size`: the size of the quantization vocabulary to be used (one of 50, 100, 200);
* `--decoder_steps`: determines the maximal duration of the produces audio.
