# Distributed pseudo-units transcription

If you ever tried to transcribe large-scale aduio datasets (e.g. [LibriLight](https://github.com/facebookresearch/libri-light) dataset with 60k hours) into discrete pseudo-units such as used by the [Generative Spoken Language Modeling (GSLM)](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm) system, you might have noticed that this task is computationally intensive and might be impractical to do in a non-distributed fashion.

This tool provides a convenient script that can leverage multiple GPUs (on multiple nodes!) to speed up and parallelize pseudo-unit transcription.
We provide recipies for two scenarios: (a) single-node, multiprocess/multi-GPU transcription that leverages distributed.run mechanism of Pytorch, and (b) multi-node, multi-GPU transcription that can be run on a SLURM-managed cluster.

## Example scripts

* `local.sh` runs provides an example of a command to transcribe a dataset in a local parallel mode;
* `slurm.sbatch` is an example of a SLURM sbatch script for a distributed pseudo-unit transcription.

Finally, `transcribe.py` can be run directly as a single process (see single.sh):
```
DENSE_NAME=hubert-base-ls960
KMEANS_NAME=hubert-base-ls960-kmeans-100
MANIFEST=manifest.tsv
TRANSCRIPT=transcript

python transcribe.py \
    --manifest $MANIFEST \
    --output=$TRANSCRIPT \
    --dense_model=$DENSE_NAME \
    --kmeans_model=$KMEANS_NAME
 ```
 
## Command line arguments

The transcription script, `transcribe.py` has a few command-line arguments:
* `--dense_model`: sets the dense Hubert model to be used (by its name, e.g. `hubert-base-ls960`);
* `--kmeans_model`: sets the k-mean quantizer to be used, e.g. `hubert-base-ls960-kmeans-100`;
* `--manifest`: specifies the manifest file describing the dataset;
* `--output`: path to the output transcript file. Unit stream will be stored in `<output>.units` file, durations (if requested) - in `<output>.durations`, and F0 values (again, if requested) in `<output>.f0s`;
* `--deduplicate`: if set, consecutive repeats of the same pseudo-unit are collapsed (as it is done in GSLM);
* `--durations`: if set, duration of each token is reported in a `<output>.durations` file (note that if `--deduplicate` is not set, all durations will be equal to 1);
* `--f0s`: if set, duration of mean F0 that correspond to each token is reported in a `<output>.f0s` file (note: F0 extraction is slow). F0 values are rounded to the closest integer and are measured in Hz;
* `--preserve_name`: if set, the transcript contains names of the original audio files;
* `--separator`: a separator between pseudo-unit tokens in the outputs;
* `--distributed_port`: a unique port, required for distributed transcription (defaults to 58554).


## Input format
`transribe.py` takes a manifest file describing an input dataset. A manifest is a tab-separated file with simple format: (a) the first line is a root of the dataset's folder, and (b) each line specifies a relative path to an audio file and its size in frames. Here is an example of a manifest corresponding to LibriSpeech dev-clean:
```
/datasets/librispeech/dev-clean
1272/128104/1272-128104-0000.flac 93680
1272/128104/1272-128104-0001.flac 77040
1272/128104/1272-128104-0002.flac 199760
1272/128104/1272-128104-0003.flac 158400
1272/128104/1272-128104-0004.flac 470400
1272/128104/1272-128104-0005.flac 144160
```
(`transcribe.py` ignores the duration field.)

**NB**: fairseq has [an utility](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py) for creating manifest files. 

## Output format

`transcribe.py` outputs one line per file, with pseudo-units separated by spaces (by default). Hence the output would look something like
```
71 12 56 57 40 63 40 63 93 50 76 53 62 ... 55 20
...
71 12 56 57 56 57 40 57 86 58 9 1 27 31 23 69 44 26 ...
```

This format is directly compatible with fairseq-preprocessing. However, if there is a need to link a particular line to its original file, please use `--preserve_name` flag.
