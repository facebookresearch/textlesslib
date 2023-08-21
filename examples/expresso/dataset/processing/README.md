# Processing Expresso dataset
We share the processing scripts for the Expresso dataset used in the paper: "EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis".

We prepare the Expresso dataset for speech synthesis experiments by cutting down to short segments of less than 15 seconds based on VAD information obtained from [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## Expresso data splits
The exprresso data splits (train/dev/test) are shared in the [expresso_splits](expresso_splits) directory.

## VAD segments
The VAD segments of long speech (longform read speech and conversational speech) can be found in the [expresso_VAD_segments.txt](expresso_VAD_segments.txt) file.

## Create the short segments dataset
We'll process the dataset by first creating a text file `operations.txt` containing the operations (e.g. copy files, segment files) to be performed, and then perform the operations with the [perform_operations](perform_operations.py) script.

### 1. Create operations

```bash
python create_short_segments_dataset.py \
    $EXPRESSO_AUDIO_DIR \
    $OUTPUT_AUDIO_DIR \
    $OUTPUT_OPERATION_FILE \
    --splits_dir expresso_splits \
    --vad_file expresso_VAD_segments.txt \
    --create_link
```

### 2. Perform operations
```bash
python perform_operations.py \
    $OPERATION_FILE \
    --n_processes 10
```

### 3. (Optional) Convert to 16khz
```bash
python convert_to_16k.py \
    $AUDIO_DIR_48khz \
    $AUDIO_DIR_16khz
```