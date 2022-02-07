# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


DENSE_NAME=hubert-base-ls960
QUANTIZER_NAME=kmeans
VOCAB_SIZE=50
MANIFEST=manifest.tsv
TRANSCRIPT=transcript

python transcribe.py \
    --manifest $MANIFEST \
    --output=$TRANSCRIPT \
    --dense_model=$DENSE_NAME \
    --quantizer_model=$QUANTIZER_NAME \
    --vocab_size=$VOCAB_SIZE \
    --durations --deduplicate
