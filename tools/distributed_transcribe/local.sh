# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

WORKERS_PER_NODE=4
DENSE_NAME=hubert-base-ls960
QUANTIZER_NAME=kmeans
VOCAB_SIZE=50
MANIFEST=manifest.tsv
TRANSCRIPT=transcript

python -m torch.distributed.run --nproc_per_node=$WORKERS_PER_NODE transcribe.py \
    --manifest $MANIFEST \
    --output=$TRANSCRIPT \
    --dense_model=$DENSE_NAME \
    --quantizer_model=$QUANTIZER_NAME \
    --vocab_size=$VOCAB_SIZE \
    --durations --deduplicate
