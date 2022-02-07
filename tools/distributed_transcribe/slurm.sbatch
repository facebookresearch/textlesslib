# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


## Set output and job name
#SBATCH --job-name=transcribe
#SBATCH --output=log.out
#SBATCH --error=log.err
## partition name
#SBATCH --partition=devlab

## number of nodes, tasks per nodes, etc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=32
#SBATCH --time 4320
#SBATCH --mem 400G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2

DENSE_NAME=hubert-base-ls960
QUANTIZER_NAME=kmeans
VOCAB_SIZE=50
MANIFEST=manifest.tsv
TRANSCRIPT=transcript

srun -u python transcribe.py \
    --manifest $MANIFEST \
    --output=$TRANSCRIPT \
    --dense_model=$DENSE_NAME \
    --quantizer_model=$QUANTIZER_NAME \
    --vocab_size=$VOCAB_SIZE \
    --durations --deduplicate
