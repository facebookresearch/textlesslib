# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from pathlib import Path
from collections import defaultdict
import scipy.io.wavfile as wavfile
from multiprocessing import Pool
from tqdm import tqdm
from time import time


OPERATIONS_LIST = ["copy", "link", "segment"]


def read_operations(operations_file):
    """
    Read operations from an operations file.
    Example operation format:
        - "copy\t{src_file} {tgt_file}" or "link\t{src_file} {tgt_file}"
        - "segment\t{src_file} (start,end) {tgt_file}" or "segment\t{src_file} (start,end,channelid) {tgt_file}"
    """

    operation_list = []
    src_root_dir = Path("")
    tgt_root_dir = Path("")
    with open(operations_file) as f:
        c = 0
        for line in f:
            if not line or line[0] == "#":
                continue
            c += 1
            if line.startswith("src_root_dir:"):
                assert c == 1, "expected src_root_dir in the first line"
                src_root_dir = Path(line.split()[1])
                continue
            elif line.startswith("tgt_root_dir:"):
                assert c == 2, "expected tgt_root_dir in the second line"
                tgt_root_dir = Path(line.split()[1])
                continue
            op, items = line.strip().split("\t")

            assert (
                op in OPERATIONS_LIST
            ), "expected operation in list: {OPERATIONS_LIST}, found '{op}'"

            if op == "segment":
                src_file, seg, tgt_file = items.split()
                src_file = src_root_dir / src_file
                tgt_file = tgt_root_dir / tgt_file
                operation_list.append((op, (src_file, seg, tgt_file)))
            else:
                src_file, tgt_file = items.split()
                src_file = src_root_dir / src_file
                tgt_file = tgt_root_dir / tgt_file
                operation_list.append((op, (src_file, tgt_file)))

    print(f"Read {len(operation_list)} operations from {operations_file}")

    return operation_list


def group_operations(operation_list):
    """
    Group operations to efficiently perform them.
    E.g. group multiple segment operations from the same file.
    Output operation examples:
    - ('copy', ('src_file','tgt_file'))
    - ('segment', ('src_file', [('seg_1', 'tgt_file_1'),..., ('seg_N', 'tgt_file_N')]))
    """

    # only need to group segment operations for now
    # Dict of src file as key, so that segment operations
    # for one src file are performed at one time
    grouped_segment_operations = defaultdict(list)

    grouped_operation_list = []
    for op, items in operation_list:
        if op == "segment":
            src_file, seg, tgt_file = items
            grouped_segment_operations[src_file].append((seg, tgt_file))
        else:
            grouped_operation_list.append((op, items))

    for src_file, tgt_segments in grouped_segment_operations.items():
        grouped_operation_list.append(("segment", (src_file, tgt_segments)))

    print(f"Grouped to {len(grouped_operation_list)} operations")

    return grouped_operation_list


def copy_file(src_file, tgt_file):
    # Create tgt dir
    tgt_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy the files
    shutil.copyfile(src_file, tgt_file)


def link_file(src_file, tgt_file):
    # Create tgt dir
    tgt_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy the files
    os.symlink(src_file, tgt_file)


def segment_file_grouped(src_file, segment_list):
    # Read the audio
    sr, audio_data = wavfile.read(src_file)

    for seg, tgt_file in segment_list:
        # Create tgt dir
        tgt_file.parent.mkdir(parents=True, exist_ok=True)

        # Get segment info
        assert seg[0] == "(" and seg[-1] == ")", (src_file, seg, tgt_file)
        seg = seg[1:-1].split(",")
        assert len(seg) in [2, 3]

        # Get start and end frames from segment
        start_time, end_time = seg[:2]
        start_time = float(start_time.rstrip("s")) if start_time != "" else None
        start_frame = round(start_time * sr) if start_time else 0
        end_time = float(end_time.rstrip("s")) if end_time != "" else None
        end_frame = round(end_time * sr) if end_time else len(audio_data)

        # Get segmented audio data
        seg_data = audio_data[start_frame:end_frame]
        if len(seg) == 3:
            assert seg[2] in ["1", "2"], seg
            channel = int(seg[2]) - 1
            seg_data = seg_data[:, channel]

        # Write the segmented audio data
        if len(seg_data) > 0:
            wavfile.write(tgt_file, sr, seg_data)


def process_one_operation(operation):
    """
    operation examples:
    - ('copy', ('src_file','tgt_file'))
    - ('segment', ('src_file', [('seg_1', 'tgt_file_1'),..., ('seg_N', 'tgt_file_N')]))
    """
    op, item = operation
    assert (
        op in OPERATIONS_LIST
    ), f"expected operation in list: {OPERATIONS_LIST}, found '{op}'"
    if op == "copy":
        copy_file(*item)
    elif op == "link":
        link_file(*item)
    elif op == "segment":
        segment_file_grouped(*item)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform operations from an operations file"
    )
    parser.add_argument("operations_file", type=str, help="Path to the operations file")
    parser.add_argument(
        "--n_processes",
        type=int,
        default=10,
        help="Number of processes to be processed at the same time",
    )
    args = parser.parse_args()

    # Read operations
    operation_list = read_operations(args.operations_file)

    # Perform operations
    operation_list = group_operations(operation_list)

    # List all operations
    operations_dict = defaultdict(list)
    for op, items in operation_list:
        operations_dict[op].append((op, items))

    # Perform by each operations
    print(f"Processing operations with {args.n_processes} processes...")
    stime = time()
    for op, operations in operations_dict.items():
        print(f"Processing {len(operations)} '{op}' operations")
        with Pool(args.n_processes) as pool:
            for _ in tqdm(
                pool.imap(process_one_operation, operations), total=len(operations)
            ):
                continue
    print(f"...done in {time()-stime:.2f} seconds!")
