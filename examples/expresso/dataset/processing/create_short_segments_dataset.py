# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This function creates an operation file which contains the operations
(copy, segments) to be performed in order to obtain the processed dataset.
The operation file can be performed with the perform_operations.py function
found in this directory.
"""


import os
from tqdm import tqdm
import soundfile
from pathlib import Path
from time import time
from collections import defaultdict


def get_manifest(audio_dir, ext = ".wav", rate=48000):
    """
    Find all the audio file from a directory and get the audio information.
    """
    # Find all sequences
    print(f"Finding all audio files with extension '{ext}' from {audio_dir}...")
    stime = time()
    audio_files = list(Path(audio_dir).rglob("*."+ext.lstrip('.')))
    print(f"Done! Found {len(audio_files)} files in {time()-stime:.2f} seconds.")

    # Getting information
    print("Getting audio information...")
    manifest = {}
    for audio in tqdm(audio_files):
        fname = Path(audio).stem
        info = soundfile.info(audio)
        frames = info.frames
        samplerate = info.samplerate
        assert samplerate == rate, \
            f"expected samplerate={rate}, found {samplerate}: {fname}"
        manifest[fname] = {
            "audio": audio,
            "frames": frames,
            "duration": frames/samplerate
        }
    return manifest


def read_split_file(split_file):
    """
    Read a split file which contains the list of audio files/segments
    belonging to a split/subset.
    Line example:
        - ex01_confused_00366
        - ex04_narration_longform_00001	(13.87s,27.74s)
        - ex03-ex01_confused_001	(,60.0s)
    """
    files = []
    with open(split_file) as f:
        for line in f:
            if line[0] == "#":
                continue
            fname, *segment = line.strip().split('\t')
            if len(segment) == 0:
                segment = (None, None)
            else:
                assert len(segment) == 1, (segment, line)
                segment = segment[0]
                assert segment[0] == '(' and segment[-1] == ')', (segment, line)
                assert len(segment.split(',')) == 2, (segment, line)
                start, end = segment[1:-1].split(',')
                start = None if start == '' else float(start.strip('s'))
                end = None if end == '' else float(end.strip('s'))
                segment = (start, end)

            files.append((fname, segment))
    print(f"Read {len(files)} files from {split_file}")
    return files


def read_vad(vad_file):
    """
    Read VAD segments from a file.
    """
    vad_dict = defaultdict(list)
    with open(vad_file) as f:
        for line in f:
            if line[0] == "#":
                continue
            fname, segments = line.replace(', ',',').split('\t')

            channelid = None
            if fname[-9:] in ['/channel1', '/channel2']:
                channelid = int(fname[-1])
                fname = fname[:-9]

            segment_list = []
            for segment in segments.split():
                assert segment[0] == '(' and segment[-1] == ')', (segment, line)
                assert len(segment.split(',')) == 2, (segment, line)
                start, end = segment[1:-1].split(',')
                start, end = float(start), float(end)
                segment_list.append((start, end, channelid))

            vad_dict[fname].extend(segment_list)
    print(f"Read VAD segments of {len(vad_dict)} files from {vad_file}")
    return vad_dict


def get_short_segments(
        file_duration,
        file_valid_segment,
        file_vad_segments,
        max_segment_len=15,
        min_segment_len=0.5
    ):
    """
    Return all segments within (min_segment_len, max_segment_len)
    Trim long vad segments into smaller segments of max_segment_len

    We denote (None, x) if the segment starts from beginning
    and (x, None) if the segment finishes at the end of file

    If the segment is taken from a specific channel,
    we denote the segment with (start, end, channelid)
    else (start, end, None)
    """

    # For files without VAD segments, consider the whole file as segment
    if file_vad_segments is None:
        file_vad_segments = [(None, None, None)] # (start, end, channelid)

    # Get the segments within the valid segment
    valid_start, valid_end = file_valid_segment
    valid_segments = []
    for segment in file_vad_segments:
        start, end, channelid = segment
        # start valid from a certain time
        if valid_start is not None:
            if start is None:
                start = valid_start
            else:
                start = max(start, valid_start)
        # end valid at a certain time
        if valid_end is not None:
            if end is None:
                end = valid_end
            else:
                end = min(end, valid_end)
        # add if segment is valid
        if start is None or end is None or start < end:
            valid_segments.append((start, end, channelid))

    # get duration of segment
    def duration(start, end):
        if start is None:
            start = 0
        if end is None:
            end = file_duration
        return end - start

    # Trim the valid segments into short segments
    short_segments = []
    for segment in valid_segments:
        start, end, channelid = segment
        if duration(start, end) <= max_segment_len:
            if duration(start, end) >= min_segment_len:
                short_segments.append((start, end, channelid))
        else:
            chunk_start = start
            while duration(chunk_start, end) > max_segment_len:
                chunk_end = chunk_start + max_segment_len if chunk_start is not None else max_segment_len
                assert chunk_end <= file_duration, (start, end, file_duration)
                short_segments.append((chunk_start, chunk_end, channelid))
                chunk_start = chunk_end
            if duration(chunk_start, end) >= min_segment_len:
                short_segments.append((chunk_start, end, channelid))

    return short_segments


def reduce_operations(operations, sort = True):
    """
    Get a common source root directory and target root directory for the file paths in operations
    """
    if sort:
        def get_key(line):
            # operation line format: op file1 xxx file2
            parts = line.split()
            return ' '.join([parts[0], parts[1], parts[-1]])
        operations = sorted(operations, key=get_key)

    src_root_dir = '/'.join(os.path.commonprefix([line.split()[1] for line in operations]).split('/')[:-1])
    tgt_root_dir = '/'.join(os.path.commonprefix([line.split()[-1] for line in operations]).split('/')[:-1])
    reduced_operations = []
    reduced_operations.append(f"src_root_dir: {src_root_dir}")
    reduced_operations.append(f"tgt_root_dir: {tgt_root_dir}")
    for line in operations:
        reduced_line = line.replace(f'\t{src_root_dir}/', '\t').replace(f' {tgt_root_dir}/', ' ')
        reduced_operations.append(reduced_line)
    return reduced_operations


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create short segments Expresso dataset for speech resynthesis')
    parser.add_argument('expresso_audio_dir', type=str,
                    help='Path to the expresso audio dataset')
    parser.add_argument('output_expresso_dir', type=str,
                    help='Path to the directory containing output audio')
    parser.add_argument('output_operations_file', type=str,
                    help='Path to the output operations file')
    parser.add_argument('--splits_dir', type=str,
                    help='Path to the directory containing train/dev/test split')
    parser.add_argument('--vad_file', type=str,
                    help='Path to the VAD segments file')
    parser.add_argument('--audio_rate', type=int, default=48000,
                    help='Expected sample rate of audio')
    parser.add_argument('--max_segment', type=float, default=15,
                    help='Max length of segment (in seconds)')
    parser.add_argument('--min_segment', type=float, default=0.5,
                    help='Min length of segment (in seconds)')
    parser.add_argument('--create_link', action='store_true',
                    help='Create link instead of copy original audio file (for short audio files)')
    args = parser.parse_args()

    audio_dir = Path(args.expresso_audio_dir)

    if args.splits_dir is None:
        # automatically find in the expresso directory
        splits_dir = audio_dir.parent / "splits"
    else:
        splits_dir = Path(args.splits_dir)

    if args.vad_file is None:
        # automatically find in the expresso directory
        vad_file = audio_dir.parent / "VAD_segments.txt"
    else:
        vad_file = Path(args.vad_file)

    # Get all audio files
    expresso_files = get_manifest(audio_dir, rate=args.audio_rate)

    # Get VAD segments
    vad_segments = read_vad(vad_file)

    # Get short segments
    # dictionary over filename/channel, value is a list of (start, end, channel, split)
    short_segments = defaultdict(list)
    for split in ["train", "dev", "test"]:
        split_files = read_split_file(splits_dir/(split+".txt"))

        print(f"Getting short segments for split: '{split}'...")
        for fname, file_valid_segment in tqdm(split_files):
            file_info = expresso_files[fname]
            # For base files, we don't have vad segments
            file_vad = vad_segments.get(fname, None)
            # Get short segments
            file_short_segments = get_short_segments(
                file_info["duration"], file_valid_segment, file_vad,
                args.max_segment, args.min_segment
            )
            # Add segments
            for (start, end, channelid) in file_short_segments:
                if channelid is not None:
                    key = f"{fname}/{channelid}"
                else:
                    key = fname
                short_segments[key].append((start, end, channelid, split))

    # Sort the segments by time & validate segments
    def get_start(segment):
        return 0 if segment[0] is None else segment[0]
    for key in short_segments:
        segments = sorted(short_segments[key], key=get_start)
        short_segments[key] = segments
        # validate segments
        for i in range(1, len(segments)):
            assert segments[i][0] >= segments[i-1][1], (key, segments[i-1:i+1])

    # Get information of keys (to get target dir)
    keyinfo = {}
    for key in short_segments:
        fname, *channels = key.split('/')
        channel = None
        if len(channels) == 1:
            channel = int(channels[0])

        if len(fname.split('_')[0].split('-')) == 2:
            # e.g. ex01-ex02_default_001
            assert channel is not None, key
            datatype = 'conversational'
            speaker = fname.split('_')[0].split('-')[channel-1]
            styles = fname.split('_')[1].split('-')
            if len(styles) == 1: # same style for both speaker
                style = styles[0]
            else:
                style = styles[channel-1]
        else:
            # e.g. ex01_default_00001 (there should be also ex01_default_emphasis_00001) but still default
            assert channel is None, key
            datatype = 'read'
            speaker = fname.split('_')[0]
            style = fname.split('_')[1]

        keyinfo[key] = speaker, datatype, style, fname

    # Write the operations to a file (to verify and perform them later)
    COPY_OP = "link" if args.create_link else "copy"
    SEGMENT_OP = "segment"
    OUTPUT_DIR = Path(args.output_expresso_dir)
    operations = []
    for key in short_segments:
        speaker, datatype, style, fname = keyinfo[key]
        substyle = datatype[:4]+'-'+style
        source_path = expresso_files[fname]["audio"]
        segments = short_segments[key]
        for segment_id, (start, end, channelid, split) in enumerate(segments):
            target_dir = OUTPUT_DIR / split / speaker / substyle

            # Copy the whole file (without segmenting)
            if (start, end) == (None, None):
                assert len(segments) == 1, (key, segments)
                assert channel is None, (key, segments)
                target_path = target_dir / (fname + '.wav')
                operations.append(f"{COPY_OP}\t{source_path} {target_path}")
            # Segmented files
            else:
                start = f"{start:.2f}s" if start is not None else ''
                end = f"{end:.2f}s" if end is not None else ''
                if datatype == "read":
                    assert channelid is None, (key, segments)
                    #rss stands for read short segment
                    target_path = target_dir / (fname + f'_rss{segment_id:03d}.wav')
                    operations.append(f"{SEGMENT_OP}\t{source_path} ({start},{end}) {target_path}")
                elif datatype == "conversational":
                    assert channelid in [1, 2], (key, segments)
                    #css stands for conversational short segment
                    target_path = target_dir / (fname + f'-{speaker}_{style}_css{segment_id:03d}.wav')
                    operations.append(f"{SEGMENT_OP}\t{source_path} ({start},{end},{channelid}) {target_path}")

    operations = reduce_operations(operations)
    INSTRUCTIONS = [
        "# Operations to be performed. Example operation format:",
        '# "copy\\t{src_file} {tgt_file}" or "link\\t{src_file} {tgt_file}"',
        '# "segment\\t{src_file} (start,end) {tgt_file}" or "segment\\t{src_file} (start,end,channelid) {tgt_file}"',
    ]
    print(f"Writing {len(operations)-2} operations to {args.output_operations_file}")
    with open(args.output_operations_file, 'w') as f:
        for line in INSTRUCTIONS + operations:
            f.write(line+'\n')

