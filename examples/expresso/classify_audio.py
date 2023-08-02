# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import os
from tqdm import tqdm


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(os.path.join(root, f))
    return out


def load_classification_model(checkpoint_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint_path)
    model = AutoModelForAudioClassification.from_pretrained(checkpoint_path).cuda().eval()
    label_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Classification model loaded from {checkpoint_path} !")
    return feature_extractor, model, label_names

@torch.inference_mode()
def predict_audio(audio, feature_extractor, model, label_names):
    if isinstance(audio, str):
        speech, _ = torchaudio.load(audio)
        resampler = torchaudio.transforms.Resample(feature_extractor.sampling_rate)
        speech = resampler(speech).squeeze().numpy()
    else:
        speech = audio

    features = feature_extractor(speech, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    features['input_values'] = features['input_values'].cuda()

    logits = model(**features).logits
    pred_id = torch.argmax(logits, dim=-1)[0].item()

    return label_names[pred_id], logits.detach().cpu().numpy()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reorganize and move audio to many sub-directories')
    parser.add_argument('--model_ckpt', type=str, required=True,
                    help='Path to the model checkpoint')
    parser.add_argument('--from_tsv', type=str,
                    help='Path to the manifest file if given')
    parser.add_argument('--from_dir', type=str,
                    help='Path to the audio directory if given')
    parser.add_argument('--extension', type=str, default='.wav',
                    help='Extension of the file to find (in case a directory is given)')
    parser.add_argument('--label_file', type=str,
                    help='Path to the label file to compute accuracy')
    parser.add_argument('--output_file', type=str,
                    help='Path to the output file to write predictions')
    parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite the output file')
    args = parser.parse_args()

    # Assertions
    assert (args.from_tsv is None) != (args.from_dir is None), \
        "Exactly one of from_tsv and from_dir must be given!"
    assert (args.label_file is not None) or (args.output_file is not None), \
        "At least one of label_file or output_file must be given"
    if args.label_file:
        assert os.path.exists(args.label_file), "label file doesn't exist!"
    if args.output_file and not args.overwrite:
        assert not os.path.exists(args.output_file), \
            f"Output file {args.output_file} exists, consider using --overwrite"

    # Load the dataset
    if args.from_tsv:
        print(f"Reading manifest file {args.from_tsv}")
        with open(args.from_tsv) as f:
            root = f.readline().strip()
            files = []
            nframes = []
            for line in f:
                files.append(os.path.join(root, line.split()[0]))
                nframes.append(int(line.split()[1]))
        print(f"{len(files)} files found from the manifest file!")
    elif args.from_dir:
        print(f"Finding all audio files with extension '{args.extension}' from {args.from_dir}...")
        files = find_all_files(args.from_dir, args.extension)
        files = [os.path.abspath(path) for path in files]
        print(f"Done! Found {len(files)} files.")

    # Load classification model
    feature_extractor, model, label_names = load_classification_model(args.model_ckpt)

    # Perform prediction
    print("Performing audio prediction...")
    predictions = []
    for audio_path in tqdm(files):
        predictions.append(predict_audio(audio_path, feature_extractor, model, label_names)[0])
    print("...done!")

    if args.output_file is not None:
        # Create output dirs
        outdir = os.path.dirname(args.output_file)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        # Write the output
        with open(args.output_file, 'w') as f:
            line = ['audio']
            f.write("audio\tprediction\n")
            for audio, pred in zip(files, predictions):
                f.write(f"{audio}\t{pred}\n")
        print(f"Wrote the predictions to {args.output_file}")

    # Compute the accuracy if the label file is given
    if args.label_file:
        assert os.path.exists(args.label_file), "label file doesn't exist!"
        with open(args.label_file) as f:
            true_labels = [line.strip() for line in f]
        assert len(true_labels) == len(predictions), \
            f"mismatch between number of audios and labels ({len(true_labels)} vs {len(predictions)})"
        correct = sum(label==pred for label, pred in zip(true_labels, predictions))
        accuracy = correct/len(true_labels)
        print(f"Accuracy: {100*accuracy:.2f} % ({correct}/{len(true_labels)} correct)")

