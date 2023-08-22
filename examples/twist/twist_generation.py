import torchaudio
import torch

from speech_lm import generate_with_offset, build_speech_lm
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder
from textless.data.speech_encoder import SpeechEncoder

def run_full_generation(hubert_encoder, twist_model, hifi_vocoder, speech_prompt):
    input_ids = hubert_encoder(speech_prompt)['units'].unsqueeze(0)
    generated_ids = generate_with_offset(twist_model, input_ids)
    full_generation = hifi_vocoder(generated_ids, dur_prediction = True)

    return full_generation



def main(args):
    audio, sample_rate = torchaudio.load(args.input_file)

    dense_model, quantizer_model, vocab = "mhubert-base-25hz", "kmeans", 500

    # Load speech encoder and vocoder
    encoder = SpeechEncoder.by_name(
        dense_model_name = dense_model,
        quantizer_model_name = quantizer_model,
        vocab_size = vocab,
        deduplicate=True,
        need_f0=False,
        add_bos_eos=False,
    ).eval()
    
    vocoder = CodeHiFiGANVocoder.by_name(
        dense_model_name = dense_model,
        quantizer_model_name = quantizer_model,
        vocab_size = vocab
    ).eval()

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        vocoder = vocoder.cuda()


    # Load twist model
    twist_model = build_speech_lm(args.twist_model_name)
    
    if audio.ndim == 2:
        audio = audio.mean(0)

    if args.prompt_duration_sec:
        prompt = int(args.prompt_duration_sec * sample_rate)
        audio = audio[:prompt]

    generated_audio = run_full_generation(encoder, twist_model, vocoder, audio)

    torchaudio.save(
        args.output_file,
        generated_audio.cpu().unsqueeze(0),
        16000,
    )


def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input filepath",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path where generated metadata is saved"
    )

    parser.add_argument(
        "--twist_model_name",
        type=str,
        default="TWIST-350M",
        choices=["TWIST-350M", "TWIST-1.3B", "TWIST-7B"],
        help="Name of TWIST model",
    )

    parser.add_argument(
        "--prompt_duration_sec",
        type=float,
        default=None,
        help="Cutting prompts to a maximum duration",
    )

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()