from transformers import AutoModelForCausalLM
import torch
import os
import wget
import zipfile

ROOT_URL = 'https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/'

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"File extracted to {extract_path}")


def maybe_download_speech_lm(name):
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')
    
    ckpt_dir = os.path.join('./ckpts', name)
    if not os.path.exists(ckpt_dir):
        url = ROOT_URL + name + '.zip'
        zip_path = ckpt_dir + '.zip'
        print(f"Downloading from {url}")
        filename = wget.download(url, zip_path)
        unzip_file(filename, ckpt_dir)

    return os.path.abspath(ckpt_dir)


def build_speech_lm(name):
    """
    retruns (model, offset)
    """
    ckpt_dir = maybe_download_speech_lm(name)

    lm_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        lm_model = lm_model.cuda()
    lm_model.eval()

    return lm_model

def generate_with_offset(lm_model, input_ids, offset=None):
    if offset is None:
        offset = lm_model.config.offset
    input_len = int(input_ids.shape[-1])
    input_ids = input_ids.to(lm_model.device)
    generated_ids = lm_model.generate(offset + input_ids, max_length=3 * input_len, do_sample=True, temperature=0.8)
    return generated_ids - offset

if __name__ == "__main__":
    lm_model = build_speech_lm("TWIST-350M")
    input_ids = torch.randint(0, 500, (1, 15, ))
    generated_ids = generate_with_offset(lm_model, input_ids)
    print(input_ids)
    print(generated_ids)
