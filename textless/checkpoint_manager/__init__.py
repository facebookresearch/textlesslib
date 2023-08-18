# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from .manager import Checkpoint, CheckpointManager

root = os.environ.get("TEXTLESS_CHECKPOINT_ROOT", "~/.textless/")
CHECKPOINT_MANAGER: CheckpointManager = CheckpointManager(disk_root=root)


def populate_checkpoints():
    global CHECKPOINT_MANAGER

    # HuBERT based
    checkpoints = [
        # Dense model
        Checkpoint(
            name="hubert-base-ls960",
            remote_path="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
            fname="hubert_base_ls960.pt",
            sha256="1703cf8d2cdc76f8c046f5f6a9bcd224e0e6caf4744cad1a1f4199c32cac8c8d",
        ),
        Checkpoint(
            name="mhubert-base-vp_en_es_fr",
            remote_path="https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt",
            fname="mhubert_base_vp_en_es_fr_it3.pt",
            sha256="23527cd6311a6617bfa199f574e2eed478cd6a58a634f4007165c10ec0019832",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang",
            remote_path="https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3.pt",
            fname="mhubert_base_vp_mls_cv_8lang_it3.pt",
            sha256="fa82eaccb557465f4732662e3a744c30c953059147ac6a4c92f1f3e0daeda61e",
        ),
        Checkpoint(
            name="mhubert-base-25hz",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher.pt",
            fname="mhubert_base_25hz_cp_mls_cv_sp_fisher.pt",
            sha256="edf0f5be2581a568a236aeb2781a5607c5af25806e12f40f7f2391e861fe3cf0",
        ),
        # Quantizers
        Checkpoint(
            name="hubert-base-ls960-kmeans-50",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin",
            fname="hubert_base_ls960_km50.pt",
            sha256="d01a7d5bc2c54b7b5f25f321ba525b4d230b06e3927f90bad0394198bc89f494",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-100",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin",
            fname="hubert_base_ls960_km100.pt",
            sha256="f14a3104615485381fc489701d6761c9abbdbb0d43607cf55518a1e2891023fe",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-200",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin",
            fname="hubert_base_ls960_km200.pt",
            sha256="b3c46c9cdd1707ad852dd53c359aa73942e79d53c432a9a8a419ed046408024b",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-500",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km500/km.bin",
            fname="hubert_base_ls960_km500.pt",
            sha256="411c8668e1314751404f58636f935fc73540a6793890435da4a8ffadf157398e",
        ),
        Checkpoint(
            name="mhubert-base-vp_en_es_fr-layer-11-kmeans-1000",
            remote_path="https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
            fname="mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
            sha256="c70f5416c38044d0c20e33fdeedf4533974a8ac4029985cdef3b755038381a0b",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-500",
            remote_path="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin",
            fname="hubert_base_ls960_L9_km500.pt",
            sha256="4450dcbc90aa7d0553306a4b0fafef35c9b3c840468f02969273dfbb9db89a23",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-expresso-2000",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hubert_base_ls960_L9_km2000_expresso.bin",
            fname="hubert_base_ls960_L9_km2000_expresso.pt",
            sha256="ae73e3cc7ea0426faf43c8bb67fc3846a0f8708232526d045a101836e8c7929f",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-2000",
            remote_path="https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3_L12_km2000.bin",
            fname="mhubert_base_vp_mls_cv_8lang_it3_L12_km2000.pt",
            sha256="05de230870427dcdce334d29922ccce354c7e7d0574edc645f5daed47b1ccc18",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-expresso-2000",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso.bin",
            fname="mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso.pt",
            sha256="bda2dbf5fe42bf59359799b281498a9d7fd0a9ec703d6dc54a4a4a0ff70a4e22",
        ),
        Checkpoint(
            name="mhubert-base-25hz-kmeans-500",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin",
            fname="mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin",
            sha256="03cc04a9c24fec4285e73e709c485756d8f116aa8e724eac555de6a7cf8d28ad",
        ),
        # Tacotron2
        Checkpoint(
            name="hubert-base-ls960-kmeans-50-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km50/tts_checkpoint_best.pt",
            fname="hubert_base_ls960_kmeans_50_tacotron.pt",
            sha256="335e881a897cfa3389804110de8ac3909159d4de395880fbf1d3167a9477451e",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-100-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/tts_checkpoint_best.pt",
            fname="hubert_base_ls960_kmeans_100_tacotron.pt",
            sha256="b208f8d6433eb5524405aa29d2b5fdacddb63a182d9830b629232e63b3543e4d",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-200-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/tts_checkpoint_best.pt",
            fname="hubert_base_ls960_kmeans_200_tacotron.pt",
            sha256="093f009935a4dadd692db85e859246e369cb17be3aecd22038fb70af4d5b0590",
        ),
        # NB: must be named '*-codes'
        Checkpoint(
            name="hubert-base-ls960-kmeans-50-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km50/code_dict",
            fname="hubert_base_ls960_kmeans_50_tacotron_codes.pt",
            sha256="5f01dd57fd3b4044fac93aaac2589bf49e34cbe1dc0713254c0f339ba2123bce",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-100-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/code_dict",
            fname="hubert_base_ls960_kmeans_100_tacotron_codes.pt",
            sha256="6d506216aa5bad159f167e2535293b4e5ec8e1073b64449d30b66b460ebf6da0",
        ),
        Checkpoint(
            name="hubert-base-ls960-kmeans-200-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/code_dict",
            fname="hubert_base_ls960_kmeans_200_tacotron_codes.pt",
            sha256="ea01ba3592e27c871b63b32e37d6532234edf7eee7077bdcc094061ee72922e6",
        ),
        # Hifigan
        Checkpoint(
            name="mhubert-base-25hz-kmeans-500-hifigan",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/hifigan_lj_mhubert_base_25hz.pt",
            fname="hifigan_lj_mhubert_base_25hz.pt",
            sha256="d88224e95c501e2cd59a6e4014753169cfe060fb7ade3cc0da03c809fef73b79",
        ),
        Checkpoint(
            name="mhubert-base-25hz-kmeans-500-hifigan-config",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/hifigan_lj_mhubert_base_25hz_config.json",
            fname="hifigan_lj_mhubert_base_25hz_config.json",
            sha256="116dc39be4970cd393e562acd0a0ec70a86aaba7e9ca469b8b836cb3fe46afc7",
        ),
    ]

    # Expresso Hifigan models
    checkpoints += [
        # hubert_base_km500_L9
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-500-hifigan",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/generator.pt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500_generator.pt",
            sha256="d50f4704e961c29cb386924ef32cd0c3a4d1089a3f9be856f8a3b6dcfdd6147a",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-500-hifigan-config",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/config.json",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500_config.json",
            sha256="31e65ef5e100cb43de08f2c8d225496ace1ed87a49fcb8ee44bd6805f079b1f0",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-500-hifigan-speakers",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/speakers.txt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500_speakers.txt",
            sha256="2efabb7a4ad786da34fe6fbe99c3906b7abd40b2e01c81b7efde76142ae3d552",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-500-hifigan-styles",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/styles.txt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500_styles.txt",
            sha256="a6f63206518448277c63c3a251ac13484784829cf2e1b0a23b0860dd32543520",
        ),
        # hubert_base_km2000_L9_expresso
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-expresso-2000-hifigan",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/generator.pt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso_generator.pt",
            sha256="b892fdcca889f4bc6d7f0a0f407ae5d01eca8aaa66b9be333ef33a1c50f2ef52",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-expresso-2000-hifigan-config",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/config.json",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso_config.json",
            sha256="73932563cedc43cd1dcddfab36ce3b2c3cb76eef3423d25d1fb536ce1ec513bb",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-expresso-2000-hifigan-speakers",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/speakers.txt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso_speakers.txt",
            sha256="2efabb7a4ad786da34fe6fbe99c3906b7abd40b2e01c81b7efde76142ae3d552",
        ),
        Checkpoint(
            name="hubert-base-ls960-layer-9-kmeans-expresso-2000-hifigan-styles",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso/styles.txt",
            fname="hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km2000_expresso_styles.txt",
            sha256="a6f63206518448277c63c3a251ac13484784829cf2e1b0a23b0860dd32543520",
        ),
        # mhubert_base_vp_mls_cv_8lang_it3_L12_km2000
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-2000-hifigan",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/generator.pt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_generator.pt",
            sha256="0d160c97519d3ed644e2ec636581b79e56767d60325308849fbb389a494c1466",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-2000-hifigan-config",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/config.json",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_config.json",
            sha256="73932563cedc43cd1dcddfab36ce3b2c3cb76eef3423d25d1fb536ce1ec513bb",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-2000-hifigan-speakers",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/speakers.txt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_speakers.txt",
            sha256="2efabb7a4ad786da34fe6fbe99c3906b7abd40b2e01c81b7efde76142ae3d552",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-2000-hifigan-styles",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000/styles.txt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_styles.txt",
            sha256="a6f63206518448277c63c3a251ac13484784829cf2e1b0a23b0860dd32543520",
        ),
        # mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-expresso-2000-hifigan",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/generator.pt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso_generator.pt",
            sha256="8d1585f20edd9b0982f80acbb387a892adf19e4aa719b8dea795c8fa96285cbc",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-expresso-2000-hifigan-config",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/config.json",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso_config.json",
            sha256="73932563cedc43cd1dcddfab36ce3b2c3cb76eef3423d25d1fb536ce1ec513bb",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-expresso-2000-hifigan-speakers",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/speakers.txt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso_speakers.txt",
            sha256="2efabb7a4ad786da34fe6fbe99c3906b7abd40b2e01c81b7efde76142ae3d552",
        ),
        Checkpoint(
            name="mhubert-base-vp_mls_cv_8lang-kmeans-expresso-2000-hifigan-styles",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso/styles.txt",
            fname="hifigan_expresso_lj_vctk_mhubert_base_vp_mls_cv_8lang_it3_L12_km2000_expresso_styles.txt",
            sha256="a6f63206518448277c63c3a251ac13484784829cf2e1b0a23b0860dd32543520",
        ),
    ]

    # CPC-based stuff
    checkpoints += [
        # dense model
        Checkpoint(
            name="cpc-big-ll6k",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/cpc_big_ll6kh_top_ctc.pt",
            fname="cpc_big_ll6kh_top_ctc.pt",
            sha256="73155dad5d7c986fe7b7f548050060a8e9cc9a0ffd111a22932f38c3e617c5b8",
        ),
        # Quantizers
        Checkpoint(
            name="cpc-big-ll6k-kmeans-50",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km50/km.bin",
            fname="cpc_big_ll6k_km50.pt",
            sha256="c48be5717aebc08169aa5165d58267449b8c8568624a346bb9f4b26eac3b0240",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-100",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km100/km.bin",
            fname="cpc_big_ll6k_km100.pt",
            sha256="077b96e010b1e87be627ef2bef0f5e5cdaa1c01722aa363b2dc0bb6a638e8b26",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-200",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km200/km.bin",
            fname="cpc_big_ll6k_km200.pt",
            sha256="2d863d1c6f251d19e667998248c87642ecb5e3ffaa410e9ebfa64341db4f6de2",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-500",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km500/km.bin",
            fname="cpc_big_ll6k_km500.pt",
            sha256="35398f665dff06801a7bf1e595bc176711ed751d554e64d999dc42f1dd106561",
        ),
        # Tacotron2
        Checkpoint(
            name="cpc-big-ll6k-kmeans-50-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km50/tts_checkpoint_best.pt",
            fname="cpc_big_ll6k_kmeans_50_tacotron.pt",
            sha256="e80a46561d1ded73bbe6e7272fcd6b9943fb7607229040b06324f654b66396fd",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-100-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km100/tts_checkpoint_best.pt",
            fname="cpc_big_ll6k_kmeans_100_tacotron.pt",
            sha256="6cfe1ce4bbdfd0f531189f4013cefc9e1296aec62f8bc0172d68d088ab344a50",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-200-tacotron",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km200/tts_checkpoint_best.pt",
            fname="cpc_big_ll6k_kmeans_200_tacotron.pt",
            sha256="5845b1c0a82d9176b9d8c33a32f098795c946077e741a9da2b665bafda17b7ff",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-50-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km50/code_dict",
            fname="cpc_big_ll6k_kmeans_50_tacotron_codes.pt",
            sha256="5f01dd57fd3b4044fac93aaac2589bf49e34cbe1dc0713254c0f339ba2123bce",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-100-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km100/code_dict",
            fname="cpc_big_ll6k_kmeans_100_tacotron_codes.pt",
            sha256="a343085a83c1acfc96a2bfecf5098b380e860cdbb832cbcbbb0bf1358bd7c932",
        ),
        Checkpoint(
            name="cpc-big-ll6k-kmeans-200-tacotron-codes",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km200/code_dict",
            fname="cpc_big_ll6k_kmeans_200_tacotron_codes.pt",
            sha256="48dae50f9f2d4561c4a84fa09ba748395808d7d3323b1d5754af71669117e2aa",
        ),
    ]

    # Common
    checkpoints += [
        # Vocoder models
        Checkpoint(
            name="waveglow",
            remote_path="https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_standalone.pt",
            fname="waveglow_256channels_standalone.pt",
            sha256="f383c7fd785502fc6a6bffd604fc14cb35d6155cdde30c53faaaeafa8a904dab",
        ),
    ]

    for checkpoint in checkpoints:
        CHECKPOINT_MANAGER.add_checkpoint(checkpoint)


populate_checkpoints()
