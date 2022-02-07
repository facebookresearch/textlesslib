# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from fairseq import hub_utils, utils
from fairseq.hub_utils import GeneratorHubInterface


class UnitLanguageModelSampler(GeneratorHubInterface):
    """
    A simple PyTorch interface for ULM
    """

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.model = self.models[0]
        self.model.eval()

    def encode(self, unit_str):
        tokens = self.task.source_dictionary.encode_line(
            unit_str, add_if_not_exist=False
        ).long()
        return tokens

    def get_prefix_size(self):
        return self.cfg.generation.prefix_size

    def post_process_predictions(self, src_tokens, hypos):
        src_tokens = utils.strip_pad(src_tokens, self.tgt_dict.pad())
        src_str = None
        if self.task.source_dictionary is not None:
            src_str = self.task.source_dictionary.string(
                src_tokens, self.cfg.common_eval.post_process
            )
        return [
            utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=self.align_dict,
                tgt_dict=self.tgt_dict,
                remove_bpe=self.cfg.common_eval.post_process,
            )[1]
            for hypo in hypos
        ]

    def sample(
        self, sentences: tp.List[str], beam: int = 1, verbose: bool = False, **kwargs
    ):
        hypotheses = self.sample_top_hypotheses(sentences, beam, verbose, **kwargs)
        return [hypos[0] for hypos in hypotheses]

    def sample_top_hypotheses(
        self, sentences: tp.List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> tp.List[str]:
        if isinstance(sentences, str):
            return self.sample_top_hypotheses(
                [sentences], beam=beam, verbose=verbose, **kwargs
            )[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)

        return [
            self.post_process_predictions(src_tokens, hypos)
            for src_tokens, hypos in zip(tokenized_sentences, batched_hypos)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="checkpoint_best.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=None,
            bpe=None,
            load_checkpoint_heads=True,
            sample_break_mode="eos",
            **kwargs,
        )
        return cls(x["args"], x["task"], x["models"])
