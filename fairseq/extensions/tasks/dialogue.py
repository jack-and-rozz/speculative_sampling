# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os, sys
import torch
from torch.autograd import detect_anomaly

import numpy as np

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    Dictionary
)
from collections import defaultdict

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

from ..criterions.vae_loss import CVAECriterion
from ..data import LanguagePairDatasetForDialogue,  LanguagePairDatasetWithExtraFeatures
from ..data import DictionaryWithInvFreqWeight
from ..sequence_scorer import SequenceScorerForCVAE
# from ..data.dictionary import DictionaryWithInvFreqWeight

def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, 
    max_source_positions, max_target_positions,
    extra_feature_dicts,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    
    extra_feature_datasets = defaultdict(list)

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')
        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        if extra_feature_dicts:
            for i, feature_type in enumerate(extra_feature_dicts):
                extra_feature_datasets[feature_type].append(
                    data_utils.load_indexed_dataset(prefix + feature_type, 
                                                    extra_feature_dicts[feature_type], 
                                                    dataset_impl)
                )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)


    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        extra_feature_datasets = {feature_type: datasets[0] for feature_type, datasets in extra_feature_datasets.items()}

    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        extra_feature_datasets = {feature_type: ConcatDataset(datasets) for feature_type, datasets in extra_feature_datasets.items()}

    if len(extra_feature_datasets.keys()) > 0:
        return LanguagePairDatasetWithExtraFeatures(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            extra_feature_dicts=extra_feature_dicts,
            extra_features=extra_feature_datasets
        )
    else:
        return LanguagePairDatasetForDialogue(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
        )

@register_task('dialogue')
class DialogueTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # Modified
        parser.add_argument('--extra-features', nargs='*', 
                            help="List of files which have the same number of lines as the src and the tgt files. Each file contains extra features including the information of the example's domains, speakers, etc.")
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, extra_feature_dicts={}):
        super(DialogueTask, self).__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        # Modified
        self.extra_feature_dicts = extra_feature_dicts
        self._num_updates = 0

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        # return cls(args, src_dict, tgt_dict)

        if args.extra_features:
            extra_feature_dicts = {feature_type:cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(feature_type)), weight_by_freq=True) for feature_type in args.extra_features}
        else:
            extra_feature_dicts = {}
        for feature_type, feature_dict in extra_feature_dicts.items():
            print('| [{}] dictionary: {} types'.format(feature_type, len(feature_dict)))

        return cls(args, src_dict, tgt_dict, extra_feature_dicts=extra_feature_dicts)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            extra_feature_dicts=self.extra_feature_dicts,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # TODO: add extra-features if exists
        return LanguagePairDatasetForDialogue(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @classmethod
    def load_dictionary(cls, filename, weight_by_freq=False):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if weight_by_freq:
            return DictionaryWithInvFreqWeight.load(filename)
        else:
            return Dictionary.load(filename)

    def update_step(self, num_updates):
        self._num_updates = num_updates


    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train() 
        if isinstance(criterion, CVAECriterion):
            # As criterions do not have num_updates, the annealing rate is computed here.
            losses, sample_size, logging_output = criterion(model, sample)
            smoothed_nll, kld = losses
            kl_weight = criterion.compute_kl_weight(self._num_updates)
            loss = smoothed_nll + kl_weight * kld
            logging_output['kl_weight'] = kl_weight
        else:
            loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval() 
        with torch.no_grad():
            if isinstance(criterion, CVAECriterion):
                losses, sample_size, logging_output = criterion(model, sample)
                kl_weight = criterion.compute_kl_weight(self._num_updates)
                loss = losses[0] + kl_weight * losses[1]
                logging_output['kl_weight'] = kl_weight
            else:
                loss, sample_size, logging_output = criterion(model, sample)

        return loss, sample_size, logging_output



    def build_latent_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorerForCVAE(self.target_dictionary)
        else:
            from .sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )




