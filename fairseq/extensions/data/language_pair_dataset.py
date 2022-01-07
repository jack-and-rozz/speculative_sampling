
import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset, LanguagePairDataset

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            # 'tgt_tokens': prev_output_tokens,
            'prev_output_tokens': prev_output_tokens, # input the golden output for training of CVAE-based models
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDatasetForDialogue(LanguagePairDataset):
    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

class LanguagePairDatasetWithExtraFeatures(LanguagePairDatasetForDialogue):
    # def __init__(self, *args, extra_features={}, extra_feature_dicts={}, **kwargs):
    #     super(LanguagePairDatasetWithExtraFeatures, self).__init__(*args, **kwargs)

    def __init__(self, *args, extra_features={}, extra_feature_dicts={}, **kwargs):
        super(LanguagePairDatasetWithExtraFeatures, self).__init__(*args, **kwargs)
        self.extra_features = extra_features
        self.extra_feature_dicts = extra_feature_dicts

    def __getitem__(self, index):
        item = super(LanguagePairDatasetWithExtraFeatures, self).__getitem__(index)

        for feature_type, features in self.extra_features.items():
            item[feature_type] = features[index]
            if hasattr(self.extra_feature_dicts[feature_type], 'weights'):
                item[feature_type + '_weights'] = self.extra_feature_dicts[feature_type].weights[features[index]]
        return item

    def prefetch(self, indices):
        super().prefetch(indices)
        for feature_type in self.extra_features:
            self.extra_features[feature_type].prefetch()

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        batch = super(LanguagePairDatasetWithExtraFeatures, self).collater(samples)

        for feature_type in self.extra_feature_dicts:
            values = [s[feature_type] for s in samples]
            batch[feature_type] = data_utils.collate_tokens(values, 0, left_pad=False, move_eos_to_beginning=False)

            if hasattr(self.extra_feature_dicts[feature_type], 'weights'):
                weights = [s[feature_type + '_weights'] for s in samples]
                batch[feature_type + '_weights'] = data_utils.collate_tokens(weights, 0, left_pad=False, move_eos_to_beginning=False)

        return batch

