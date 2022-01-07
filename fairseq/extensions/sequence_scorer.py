# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys

from fairseq import utils
from icecream import ic

class SequenceScorerForCVAE(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None):
        self.pad = tgt_dict.pad()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None

        # Gather generation probabilities of tokens in the reference and take the average over ensembled models.
        for model in models:
            model.eval()
            decoder_out = model.forward(**net_input)
            attn = decoder_out[1]
            # print(attn)
            # print(attn.keys())
            # print(sample['net_input']['src_tokens'].shape)
            # print(attn['encoder_out'].shape) # [max_seq_len, batch_size, hidden_dims]
            # print(attn['latent_out'].shape)  # [batch_size, hidden_dims]
            # print(attn['prior_mean'].shape)
            # print(attn['prior_std'].shape)
            # exit(1)
            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target
            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        if attn is None or len(models) > 1:
            raise NotImplementedError("Currently this script does not handle ensembled models.")

            # print(attn['latent_out'].shape)  # [batch_size, hidden_dims]
            # print(attn['prior_mean'].shape)
            # print(attn['prior_std'].shape)
        latent_out = attn['latent_out']
        prior_mean = attn['prior_mean']
        prior_std = attn['prior_std']
        post_mean = None
        post_std = None
        if 'post_mean' in attn:
            post_mean = attn['post_mean']
            post_std = attn['post_std']
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i, start_idxs[i]:]
                _, alignment = avg_attn_i.max(dim=0)
            else:
                avg_attn_i = alignment = None
            d = {
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'latent_out': latent_out[i],
                'prior_mean': prior_mean[i],
                'prior_std': prior_std[i],
            }
            if post_mean is not None:
                d['post_mean'] = post_mean[i]
                d['post_std'] = post_std[i]
            hypos.append([d])
        return hypos
