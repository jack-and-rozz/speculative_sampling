# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

def label_smoothed_nll_loss_to_best_latent(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # Partly copied from 'criterions.label_smoothed_cross_entropy'.
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    batch_size, num_latent_sampling, seq_len, decoder_embed_dim = lprobs.shape

    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = target.reshape(-1, 1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index).float()
        nll_loss = nll_loss * non_pad_mask 
        smooth_loss = smooth_loss * non_pad_mask 

    nll_loss = nll_loss.reshape(batch_size, num_latent_sampling, seq_len)
    smooth_loss = smooth_loss.reshape(batch_size, num_latent_sampling, seq_len)

    best_latent_indice = torch.argmin(torch.sum(nll_loss, dim=-1), dim=1) # [batch_size]
    expanded_indice = best_latent_indice.view(-1, 1, 1).expand(batch_size, 1, seq_len)
    nll_loss = nll_loss.gather(dim=1, index=expanded_indice).squeeze(1)
    smooth_loss = smooth_loss.gather(dim=1, index=expanded_indice).squeeze(1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss, best_latent_indice

def _linear_anneal(num_updates, max_updates):
    return min([num_updates/max_updates, 1.0])




@register_criterion('cvae_loss')
class CVAECriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kl_annealing_steps = args.kl_annealing_steps
        self.kl_annealing_steps_per_cycle = args.kl_annealing_steps_per_cycle
        self.max_kl_weight = args.max_kl_weight
        self.bow_loss_weight = args.bow_loss_weight
        self.kl_annealing_function = args.kl_annealing_function

        if args.bow_loss_weight > 0:
            tgt_dict = task.tgt_dict
            class_weight = torch.ones(len(tgt_dict))
            class_weight[tgt_dict.pad_index] = 0
            class_weight[tgt_dict.eos_index] = 0
            self.bow_loss_fc = torch.nn.CrossEntropyLoss(
                weight=class_weight, ignore_index=tgt_dict.pad_index)


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        # Modified for CVAE
        parser.add_argument('--kl-annealing-steps', default=0, type=int, 
                            metavar='D',
                            help='Linearly increase the weight to KL-divergence term from 0 to args.max_kl_weight until this step. If zero, the weight is fixed to args.max_kl_weight.')
        parser.add_argument('--kl-annealing-steps-per-cycle', default=0, type=int,
                            help='if zero, cyclical annealing is not used.')

        parser.add_argument('--kl-annealing-function', 
                            default='linear', type=str,
                            choices=['linear'], 
                            help=' ')
        parser.add_argument('--max-kl-weight', default=1.0, type=float,
                            help=' ')

        # fmt: on

    def compute_kl_weight(self, _num_updates):
        max_updates = self.kl_annealing_steps
        if self.kl_annealing_function == 'linear':
            annealing_f = _linear_anneal

        if self.kl_annealing_steps > 0:
            if self.kl_annealing_steps_per_cycle > 0:
                # Cyclical scheduling (https://www.aclweb.org/anthology/N19-1021/).
                num_updates = _num_updates % self.kl_annealing_steps_per_cycle
                weight = annealing_f(num_updates, self.kl_annealing_steps) if num_updates < self.kl_annealing_steps else 1.0
            else:
                # Monotonic (linear) scheduling.
                num_updates = _num_updates
                weight = annealing_f(num_updates, self.kl_annealing_steps)
        else:
            # Constant scheduling.
            weight = 1.0

        return weight * self.max_kl_weight

    def compute_loss(self, model, net_output, sample, reduce=True):
        # TransformerDecoder.forward の中で確率分布とextract_featuresで集めたmodel-specific featuresを返している
        nto = net_output[0]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        multiple_latent = (len(lprobs.shape) == 4)

        if multiple_latent:
            batch_size, num_latent_sampling, seq_len, _ = lprobs.shape
            target = target.unsqueeze(1).expand(batch_size, num_latent_sampling, seq_len)


            loss, nll_loss, best_latent_indice = label_smoothed_nll_loss_to_best_latent(
                lprobs, target, self.eps, ignore_index=self.padding_idx, 
                reduce=reduce
            )
        else:
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = target.reshape(-1, 1)
            loss, nll_loss, = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, 
                reduce=reduce
            )
            best_latent_indice = None
        return loss, nll_loss, best_latent_indice

    def compute_bow_loss(self, model, net_output, sample, 
                         best_latent_indice=None, reduce=True):
        # https://www.aclweb.org/anthology/P17-1061.pdf

        extra = net_output[1]
        bow_logprobs = extra['bow_logprobs']

        if best_latent_indice is not None:
            # When sampling multiple latent variables
            batch_size, _, vocab_size = bow_logprobs.shape
            expanded_indice = best_latent_indice.view(-1, 1, 1).expand(
                batch_size, 1, vocab_size)
            bow_logprobs = bow_logprobs.gather(dim=1, index=expanded_indice).squeeze(1)

        # Taking the sum of logits over tokens included in a target sentence.
        # (This corresponds to log{\prod(logits(w_i))}). 
        pad_idx = model.decoder.dictionary.pad_index
        eos_idx = model.decoder.dictionary.eos_index
        mask = torch.zeros_like(bow_logprobs, dtype=torch.float32).scatter_(1, sample['target'], 1.0)
        mask[:, pad_idx] = 0
        mask[:, eos_idx] = 0
        mask_sum = torch.sum(mask, axis=-1)
        bow_loss = -torch.sum(mask * bow_logprobs, axis=-1)
        if reduce:
            bow_loss = bow_loss.sum()
        return bow_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, best_latent_indice = self.compute_loss(model, net_output, sample, reduce=reduce)


        # KLD is computed at sentence level.
        assert self.args.sentence_avg == True 
        if self.args.sentence_avg:
            sample_size = sample['target'].size(0) 
        else:
            sample_size = sample['ntokens']

        extra = net_output[1]

        # Compute BoW loss and add it to the whole loss.
        if 'bow_logprobs' in extra:
            bow_loss = self.compute_bow_loss(
                model, net_output, sample, 
                best_latent_indice=best_latent_indice,
                reduce=reduce)
            loss += self.bow_loss_weight * bow_loss


        kld = extra['kld']
        prior_std_norm = torch.norm(net_output[1]['prior_std'], dim=-1)

        # Add KLD to log.
        # the weight for KLD depends on the current training steps that Criterion does not have an access to. So KLD is added to the whole loss in "tasks/dialogue.py"
        if 'kld' in extra:
            kld = torch.sum(extra['kld'])
            prior_mean_norm = torch.sum(torch.norm(net_output[1]['prior_mean'], dim=-1))
            post_mean_norm = torch.sum(torch.norm(net_output[1]['post_mean'], dim=-1))
            prior_std_norm = torch.sum(torch.norm(net_output[1]['prior_std'], dim=-1))
            post_std_norm = torch.sum(torch.norm(net_output[1]['post_std'], dim=-1))

        loss_log = utils.item(loss.data + kld.data) if reduce else loss.data + kld.data

        logging_output = {
            'loss': loss_log,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if 'kld' in extra:
            logging_output_latent = {
                'kld' : utils.item(kld.data) if reduce else kld.data,
                'prior_mean_norm': utils.item(prior_mean_norm.data) if reduce else prior_mean_norm.data,
                'prior_std_norm': utils.item(prior_std_norm.data) if reduce else prior_std_norm.data, 
                'post_mean_norm': utils.item(post_mean_norm.data) if reduce else post_mean_norm.data,
                'post_std_norm': utils.item(post_std_norm.data) if reduce else post_std_norm.data, 
            }
            logging_output.update(logging_output_latent)
        if 'bow_logprobs' in extra:
            logging_output['bow_loss'] = utils.item(bow_loss.data) if reduce else bow_loss.data

        # Criterion does not have num_updates to compute kl_weight and thus KLD is simply returned here without being added to the loss. 
        loss = (loss, kld) 
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        aggregated = {
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,

            'kld': sum(log.get('kld', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'kl_weight': logging_outputs[0].get('kl_weight', 0),
            'prior_mu': sum(log.get('prior_mean_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'post_mu': sum(log.get('post_mean_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'prior_std': sum(log.get('prior_std_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'post_std': sum(log.get('post_std_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'bow_loss': sum(log.get('bow_loss', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,

        }
        return aggregated

