# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

def root_mean_squared_difference(v1, v2, clip_max=0.3):
    diff = v1 - v2
    # rmsd = diff
    mean_squared = torch.mean(torch.mul(diff, diff), -1)
    rmsd = torch.sqrt(mean_squared + 1e-9) # To avoid NaN caused by sqrt(0).
    if clip_max > 0:
        rmsd = torch.clamp(rmsd, 0, clip_max)
    return rmsd

@register_criterion('tcvae_spacefusion_loss')
class TCVAESpaceFusionCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.bow_loss_weight = args.bow_loss_weight
        self.interp_loss_weight = args.interp_loss_weight
        self.fuse_loss_weight = args.fuse_loss_weight
        self.euclidean_distance_clip = args.euclidean_distance_clip

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
        parser.add_argument('--interp-loss-weight', 
                            default=1, type=float, help='alpha')
        parser.add_argument('--fuse-loss-weight', 
                            default=30, type=float, help='beta')
        parser.add_argument('--euclidean-distance-clip', 
                            default=0.3, type=float)


    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        nto = net_output[0]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        loss, nll_loss, = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, 
            reduce=reduce
        )
        return loss, nll_loss

    def compute_fuse_loss(self, encoder_out, reduce=True):
        '''
        *NOTE*
        The fuse_loss is not divided by the batch size to make the scale equal to other losses.
        The reduction method used in Fairseq is summation over examples in a batch and the averaged fuse_loss over batch is computed only in aggregate_logging_outputs().
        '''
        prior_out = encoder_out['prior_out']
        post_out = encoder_out['post_out']
        batch_size = prior_out.shape[0]

        # Make z_s2s[i] and z_AE[i] close.
        fuse1 = root_mean_squared_difference(
            prior_out, post_out, 
            clip_max=self.euclidean_distance_clip
        )

        # Make z_s2s[i] and z_s2s[j] distant.
        fuse2 = torch.sum(root_mean_squared_difference(
            prior_out.unsqueeze(1), 
            prior_out.unsqueeze(0),
            clip_max=self.euclidean_distance_clip
        ), -1) / (batch_size - 1) 

        # Make z_AE[i] and z_AE[j] distant.
        fuse3 = torch.sum(root_mean_squared_difference(
            post_out.unsqueeze(1), 
            post_out.unsqueeze(0),
            clip_max=self.euclidean_distance_clip
        ), -1) / (batch_size - 1)

        fuse_loss = fuse1 - (fuse2 + fuse3)
        if reduce is True:
            fuse_loss = fuse_loss.sum()
        return fuse_loss


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        encoder_out = model.encoder(**sample['net_input'])
        prev_output_tokens = sample['net_input']['prev_output_tokens']

        # z_x -> y
        prior_decoder_out = model.decoder(prev_output_tokens, 
                                          encoder_out=encoder_out, 
                                          latent_variable_type='prior_out')
        # z_y -> y
        post_decoder_out = model.decoder(prev_output_tokens, 
                                         encoder_out=encoder_out, 
                                         latent_variable_type='post_out')

        # u*z_x + (1-u)*z_y -> y
        interp_decoder_out = model.decoder(prev_output_tokens, 
                                           encoder_out=encoder_out, 
                                           latent_variable_type='interp_out')

        prior_loss, prior_nll_loss = self.compute_ce_loss(model, prior_decoder_out, sample, reduce=reduce)
        post_loss, post_nll_loss = self.compute_ce_loss(model, post_decoder_out, sample, reduce=reduce)
        interp_loss, interp_nll_loss = self.compute_ce_loss(model, interp_decoder_out, sample, reduce=reduce)


        # d(x_i, y_i) - d(x_i, x_j) - d(y_i, y_j)
        fuse_loss = self.compute_fuse_loss(encoder_out, reduce=reduce)

        # As T-CVAE optimizes cross-entropy and KLD, cross-entropy loss should be computed at sentence level but not at token level to make the scale of the losses compatible.
        assert self.args.sentence_avg == True
        if self.args.sentence_avg:
            # When args.sentence_avg == True, all losses directly used for optimization are the sum of losses computed at sentence level. This is for a case where other loss is added to the cross-entropy.
            sample_size = sample['target'].size(0)
            ntokens_per_sent = sample['ntokens'] / sample['target'].size(0)

            # The losses are divided by the avg. length of the outputs to make the scales of NLL_loss and other losses equal. They are computed at sentence level.
            prior_loss /= ntokens_per_sent
            prior_nll_loss /= ntokens_per_sent
            post_loss /= ntokens_per_sent
            post_nll_loss /= ntokens_per_sent
            interp_loss /= ntokens_per_sent
            interp_nll_loss /= ntokens_per_sent
        else:
            sample_size = sample['ntokens']

        loss = prior_loss + post_loss + self.interp_loss_weight * interp_loss + self.fuse_loss_weight * fuse_loss



        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']


        prior_mean_norm = torch.sum(torch.norm(encoder_out['prior_mean'], dim=-1))
        prior_std_norm = torch.sum(torch.norm(encoder_out['prior_std'], dim=-1))
        post_mean_norm = torch.sum(torch.norm(encoder_out['post_mean'], dim=-1))
        post_std_norm = torch.sum(torch.norm(encoder_out['post_std'], dim=-1))

        loss_log = utils.item(loss.data) if reduce else loss.data

        logging_output = {
            'loss': loss_log,
            'nll_loss': utils.item(post_nll_loss.data) if reduce else post_nll_loss.data,
            'prior_nll_loss': utils.item(prior_nll_loss.data) if reduce else prior_nll_loss.data,
            'post_nll_loss': utils.item(post_nll_loss.data) if reduce else post_nll_loss.data,
            'interp_nll_loss': utils.item(interp_nll_loss.data) if reduce else interp_nll_loss.data,
            'fuse_loss': utils.item(fuse_loss.data) if reduce else fuse_loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        logging_output_latent = {
            'prior_mean_norm': utils.item(prior_mean_norm.data) if reduce else prior_mean_norm.data,
            'prior_std_norm': utils.item(prior_std_norm.data) if reduce else prior_std_norm.data, 
            'post_mean_norm': utils.item(post_mean_norm.data) if reduce else post_mean_norm.data,
            'post_std_norm': utils.item(post_std_norm.data) if reduce else post_std_norm.data, 
        }
        logging_output.update(logging_output_latent)

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
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,

            'prior_nll_loss': sum(log.get('prior_nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'post_nll_loss': sum(log.get('post_nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'interp_nll_loss': sum(log.get('interp_nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,

            'fuse_loss': sum(log.get('fuse_loss', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'prior_mu': sum(log.get('prior_mean_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'post_mu': sum(log.get('post_mean_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'prior_std': sum(log.get('prior_std_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
            'post_std': sum(log.get('post_std_norm', 0) for log in logging_outputs) / nsentences if nsentences > 0 else 0.,
        }
        return aggregated

