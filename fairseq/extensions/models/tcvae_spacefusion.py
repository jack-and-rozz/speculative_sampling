# 
import math, sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from fairseq.modules import LayerNorm
from fairseq.modules import MultiheadAttention

from fairseq import options, utils

from fairseq.models.transformer import (
    TransformerModel, 
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    Embedding, 
    Linear,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture as _base_architecture,
) 

from fairseq.models import (
    FairseqEncoder,
     FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from ..utils import parse_embedding
from .t_cvae import TransformerCVAE, TCVAETransformerEncoder, TCVAETransformerDecoder
# from ..models import TransformerCVAE, TCVAELatent


class TCVAESpaceFusionLatent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_posterior = None
        self.latent_dim = args.decoder_embed_dim

        # In the original code of T-CVAE, only prior-net has another FFN with tanh.
        self.prior_net1 = nn.Linear(args.encoder_embed_dim, 256)

        self.prior_net2 = nn.Linear(256, self.latent_dim, bias=False)
        self.post_net = nn.Linear(args.encoder_embed_dim, self.latent_dim,
                                  bias=False)

        self.prior_attn = torch.nn.MultiheadAttention(
            args.encoder_embed_dim, 
            args.encoder_attention_heads, 
            dropout=args.attention_dropout, 
            bias=False, add_bias_kv=False)

        if args.disable_sharing_prior_post_attn:
            self.post_attn = torch.nn.MultiheadAttention(
                args.encoder_embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                bias=False, add_bias_kv=False)
        else:
            self.post_attn = self.prior_attn

        self.query = nn.Parameter(torch.Tensor(1, 1, args.encoder_embed_dim)) 
        self.reset_parameters()
        self.stddev = args.gaussian_stddev

    def reset_parameters(self):
        xavier_normal_(self.query)

    def forward(self, prior_out, post_out):

        batch_size = prior_out['encoder_out'].shape[1]

        query = self.query.expand(1, batch_size, self.query.shape[-1])

        if prior_out['encoder_padding_mask'] is not None and \
           not prior_out['encoder_padding_mask'].any():
            prior_out['encoder_padding_mask'] = None

        prior_attn_out, _ = self.prior_attn(
            query=query,
            key=prior_out['encoder_out'],
            value=prior_out['encoder_out'],
            key_padding_mask=prior_out['encoder_padding_mask'])
        prior_mean = self.prior_net2(torch.tanh(
            self.prior_net1(prior_attn_out[0])))
        device = prior_mean.device

        prior_std = torch.ones((batch_size, self.latent_dim)) * self.stddev

        prior_eps = torch.randn((batch_size, self.latent_dim), 
                                device=device)

        prior_z = prior_mean + self.stddev * prior_eps

        if post_out is not None:
            if post_out['encoder_padding_mask'] is not None and \
               not post_out['encoder_padding_mask'].any():
                post_out['encoder_padding_mask'] = None
            post_attn_out, _ = self.post_attn(
                query=query,
                key=post_out['encoder_out'],
                value=post_out['encoder_out'],
                key_padding_mask=post_out['encoder_padding_mask'])

            post_mean = self.post_net(post_attn_out[0])
            post_std = prior_std
            post_eps = torch.randn((batch_size, self.latent_dim), 
                                   device=device)
            post_z = post_mean + self.stddev * post_eps
            # Interpolate S2S and AE representations by weights randomly sampled from uniform distribution.
            interp_weights = torch.rand((batch_size, 1), device=device)
            interp_z = interp_weights * prior_z + (1 - interp_weights) * post_z

        assert (not self.use_posterior) or (post_out is not None)
        if self.use_posterior is None:
            use_posterior = post_out is not None
        else:
            use_posterior = self.use_posterior

        if use_posterior: 
            latent_out = {
                'latent_out': post_z,
                'interp_out': interp_z,
                'prior_out': prior_z,
                'prior_mean': prior_mean, 
                'prior_std': prior_std,
                'post_out': post_z,
                'post_mean': post_mean,
                'post_std': post_std,
            }
        else:
            latent_out = {
                'latent_out': prior_z,
                'prior_out': prior_z,
                'prior_mean': prior_mean,
                'prior_std': prior_std,
            }
        return latent_out


@register_model('tcvae_spacefusion')
class TCVAESpaceFusion(TransformerCVAE):
    @staticmethod
    def add_args(parser):
        # TransformerModel.add_args(parser)
        TransformerCVAE.add_args(parser)

        parser.add_argument('--gaussian-stddev', 
                            default=0.1, type=float,
                            help="The default stddev was copied from the original code by [Gao+, NAACL'19].")

    def __init__(self, encoder, decoder, latent, 
                 extra_feature_dicts={},
                 num_latent_sampling=1):
        # super().__init__(encoder, decoder)
        TransformerModel.__init__(self, encoder, decoder)
        self.extra_feature_dicts = extra_feature_dicts
        self.latent = latent
        self.num_latent_sampling=num_latent_sampling # currently not used

    @classmethod
    def build_latent(cls, args):
        return TCVAESpaceFusionLatent(args)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, latent, tgt_dict):
        return TCVAESpaceFusionEncoder(args, src_dict, embed_tokens, latent, tgt_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, encoder):
        if args.disable_sharing_decoder:
            encoder = None 
        return TCVAESpaceFusionDecoder(args, tgt_dict, embed_tokens, encoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, 
                                   prev_output_tokens=prev_output_tokens, 
                                   **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, 
                                   **kwargs)
        return decoder_out

    def use_posterior(self):
        self.encoder.latent.use_posterior = True

    def use_prior(self):
        self.encoder.latent.use_posterior = False

@register_model_architecture('tcvae_spacefusion', 'tcvae_spacefusion')
def base_architecture(args):
    _base_architecture(args)
    args.disable_sharing_decoder = getattr(
        args, 'disable_sharing_decoder', False)
    args.disable_sharing_prior_post_attn = getattr(
        args, 'disable_sharing_prior_post_attn', False)
    # args.enable_joined_encoding = getattr(
    #     args, 'enable_joined_encoding', False)

    args.gaussian_stddev = getattr(
        args, 'gaussian-stddev', 0.1)



class TCVAESpaceFusionEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, latent, tgt_dict):
        super().__init__(args, dictionary, embed_tokens)
        self.latent = latent
        self.tgt_dict = tgt_dict
        self.left_pad_source = args.left_pad_source
        self.left_pad_target = args.left_pad_target

    # def forward(self, src_tokens, src_lengths, tgt_tokens=None):
    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        tgt_tokens = prev_output_tokens

        src_encoder_out = super().forward(src_tokens, src_lengths)
        prior_encoder_out = src_encoder_out

        if tgt_tokens is None: # At inference step.
            post_encoder_out = None
        else:
            if self.left_pad_source and (not self.left_pad_target):
                tgt_tokens, tgt_lengths = self.shift_tgt_tokens(tgt_tokens) 
            else:
                raise NotImplementedError("")

            post_encoder_out = super().forward(tgt_tokens, 
                                               src_lengths=tgt_lengths)

        latent_out = self.latent(prior_encoder_out, post_encoder_out)

        src_encoder_out.update(latent_out)
        return src_encoder_out

    def shift_tgt_tokens(self, prev_output_tokens):
        # Shift the positions of paddings.
        device = prev_output_tokens.device
        encoder_pad_idx = self.dictionary.pad()
        decoder_pad_idx = self.tgt_dict.pad()
        tgt_lengths = prev_output_tokens.shape[1] - ((prev_output_tokens - decoder_pad_idx) == 0).sum(dim=1)
        max_length = torch.max(tgt_lengths)
        batch_size = prev_output_tokens.shape[0]
        tgt_tokens = torch.ones([batch_size, max_length], dtype=torch.int64, 
                                device=device) * encoder_pad_idx
        if (self.left_pad_source and self.left_pad_target) or \
           ((not self.left_pad_source) and (not self.left_pad_target)):
            return prev_output_tokens
        elif self.left_pad_source and (not self.left_pad_target):
            for i in range(batch_size):
                lt = tgt_lengths[i]
                offset = max_length - lt
                tgt_tok_ids = prev_output_tokens[i, :lt]
                tgt_tokens[i, offset:] = tgt_tok_ids
        elif (not self.left_pad_source) and self.left_pad_target:
            for i in range(batch_size):
                lt = tgt_lengths[i]
                offset = max_length - lt
                tgt_tok_ids = prev_output_tokens[i, offset:]
                tgt_tokens[i, :lt] = tgt_tok_ids
        return tgt_tokens, tgt_lengths

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out = super().reorder_encoder_out(encoder_out, new_order)

        latent_keys = ['latent_out', 'prior_out', 'post_out', 'interp_out'] # need to apply index_select also to stats of latent variables? (kld, mean/std, etc.)
        for k in latent_keys:
            if k in encoder_out:
                encoder_out[k] = encoder_out[k].index_select(0, new_order)
        return encoder_out

class TCVAESpaceFusionDecoder(TCVAETransformerDecoder):
    def forward(self, prev_output_tokens, encoder_out=None, 
                incremental_state=None, latent_variable_type=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        if latent_variable_type is None:
            latent_variable_type = 'latent_out'

        if encoder_out is not None:
            for k, v in encoder_out.items():
                if k in extra:
                    raise ValueError("The name of values fed from encoder to decoder must not be duplicated with those in decoder.")
                extra[k] = v

            z = encoder_out[latent_variable_type]
            batch_size, seq_len, decoder_embed_dim = x.shape
            # [batch_size, embed_dim]
            expanded_z = z.unsqueeze(1).expand(
                batch_size, seq_len, decoder_embed_dim)
            x = torch.cat([x, expanded_z], dim=-1) 
            x = self.latent_fc(x)
            x = self.activation_fn(x)

        if self.bow_loss_fc is not None:
            bow_scores = self.bow_loss_fc(z)
            bow_logprobs = F.log_softmax(bow_scores, dim=-1)
            extra['bow_logprobs'] = bow_logprobs # [batch_size, vocab_size]

        x = self.output_layer(x)

        return x, extra


