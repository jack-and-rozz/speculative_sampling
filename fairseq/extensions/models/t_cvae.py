# 
import math, sys

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

class TCVAELatent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_posterior = None
        self.latent_dim = args.decoder_embed_dim
        self.enable_mean_sampled_latent = args.enable_mean_sampled_latent
        # In the original code, only prior-net has another FFN with tanh for some reason.
        self.prior_net1 = nn.Linear(args.encoder_embed_dim, 256)

        self.prior_net2 = nn.Linear(256, self.latent_dim * 2, bias=False)
        self.post_net = nn.Linear(args.encoder_embed_dim, self.latent_dim * 2,
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
        self.enable_stopping_klgrad_to_posterior = args.enable_stopping_klgrad_to_posterior
        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.query)

    def forward(self, prior_out, post_out, num_latent_sampling=1):
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
        prior_meanlogvar = self.prior_net2(torch.tanh(
            self.prior_net1(prior_attn_out[0])))
        prior_mean, prior_logvar = torch.split(prior_meanlogvar,
                                               self.latent_dim, dim=-1)
        prior_std = torch.exp(0.5 * prior_logvar)

        # Sample variables from the posterior distribution too when the gold response is given.
        if post_out is not None:
            if post_out['encoder_padding_mask'] is not None and \
               not post_out['encoder_padding_mask'].any():
                post_out['encoder_padding_mask'] = None
            post_attn_out, _ = self.post_attn(
                query=query,
                key=post_out['encoder_out'],
                value=post_out['encoder_out'],
                key_padding_mask=post_out['encoder_padding_mask'])

            post_mean, post_logvar = torch.split(self.post_net(post_attn_out[0]),
                                                 self.latent_dim, dim=-1)

            post_std = torch.exp(0.5 * post_logvar)
            kld = self.gaussian_kld(post_mean, post_logvar, 
                                    prior_mean, prior_logvar, 
                                    self.enable_stopping_klgrad_to_posterior)

        assert (not self.use_posterior) or (post_out is not None)

        if self.use_posterior is None:
            use_posterior = post_out is not None
        else:
            use_posterior = self.use_posterior

        if use_posterior: 
            prior_z = self.sample(
                batch_size, prior_mean, prior_std,
                num_latent_sampling=num_latent_sampling)

            post_z = self.sample(
                batch_size, post_mean, post_std,
                num_latent_sampling=num_latent_sampling)

            latent_out = {
                'latent_out': post_z,
                'prior_out': prior_z,
                'prior_mean': prior_mean, 
                'prior_logvar': prior_logvar,
                'prior_std': prior_std,
                'post_out': post_z,
                'post_mean': post_mean,
                'post_logvar': post_logvar,
                'post_std': post_std,
                'kld': kld, 
            }
        else:
            prior_z = self.sample(
                batch_size, prior_mean, prior_std,
                num_latent_sampling=num_latent_sampling)

            latent_out = {
                'latent_out': prior_z,
                'prior_out': prior_z,
                'prior_mean': prior_mean,
                'prior_logvar': prior_logvar,
                'prior_std': prior_std,

            }
        return latent_out
        # return z, extra_z

    def sample(self, batch_size, z_mean, z_std, num_latent_sampling=1):
        # if num_latent_sampling > 1:
        if num_latent_sampling > 1 and self.training:
            epsilon = torch.randn((batch_size, num_latent_sampling, self.latent_dim), device=z_mean.device)
            if z_std.dtype == torch.float16:
                epsilon = epsilon.half()

            z = epsilon * z_std.unsqueeze(1) + z_mean.unsqueeze(1) # [batch_size, num_latent, latent_dim]

            if self.enable_mean_sampled_latent:
                z = torch.mean(z, 1)
        else:
            epsilon = torch.randn((batch_size, self.latent_dim), 
                                  device=z_mean.device)
            if z_std.dtype == torch.float16:
                epsilon = epsilon.half()
            z = epsilon * z_std + z_mean # [batch_size, latent_dim]
        return z

    @staticmethod
    def gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar, 
                     stop_klgrad_to_posterior):
        if stop_klgrad_to_posterior is True:
            # Gradients from the KL-divergence loss are not propageted to the posterior.
            post_mu_ = post_mu.detach()
            post_logvar_ = post_logvar.detach()
        else:
            post_mu_ = post_mu
            post_logvar_ = post_logvar

        kld = -0.5 * torch.sum(1 + (post_logvar_ - prior_logvar)
                               - torch.div(torch.pow(prior_mu - post_mu_, 2), 
                                           torch.exp(prior_logvar))
                               - torch.div(torch.exp(post_logvar_), 
                                           torch.exp(prior_logvar)), dim=-1)
        return kld


@register_model('t_cvae')
class TransformerCVAE(TransformerModel):
    @staticmethod
    def add_args(parser):
        # """Add model-specific arguments to the parser."""
        # # fmt: off
        # parser.add_argument('--activation-fn',
        #                     choices=utils.get_available_activation_fns(),
        #                     help='activation function to use')
        # parser.add_argument('--dropout', type=float, metavar='D',
        #                     help='dropout probability')
        # parser.add_argument('--attention-dropout', type=float, metavar='D',
        #                     help='dropout probability for attention weights')
        # parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
        #                     help='dropout probability after activation in FFN.')
        # parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
        #                     help='path to pre-trained encoder embedding')
        # parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
        #                     help='encoder embedding dimension')
        # parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
        #                     help='encoder embedding dimension for FFN')
        # parser.add_argument('--encoder-layers', type=int, metavar='N',
        #                     help='num encoder layers')
        # parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
        #                     help='num encoder attention heads')
        # parser.add_argument('--encoder-normalize-before', action='store_true',
        #                     help='apply layernorm before each encoder block')
        # parser.add_argument('--encoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the encoder')
        # parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
        #                     help='path to pre-trained decoder embedding')
        # parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension')
        # parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension for FFN')
        # parser.add_argument('--decoder-layers', type=int, metavar='N',
        #                     help='num decoder layers')
        # parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
        #                     help='num decoder attention heads')
        # parser.add_argument('--decoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the decoder')
        # parser.add_argument('--decoder-normalize-before', action='store_true',
        #                     help='apply layernorm before each decoder block')
        # parser.add_argument('--share-decoder-input-output-embed', action='store_true',
        #                     help='share decoder input and output embeddings')
        # parser.add_argument('--share-all-embeddings', action='store_true',
        #                     help='share encoder, decoder and output embeddings'
        #                          ' (requires shared dictionary and embed dim)')
        # parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
        #                     help='if set, disables positional embeddings (outside self attention)')
        # parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
        #                     help='comma separated list of adaptive softmax cutoff points. '
        #                          'Must be used with adaptive_loss criterion'),
        # parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
        #                     help='sets adaptive softmax dropout for the tail projections')
        TransformerModel.add_args(parser)

        parser.add_argument('--disable-sharing-decoder', action='store_true',
                            help='if set, the encoder and decoder have independent parameters each other.')
        parser.add_argument('--disable-sharing-prior-post-attn', action='store_true', help='if set, the model has independent parameters for inducing prior and posterior distributions by attention. Otherwise they are shared.')

        parser.add_argument('--enable-joined-encoding', action='store_true',
                            help='if set, the input and the output are independently encoded by the encoder, and then their representations are concatenated. Otherwise, they are encoded as a concatenated input. As this model is mainly for dialogue or translation, we do not set this argment as default differently from the original T-CVAE (Wang+, IJCAI 2019).')

        parser.add_argument('--bow-loss-weight', 
                            default=0.0, type=float,
                            help='the weight to bag-of-word loss. If equal or less than zero, BoW loss is not used. This is an argument to use BoW loss (Zhao+, ACL 2017).')
        parser.add_argument('--num-latent-sampling', type=int, metavar='N',
                            default=1, help='Number of sampled latent variables in training. This argument is for our speculative sampling.')

        parser.add_argument('--enable-mean-sampled-latent', action='store_true',
                            help='if set, it samples args.num_latent_sampling latent variables and use their average. This is an argument to use Monte Carlo method in training proposed by (Kruengkrai, ACL 2019)')


        parser.add_argument('--enable-stopping-klgrad-to-posterior', 
                            action='store_true',
                            help='if set, the gradients from the KL-divergence loss are not propagated to parameters of the posterior. This is for preliminary experiments to examine the problem that both prior and posterior distributions get closer to each other.')
        parser.add_argument('--no-encoder-attn', 
                            action='store_true',
                            help='if set, the attention mechanism from the decoder to the encoder is disabled.')

        # fmt: on

    def __init__(self, encoder, decoder, latent, 
                 extra_feature_dicts={},
                 num_latent_sampling=1, enable_mean_sampled_latent=False):
        super().__init__(encoder, decoder)
        self.extra_feature_dicts = extra_feature_dicts
        self.latent = latent
        self.num_latent_sampling=num_latent_sampling

    @classmethod
    def build_model(cls, args, task):

        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            # if provided, load from preloaded dictionaries
            if path:
                # embed_dict = utils.parse_embedding(path)
                embed_dict = parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)

            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path,
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if not args.disable_sharing_decoder:
            assert args.share_all_embeddings
            assert args.encoder_attention_heads == args.decoder_attention_heads
            assert args.encoder_embed_dim == args.decoder_embed_dim
            assert args.encoder_layers == args.decoder_layers
            assert args.encoder_layers == args.decoder_layers
            assert args.encoder_ffn_embed_dim == args.decoder_ffn_embed_dim

        latent = cls.build_latent(args)

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens,
                                    latent, tgt_dict)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, 
                                    encoder)

        if args.enable_mean_sampled_latent is True:
            return cls(encoder, decoder, latent, 
                       extra_feature_dicts=task.extra_feature_dicts,
                       num_latent_sampling=args.num_latent_sampling,
                       enable_mean_sampled_latent=args.enable_mean_sampled_latent)
        else:
            return cls(encoder, decoder, latent, 
                       extra_feature_dicts=task.extra_feature_dicts,
                       num_latent_sampling=args.num_latent_sampling)
        
    @classmethod
    def build_latent(cls, args):
        return TCVAELatent(args)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, latent, tgt_dict):
        return TCVAETransformerEncoder(args, src_dict, embed_tokens, latent, tgt_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, encoder):
        if args.disable_sharing_decoder:
            encoder = None 
        return TCVAETransformerDecoder(args, tgt_dict, embed_tokens, encoder, 
                                       no_encoder_attn=args.no_encoder_attn,
                                       bow_loss_weight=args.bow_loss_weight,)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, 
                                   prev_output_tokens=prev_output_tokens, 
                                   num_latent_sampling=self.num_latent_sampling,
                                   **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def use_posterior(self):
        self.encoder.latent.use_posterior = True

    def use_prior(self):
        self.encoder.latent.use_posterior = False



@register_model_architecture('t_cvae', 't_cvae')
def base_architecture(args):
    _base_architecture(args)
    args.disable_sharing_decoder = getattr(
        args, 'disable_sharing_decoder', False)
    args.disable_sharing_prior_post_attn = getattr(
        args, 'disable_sharing_prior_post_attn', False)
    args.enable_joined_encoding = getattr(
        args, 'enable_joined_encoding', False)
    args.bow_loss_weight = getattr(args, 'bow_loss_weight', 0.)
    args.num_latent_sampling = getattr(args, 'num_latent_sampling', 1)
    args.enable_stopping_klgrad_to_posterior = getattr(
        args, 'enable_stopping_klgrad_to_posterior', False)

    args.enable_mean_sampled_latent = getattr(args, 'enable_mean_sampled_latent', 
                                              False)
    args.no_encoder_attn = getattr(args, 'no_encoder_attn', False)


def replace_decoder_parameters(encoder_layer, decoder_layer):
    e_params = dict(encoder_layer.named_parameters())
    d_params = dict(decoder_layer.named_parameters())

    # print('<encoder>')
    # for k, v in dict(encoder_layer.named_parameters()).items():
    #     print(k, v.shape)
    # print('<decoder>')
    # for k, v in dict(decoder_layer.named_parameters()).items():
    #     print(k, v.shape)

    decoder_layer.activation_fn = encoder_layer.activation_fn

    # Share parameters of self-attention layer.
    decoder_layer.self_attn.in_proj_weight = encoder_layer.self_attn.in_proj_weight
    if hasattr(decoder_layer.self_attn, 'in_proj_bias'):
        decoder_layer.self_attn.in_proj_bias = encoder_layer.self_attn.in_proj_bias

    decoder_layer.self_attn.out_proj.weight = encoder_layer.self_attn.out_proj.weight
    decoder_layer.self_attn.out_proj.bias = encoder_layer.self_attn.out_proj.bias

    decoder_layer.encoder_attn.in_proj_weight = encoder_layer.self_attn.in_proj_weight
    if hasattr(decoder_layer.encoder_attn, 'in_proj_bias'):
        decoder_layer.encoder_attn.in_proj_bias = encoder_layer.self_attn.in_proj_bias
    decoder_layer.encoder_attn.out_proj.weight = encoder_layer.self_attn.out_proj.weight
    decoder_layer.encoder_attn.out_proj.bias = encoder_layer.self_attn.out_proj.bias

    decoder_layer.fc1 = encoder_layer.fc1
    decoder_layer.fc2 = encoder_layer.fc2

    # e_params = dict(encoder_layer.named_parameters())
    # d_params = dict(decoder_layer.named_parameters())
    # for k, v in dict(encoder_layer.named_parameters()).items():
    #     print(k, e_params[k] is d_params[k])
    # exit(1)


class TCVAETransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, latent, tgt_dict):
        super().__init__(args, dictionary, embed_tokens)
        self.latent = latent
        self.tgt_dict = tgt_dict
        self.left_pad_source = args.left_pad_source
        self.left_pad_target = args.left_pad_target
        self.joined_encoding = args.enable_joined_encoding

    # def forward(self, src_tokens, src_lengths, tgt_tokens=None):
    def forward(self, src_tokens, src_lengths, prev_output_tokens=None, 
                num_latent_sampling=1, use_posterior=None):
        tgt_tokens = prev_output_tokens

        src_encoder_out = super().forward(src_tokens, src_lengths)
        prior_encoder_out = src_encoder_out

        if tgt_tokens is None: # At inference step.
            post_encoder_out = None
        elif self.joined_encoding: # When the combination of the input and output is jointly encoded.
            joined_tokens, joined_lengths = self.join_src_and_tgt(
                src_tokens, src_lengths, tgt_tokens)
            post_encoder_out = super().forward(joined_tokens, 
                                               src_lengths=joined_lengths)
        elif (not self.joined_encoding): # When the input and output senenteces are independently encoded.

            if self.left_pad_source and (not self.left_pad_target):
                tgt_tokens, tgt_lengths = self.shift_tgt_tokens(tgt_tokens) 
            else:
                raise NotImplementedError("")

            tgt_encoder_out = super().forward(tgt_tokens, 
                                              src_lengths=tgt_lengths)
            post_encoder_out = self.concat_encoder_outs(
                src_tokens, tgt_tokens, src_encoder_out, tgt_encoder_out)

        latent_out = self.latent(prior_encoder_out, post_encoder_out, 
                                 num_latent_sampling=num_latent_sampling)
        src_encoder_out.update(latent_out)
        return src_encoder_out

    def concat_encoder_outs(self, src_tokens, tgt_tokens, 
                            encoder_out, tgt_encoder_out):
        if encoder_out['encoder_padding_mask'] is None:
            encoder_out['encoder_padding_mask'] = torch.zeros_like(
                src_tokens, dtype=torch.bool, 
                device=src_tokens.device)
        if tgt_encoder_out['encoder_padding_mask'] is None:
            tgt_encoder_out['encoder_padding_mask'] = torch.zeros_like(
                tgt_tokens, dtype=torch.bool, 
                device=tgt_tokens.device)

        joined_encoder_out = {
            'encoder_out': torch.cat([
                encoder_out['encoder_out'],
                tgt_encoder_out['encoder_out'],
            ], dim=0),
            'encoder_padding_mask': torch.cat([
                encoder_out['encoder_padding_mask'],
                tgt_encoder_out['encoder_padding_mask'],
            ], dim=-1),
        }
        return joined_encoder_out

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

    def join_src_and_tgt(self, src_tokens, src_lengths, prev_output_tokens):
        # Create joined sequences of ids (src + tgt) to compute posterior distributions.
        device = src_tokens.device
        batch_size = src_tokens.shape[0]
        # eot_idx = self.decoder.dictionary.indices[self.turn_delimiter]

        encoder_pad_idx = self.encoder.dictionary.pad()
        decoder_pad_idx = self.decoder.dictionary.pad()

        tgt_lengths = prev_output_tokens.shape[1] - ((prev_output_tokens - decoder_pad_idx) == 0).sum(dim=1)

        joined_lengths = src_lengths + tgt_lengths
        max_length = torch.max(joined_lengths)

        joined_tokens = torch.ones([batch_size, max_length], dtype=torch.int64, 
                                   device=device) * encoder_pad_idx

        for i in range(batch_size):
            ls = src_lengths[i]
            lt = tgt_lengths[i]
            tgt_tok_ids = prev_output_tokens[i, prev_output_tokens.shape[1]-lt:] if self.left_pad_target else prev_output_tokens[i, :lt]

            if self.left_pad_source:
                src_tok_ids = src_tokens[i, src_tokens.shape[1]-ls:]
                offset = max_length - joined_lengths[i]
                joined_tokens[i, offset:offset+ls] = src_tok_ids
                joined_tokens[i, offset+ls:offset+ls+lt] = tgt_tok_ids
            else:
                src_tok_ids = src_tokens[i, :ls]
                joined_tokens[i, :ls] = src_tok_ids
                joined_tokens[i, ls:ls+lt] = tgt_tok_ids
        return joined_tokens, joined_lengths


    def reorder_encoder_out(self, encoder_out, new_order):
        # For generation
        encoder_out = super().reorder_encoder_out(encoder_out, new_order)

        latent_keys = ['latent_out'] # need to apply index_select also to stats of latent variables? (kld, mean/std, etc.)
        for k in latent_keys:
            encoder_out[k] = encoder_out[k].index_select(0, new_order)
        return encoder_out

class TCVAETransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, encoder,
                 no_encoder_attn=False, bow_loss_weight=0):
        # super(TransformerDecoder).__init__()
        FairseqIncrementalDecoder.__init__(self, dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn, 
                                    add_bias_kv=False,
                                    add_zero_attn=False)
            for i in range(args.decoder_layers)
        ])

        if encoder is not None:
            for i in range(len(self.layers)):
                replace_decoder_parameters(encoder.layers[i], self.layers[i])

        self.adaptive_softmax = None
        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None


        # To combine latent variable with decoder's output before output layer.
        self.latent_fc = nn.Linear(args.decoder_embed_dim * 2, 
                                   args.decoder_embed_dim, bias=False)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )

        # To compute BoW loss.
        if bow_loss_weight > 0:
            self.bow_loss_fc = nn.Linear(args.decoder_embed_dim, 
                                         len(dictionary)) 
        else:
            self.bow_loss_fc = None

    def forward(self, prev_output_tokens, encoder_out=None, 
                incremental_state=None, **unused):
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

        if encoder_out is not None:
            for k, v in encoder_out.items():
                if k in extra:
                    raise ValueError("The name of values fed from encoder to decoder must not be duplicated with those in decoder.")
                extra[k] = v

            z = encoder_out['latent_out']
            batch_size, seq_len, decoder_embed_dim = x.shape
            if len(z.shape) == 3: # [batch_size, num_latent, embed_dim]
                num_latent = z.shape[1]
                expanded_z = z.unsqueeze(2).expand(
                    batch_size, num_latent, seq_len, decoder_embed_dim)
                expanded_x = x.unsqueeze(1).expand(
                     batch_size, num_latent, seq_len, decoder_embed_dim)
                x = torch.cat([expanded_x, expanded_z], dim=-1) 
            else: # [batch_size, embed_dim]

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


    # def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
    #     """
    #     Similar to *forward* but only return features.

    #     Returns:
    #         tuple:
    #             - the decoder's features of shape `(batch, tgt_len, embed_dim)`
    #             - a dictionary with any model-specific outputs
    #     """
    #     # embed positions
    #     positions = self.embed_positions(
    #         prev_output_tokens,
    #         incremental_state=incremental_state,
    #     ) if self.embed_positions is not None else None

    #     if incremental_state is not None:
    #         prev_output_tokens = prev_output_tokens[:, -1:]
    #         if positions is not None:
    #             positions = positions[:, -1:]

    #     # embed tokens and positions
    #     x = self.embed_scale * self.embed_tokens(prev_output_tokens)

    #     if self.project_in_dim is not None:
    #         x = self.project_in_dim(x)

    #     if positions is not None:
    #         x += positions
    #     x = F.dropout(x, p=self.dropout, training=self.training)

    #     # B x T x C -> T x B x C
    #     x = x.transpose(0, 1)
    #     attn = None

    #     inner_states = [x]
    #     # decoder layers
    #     for layer in self.layers:
    #         x, attn = layer(
    #             x,
    #             encoder_out['encoder_out'] if encoder_out is not None else None,
    #             encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
    #             incremental_state,
    #             self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
    #         )
    #         inner_states.append(x)

    #     if self.layer_norm:
    #         x = self.layer_norm(x)

    #     # T x B x C -> B x T x C
    #     x = x.transpose(0, 1)

    #     if self.project_out_dim is not None:
    #         x = self.project_out_dim(x)

    #     return x, {'attn': attn, 'inner_states': inner_states}
