#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from extensions.models.t_cvae import TransformerCVAE

import random
import numpy as np
from icecream import ic
from pprint import pprint

def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def main(args):
    set_random_seed(args)
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble (even if only a single model is used)
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)


    # Initialize generator
    assert isinstance(models[0], TransformerCVAE)
    generator = task.build_latent_generator(args) 

    # If True, a scorer is built intead of generator.
    if args.score_reference:
        save_latents_and_scores(args, task, models, itr, generator, 
                                src_dict, tgt_dict)
    else:
        raise NotImplementedError

def save_latents_and_scores(args, task, models, itr, generator, src_dict, tgt_dict):

    '''
    - batch_idx: the index of a batch sampled from a dataset.
    - example_idx: the index of an example in a batch.
    - data_id: the number of an example in the dataset.
    '''
    input_data = [] # [(data_id, input_text, output_text), ...]
    use_cuda = torch.cuda.is_available() and not args.cpu
    prior_out_path_temp = args.analysis_output_dir + "/%d.prior.latent"
    post_out_path_temp = args.analysis_output_dir + "/%d.post.latent"
    both_out_path_temp = args.analysis_output_dir + "/%d.both.latent"
    prior_score_path_temp = args.analysis_output_dir + "/%d.prior.score"
    post_score_path_temp = args.analysis_output_dir + "/%d.post.score"
    both_score_path_temp = args.analysis_output_dir + "/%d.both.score"
    data_summary_path = args.analysis_output_dir + '/input.txt'
    data_summary_f = open(data_summary_path, 'w')

    for batch_idx, batch in enumerate(itr):
        batch = utils.move_to_cuda(batch) if use_cuda else batch
        if 'net_input' not in batch:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = batch['target'][:, :args.prefix_size]
        bsz = batch['id'].shape[0]

        results = [{
            'data_id': None,
            'prior_score': [], # [num_latent_sampling_per_response]
            'prior_out': [], # [num_latent_sampling_per_response, latent_dim]
            'prior_mean': None, # [latent_dim]
            'prior_std': None, # [latent_dim]
            'post_score': [], # [num_latent_sampling_per_response]
            'post_out': [], # [num_latent_sampling_per_response, latent_dim]
            'post_mean': None, # [latent_dim]
            'post_std': None, # [latent_dim]

        } for _ in range(bsz)] # [batch_size]

        for sampling_cnt in range(args.num_latent_sampling_per_response):

            # Compute scores for LVs sampled from the posterior.
            for m in models:
                m.use_posterior()
            post_hypos = task.inference_step(generator, models, batch, prefix_tokens) #[batch_size, num_ensembled_models]
            if len(post_hypos[0]) > 1:
                raise ValueError("For simplicity, this script does not handle ensembling.")
                exit(1)

            for example_idx, (data_id, hypo) in enumerate(zip(batch['id'].tolist(), post_hypos)):
                hypo = hypo[0]
                results[example_idx]['data_id'] = data_id
                results[example_idx]['post_mean'] = hypo['post_mean'].cpu()
                results[example_idx]['post_std'] = hypo['post_std'].cpu()
                results[example_idx]['post_score'].append(hypo['score'].cpu())
                # results[example_idx]['post_out'].append(hypo['post_out'].cpu())
                results[example_idx]['post_out'].append(hypo['latent_out'].cpu())

        for sampling_cnt in range(args.num_latent_sampling_per_response):
            # Compute scores for LVs sampled from the prior.
            for m in models:
                m.use_prior()
            prior_hypos = task.inference_step(generator, models, batch, prefix_tokens) #[batch_size, num_ensembled_models]

            for example_idx, (data_id, hypo) in enumerate(zip(batch['id'].tolist(), prior_hypos)):
                hypo = hypo[0]
                results[example_idx]['prior_mean'] = hypo['prior_mean'].cpu()
                results[example_idx]['prior_std'] = hypo['prior_std'].cpu()
                results[example_idx]['prior_score'].append(hypo['score'].cpu())
                # results[example_idx]['prior_out'].append(hypo['prior_out'].cpu())
                results[example_idx]['prior_out'].append(hypo['latent_out'].cpu())

        for example_idx, data_id in enumerate(batch['id'].tolist()):
            results[example_idx]['prior_score'] = np.stack(results[example_idx]['prior_score'], axis=0)
            results[example_idx]['post_score'] = np.stack(results[example_idx]['post_score'], axis=0)
            results[example_idx]['prior_out'] = np.stack(results[example_idx]['prior_out'], axis=0)
            results[example_idx]['post_out'] = np.stack(results[example_idx]['post_out'], axis=0)

            src_tokens = utils.strip_pad(batch['net_input']['src_tokens'][example_idx, :], src_dict.pad())
            tgt_tokens = utils.strip_pad(batch['target'][example_idx, :], tgt_dict.pad()).int().cpu()
            src_sent = src_dict.string(src_tokens, args.remove_bpe)
            tgt_sent = tgt_dict.string(tgt_tokens, args.remove_bpe)
            results[example_idx]['src_tokens'] = src_sent
            results[example_idx]['tgt_tokens'] = tgt_sent
            prior_out_path = prior_out_path_temp % (data_id)
            post_out_path = post_out_path_temp % (data_id)
            both_out_path = both_out_path_temp % (data_id)
            prior_score_path = prior_score_path_temp % (data_id)
            post_score_path = post_score_path_temp % (data_id)
            both_score_path = both_score_path_temp % (data_id)
            # prior_path = "%s/%s/%d.score" % (args.model_root, args.analysis_output_dir, data_id)
            np.savetxt(prior_out_path, results[example_idx]['prior_out'])
            np.savetxt(post_out_path, results[example_idx]['post_out'])
            np.savetxt(
                both_out_path, 
                np.concatenate([results[example_idx]['prior_out'],
                                results[example_idx]['post_out']],
                               axis=0
                )
            )

            np.savetxt(prior_score_path, results[example_idx]['prior_score'])
            np.savetxt(post_score_path, results[example_idx]['post_score'])
            np.savetxt(
                both_score_path, 
                np.concatenate([results[example_idx]['prior_score'],
                               results[example_idx]['post_score']],
                               axis=0
                )
            )
            line = '\t'.join([str(results[example_idx]['data_id']), src_sent, tgt_sent])
            print(line, file=data_summary_f)
        print(batch_idx, args.num_batches_for_analysis, len(results), args.max_tokens)
        if args.num_batches_for_analysis and batch_idx >= args.num_batches_for_analysis -1:
            break
    return 

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--analysis-output-dir', required=True)
    parser.add_argument('--num-latent-sampling-per-response', default=50, type=int)
    parser.add_argument('--num-batches-for-analysis', default=1, type=int,
                        help='number of batches used in this scripts (other testing examples are ignored)')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
