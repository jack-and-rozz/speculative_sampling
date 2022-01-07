# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

import sys, os, random
import cgi, ssl, json, urllib
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

from pprint import pprint
sys.path.append(os.getcwd())
from common import dotDict

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    # tokens = [
    #     task.source_dictionary.encode_line(
    #         encode_fn(src_str), add_if_not_exist=False
    #     ).long()
    #     for src_str in lines
    # ]
    tokens = [
        task.source_dictionary.encode_line(
            src_str,
            line_tokenizer=encode_fn, 
            add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], 
            src_lengths=batch['net_input']['src_lengths'],
        )


def build_tokenizer(args):
    if args.tokenizer_type == 'none':
        encode_f = lambda x: x.strip()
        decode_f = lambda x: ''.join(x.split())
    elif args.tokenizer_type == 'mecab':
        import MeCab
        tagger = MeCab.Tagger("-Owakati")
        encode_f = lambda x: tagger.parse(x.strip())
        decode_f = lambda x: ''.join(x.split())

    tokenizer = dotDict({
        'encode': encode_f,
        'decode': decode_f,
    })
    return tokenizer

def build_bpe(args):
    import sentencepiece as spm
    src_sp = spm.SentencePieceProcessor()
    tgt_sp = spm.SentencePieceProcessor()
    src_sp.Load(args.src_sentencepiece)
    tgt_sp.Load(args.tgt_sentencepiece)
    encode_f = lambda x: src_sp.EncodeAsPieces(x.strip())
    decode_f = lambda x: tgt_sp.DecodePieces(x)
    bpe = dotDict({
        'encode': encode_f,
        'decode': decode_f,
    })
    return bpe


# copied from fairseq/interactive.py
def get_generate_func(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

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

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    # tokenizer = encoders.build_tokenizer(args)
    # bpe = encoders.build_bpe(args)
    tokenizer = build_tokenizer(args)
    bpe = build_bpe(args)

    def encode_fn(x: str) -> list:
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x: str) -> str:
        if bpe is not None:
            x = bpe.decode(x.split())
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)

    def _generate_one(inputs:list) -> list:
        """
        <Args>
        - inputs: a list of strings.
        <Return>
        a list of dictionaries.
        """
        start_id = 0 

        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # Return a summary dict
        res = []
        for id, src_ids, hypos in sorted(results, key=lambda x: x[0]):
            src_str = src_dict.string(src_ids, args.remove_bpe)
            res_dict = {}
            res_dict['input'] = inputs[id]
            res_dict['input_tokens'] = src_str
            res_dict['input_ids'] = src_ids.tolist()
            res_dict['output'] = []
            res_dict['output_tokens'] = []
            res_dict['output_ids'] = []
            res_dict['alignment'] = []
            res_dict['attention'] = []
            res_dict['positional_scores'] = []
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                res_dict['output'].append(decode_fn(hypo_str))
                res_dict['output_tokens'].append(hypo_str.split())
                res_dict['output_ids'].append(hypo_tokens.tolist())
                res_dict['alignment'].append(alignment.tolist())
                res_dict['attention'].append(hypo['attention'].tolist())
                res_dict['positional_scores'].append(hypo['positional_scores'].tolist())
            res.append(res_dict)
        return res

    return _generate_one




class DialogueRequestHandler(BaseHTTPRequestHandler):
    # def do_POST(self):
    #     content_len = int(self.headers.get("content-length"))
    #     req_body = self.rfile.read(content_len).decode()
    #     print(req_body)
    #     print(type(req_body))
    #     print(urllib.parse.unquote(req_body))
    #     d = json.loads(req_body)
    #     print(type(d), d)
    #     if req_body:
    #         self.send_response(200)
    #         self.send_header("Content-Type", "application/json")
    #         self.end_headers()
    #         response = {'aaa': 'aaaa'}

    #         json_response = json.dumps(response).encode()
    #         self.wfile.write(json_response)
    #         return
    #     else:
    #         self.send_response(400)
    #         self.send_header("Content-Type", "application/json")
    #         self.end_headers()

    # def do_GET(self):
    #     idx = self.path.find("?")
    #     print ('path', (self.path))
    #     if idx >= 0:
    #         params = cgi.parse_qs(self.path[idx+1:])
    #         if 'input' in params:
    #             print ('params', params)
    #             response = {'aaa': 'aaaa'}
    #             self.send_response(200)
    #             self.send_header("Content-Type", "application/json")
    #             self.end_headers()
    #             # json_response = "jsonCallback({})".format(json.dumps(response)).encode()
    #             json_response = json.dumps(response).encode()
    #             self.wfile.write(json_response)
    #             return

    #     self.send_response(400)
    #     self.send_header("Content-Type", "application/json")
    #     self.end_headers()

    def do_GET(self):
        def get_first_elem(d, keys):
            for k in keys:
                d[k] = d[k][0]
            return d

        idx = self.path.find("?")
        if idx >= 0:
            params = urllib.parse.parse_qs(self.path[idx+1:])
            print(params)
            if 'input' in params:
                response = generate_f(params['input'])[0]
                show_candidates = False if not 'show_candidates' in params or params['show_candidates'][0] != 'True' else True # booleanで返してくれない？
                if show_candidates:
                    pass
                else:
                    keys = [
                        'attention', 'alignment', 'output', 
                        'output_ids', 'output_tokens', 'positional_scores',
                    ]
                    response = get_first_elem(response, keys)
                response['args'] = args.__dict__
                json_response = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json_response)
                return

        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()



def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    set_random_seed(args)

    global generate_f
    generate_f = get_generate_func(args)
    if args.port is None: 
        # Interactive test via terminal
        print('| Type the input sentence and press return:')
        for inputs in buffered_read(args.input, args.buffer_size):
            res = generate_f(inputs)
            pprint(res)

    else: 
        # Interprocess 
        server= HTTPServer(("", args.port), DialogueRequestHandler)
        print("| Running server at port {}".format(args.port))
        server.serve_forever()

def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--tokenizer-type', default='mecab', type=str,
                        choices=['none', 'mecab'])
    parser.add_argument('--src-sentencepiece', default='', type=str, required=True)
    parser.add_argument('--tgt-sentencepiece', default='', type=str, required=True)
    parser.add_argument('--port', default=None, type=int)

    global args
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
