# coding: utf-8
import os, re, sys, time, argparse, subprocess, random

import sentencepiece as spm

SCRIPT_NAME=os.path.basename(__file__)
def main(args):

    if not args.overwrite and os.path.exists(args.output_path):
        print(SCRIPT_NAME, ':', args.output_path, "already exists. Delete it or set --overwrite.")
        exit(1)
    print("Encoding '%s' to '%s'... " % (args.input_path, args.output_path))

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_path)
    fin = open(args.input_path)
    fout = open(args.output_path, 'w')

    for inp in fin:
        responses = [' '.join(sp.EncodeAsPieces(r)) for r in inp.split('\t')]
        print('\t'.join(responses), file=fout)

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--spm-path', required=True)
    parser.add_argument('--overwrite', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
