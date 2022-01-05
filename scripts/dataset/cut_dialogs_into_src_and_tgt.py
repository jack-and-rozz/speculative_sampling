# coding: utf-8
from glob import glob
import string, json, subprocess
from collections import Counter, defaultdict
import os, re, sys, random, time, argparse, copy
sys.path.append(os.getcwd())


def main(args):
    output_header = '.'.join(args.dialog_file.split('.')[:-1])
    src_out_path = output_header + '.' + args.src_suffix
    tgt_out_path = output_header + '.' + args.tgt_suffix
    assert args.dialog_file != src_out_path
    assert args.dialog_file != tgt_out_path

    if not args.overwrite and os.path.exists(src_out_path):
        print(src_out_path, 'already exists. Set --overwrite or remove it.')

    if not args.overwrite and os.path.exists(tgt_out_path):
        print(tgt_out_path, 'already exists. Set --overwrite or remove it.')

    src_f = open(src_out_path, 'w')
    tgt_f = open(tgt_out_path, 'w')
    for l in open(args.dialog_file):
        l = l.split(args.eot_delimiter)
        print(args.eot_delimiter.join((l[:-1])).strip(), file=src_f)
        print(l[-1].strip(), file=tgt_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dialog_file')
    parser.add_argument('-src', '--src_suffix', default='src')
    parser.add_argument('-tgt', '--tgt_suffix', default='tgt')
    parser.add_argument('-eot', '--eot_delimiter', default='<eot>')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
