# coding: utf-8
import sys, os, random, argparse, re, time, subprocess
sys.path.append(os.getcwd())


def main(args):
    lengths = []
    for i, l in open(args.source_path):
        lengths.append()
    pass


if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_path')
    parser.add_argument('-mr', '--max-rows', type=int, default=10000, help= ' ')
    global args
    args = parser.parse_args()
    main(args)
