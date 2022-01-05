# coding: utf-8
import os, re, sys, time, argparse, subprocess, random

def main(args):
    indices = set([int(idx) for idx in open(args.indice_path)])
    line = sys.stdin.readline()
    cnt=0
    while line: 
        if cnt in indices:
            sys.stdout.write(line)
        line = sys.stdin.readline()
        cnt += 1

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--indice-path', required=True)
    args = parser.parse_args()
    main(args)
