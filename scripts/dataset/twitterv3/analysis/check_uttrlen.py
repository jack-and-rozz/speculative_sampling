# coding: utf-8
import sys, os, random, argparse, re, time
sys.path.append(os.getcwd())
import glob
from collections import defaultdict, OrderedDict, Counter


def main(args):
    cnt = 0
    context_path = args.source_dir + '/train.src'
    response_path = args.source_dir + '/train.tgt'
    fs = open(context_path)
    ft = open(response_path)

    valid_context_ids = set()
    valid_response_ids = set()

    for i, (c, r) in enumerate(zip(fs, ft)):
        if args.max_rows and i >= args.max_rows:
            break
        c = c.strip()
        r = r.strip()
        if len(c) >= args.min_words and len(c) <= args.max_words:
            valid_context_ids.add(i)
        else:
            pass
            # print('%d (%d, %d):\t' % (i, len(c), len(r)), c, '<eot>', r)

        if len(r) >= args.min_words and len(r) <= args.max_words:
            valid_response_ids.add(i)
        else:
            pass
            # print('%d (%d, %d):\t' % (i, len(c), len(r)), c, '<eot>', r)
        if i in valid_context_ids and i in valid_response_ids:
            print('%d (%d, %d):\t' % (i, len(c), len(r)), c, '<eot>', r)


    print('# all dialogs: ', i)
    print('# filtered contexts', len(valid_context_ids))
    print('# filtered responses', len(valid_response_ids))
    print('# filtered dialogs', len(valid_response_ids.intersection(valid_context_ids)))

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_dir', type=str, help ='')
    parser.add_argument('-min', '--min_words', type=int, default=3)
    parser.add_argument('-max', '--max_words', type=int, default=256)
    parser.add_argument('-mr', '--max_rows', type=int, default=100000)
    global args
    args = parser.parse_args()
    main(args)



