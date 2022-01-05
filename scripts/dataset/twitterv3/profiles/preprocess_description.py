# coding: utf-8
import sys, os, random, argparse, re, time
sys.path.append(os.getcwd())
import glob
from collections import defaultdict, OrderedDict, Counter
import emoji
from common import flatten

ALL_EMOJIS = set(emoji.UNICODE_EMOJI)
URL_PATTERN = re.compile("(http:|https:|www\.)\S+")
SYMURL = "<URL>"
def process(l):
    # l = l.replace('\n', ' ').strip()
    l = l.strip()
    l = "".join([c for c in l if c not in emoji.UNICODE_EMOJI])
    # l = emoji.demojize(l)
    l = URL_PATTERN.sub(SYMURL, l)
    l = " ".join([w for w in l.split() if w])
    l = "" if l == SYMURL else l
    return l

def main(args):
    # path = 'prof.gathered/user.prof.uid'
    # uids = set([l.strip() for l in open(path)])

    # path = '../dataset.ja.bak2/train.2018.uids'
    # train = set(flatten([l.strip().split('\t') for l in open(path)]))
    # print(len(uids), len(train), len(uids.intersection(train)))

    # path = '../dataset.ja.bak2/train.2017.uids'
    # train = set(flatten([l.strip().split('\t') for l in open(path)]))
    # print(len(uids), len(train), len(uids.intersection(train)))
    


    target_path=args.source_path + '.processed'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
        exit(1)

    fs = open(args.source_path, encoding="utf8", errors='ignore')
    ft = open(target_path, 'w')
    l = fs.readline()
    while l:
        l = process(l)
        print(l, file=ft)
        try:
            l = fs.readline()
        except Exception as e:
            l = "\n"
    # print(cnt)

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_path', 
                        default='prof.gathered/user.prof.description',
                        type=str, help ='')
    # parser.add_argument('--source_path', 
    #                     default='prof.gathered/user.prof.description.joined',
    #                     type=str, help ='')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='if True, overwrite them even if there already exists preprocessed data.')
    global args
    args = parser.parse_args()
    main(args)

