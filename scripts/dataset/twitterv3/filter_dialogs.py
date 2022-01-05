# coding: utf-8
import sys, os, random, argparse, re, time, subprocess
sys.path.append(os.getcwd())

import glob
from pprint import pprint
from common import timewatch, format_zen_han, flatten
from collections import defaultdict, OrderedDict, Counter

@timewatch()
def get_filtered_idx_by_uid(uids_path, bot_ids):
    idxs = [i for i, l in enumerate(open(uids_path))
            if (not args.max_rows or i < args.max_rows) and
            len([uid for uid in l.rstrip().split('\t') if uid in bot_ids]) > 0]
    return set(idxs)

@timewatch()
def get_filtered_idx_by_nwords(dialogs_path, eot_token):
    def filter_by_nwords(l):
        l = l.rstrip().split(eot_token)
        for uttr in l:
            if len(uttr.split()) < args.min_words or len(uttr.split()) > args.max_words:
                return False
        return True
    idxs = [i for i, l in enumerate(open(dialogs_path)) if (not args.max_rows or i < args.max_rows) and not filter_by_nwords(l) ]
    return set(idxs)

@timewatch()
def filtering(header, bot_ids):
    '''
    Remove dialogs if
    - one of the messages were made by a user whose name or profile contained 'bot.'
    - one of the tokenized messages were too short or long.
    '''
    files = glob.glob(header + '.*')
    uids_path = header + '.uids'
    dialogs_path = header + '.dialogs' 
    eot_token = " " + args.eot_token + " "

    assert uids_path in files 
    assert dialogs_path in files 

    # Tokenization is done in another script.
    # if args.lang == 'ja' and args.enable_tokenization:
    #     tokenized_dialogs_path +=  '.%s' % args.tokenize_suffix
    #     command = ['mecab', '-Owakati', '<', dialogs_path, '>', tokenized_dialogs_path]
    #     command = ' '.join(command)

    #     if not os.path.exists(tokenized_dialogs_path):
    #         print(command)
    #         os.system(command)
    #     dialogs_path = tokenized_dialogs_path
    # elif args.lang == 'en' and args.enable_tokenization:
    #     raise NotImplementedError

    filtered_idx = set()
    filtered_idx_uid = get_filtered_idx_by_uid(uids_path, bot_ids)
    filtered_idx = filtered_idx.union(filtered_idx_uid)

    if args.min_words > 0 and args.max_words > 0:
        filtered_idx_nwords = get_filtered_idx_by_nwords(dialogs_path, eot_token)
        filtered_idx = filtered_idx.union(filtered_idx_nwords)

    print('%d lines will be filtered out from "%s"' % (len(filtered_idx), header))
    save_filtered_files(header, filtered_idx)

def save_filtered_files(header, idxs):
    files = [path for path in glob.glob(header + '.*')]

    def _save(source_path, target_path, filtered_idxs):
        fs = open(source_path)
        ft = open(target_path, 'w')
        for idx, l in enumerate(fs):
            if args.max_rows and idx >= args.max_rows:
                break
            if idx not in filtered_idxs:
                ft.write(l)

    source_dir = '/'.join(header.split('/')[:-1])
    target_dir = source_dir + '.filtered'
    os.makedirs(target_dir, exist_ok=True)

    for source_path in files:
        target_path = target_dir + '/' + os.path.basename(source_path)

        if not os.path.exists(target_path) or args.overwrite:
            _save(source_path, target_path, idxs)

        # Tokenization is done in another script.
        # suffix = '.%s' % args.tokenize_suffix
        # if target_path.split('.')[-1] in ['src', 'tgt'] and \
        #    (args.overwrite or not os.path.exists(target_path + suffix)):
        #     command = ['mecab', '-Owakati', '<', target_path, '>', target_path + suffix]
        #     command = ' '.join(command)
        #     os.system(command)


def main(args):
    source_headers = ['.'.join(path.split('.')[:-1]) for path in glob.glob("%s/*.dialogs" % args.source_dir)]

    bot_ids = set([l.rstrip() for l in open(args.bot_id_path)])

    for header in source_headers:
        filtering(header, bot_ids)

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_dir')
    parser.add_argument('--bot-id-path', 
                        default='profile/prof.gathered/bot.uid', 
                        type=str, help ='Ids of users whose profile includes "bot."')
    parser.add_argument('-min', '--min-words', type=int, default=0, 
                        help="Messages (and subsequent conversations) containing less than this number of words are filtered out. Note that using this argument requires each message to be tokenized in advance. If set to 0, this filtering is disabled.")
    parser.add_argument('-max', '--max-words', type=int, default=0,
                        help="Messages (and subsequent conversations) containing more than this number of words are filtered out. Note that using this argument requires each message to be tokenized in advance. If set to 0, this filtering is disabled.")
    parser.add_argument('--lang', default='ja', help=' ')
    parser.add_argument('--eot-token', default='<eot>', help=' ')
    parser.add_argument('--overwrite', action='store_true', help= ' ')
    parser.add_argument('-mr', '--max-rows', type=int, default=0, help= 'mainly for debug.')

    global args
    args = parser.parse_args()
    
    main(args)




