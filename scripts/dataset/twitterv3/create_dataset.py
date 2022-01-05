# coding: utf-8
import sys, os, random, argparse, re, time, random, subprocess
from datetime import datetime
sys.path.append(os.getcwd())
import glob
from collections import defaultdict, OrderedDict, Counter
import emoji

from common import timewatch,  flatten

ALL_EMOJIS = set(emoji.UNICODE_EMOJI)
EOT = ' <eot> '

# @timewatch() 
ITEM_CANDIDATES=('utc', 'uids', 'emojis', 'hashtags')
UTC_FORMAT="%Y-%m%d-%H%M"

def convert_from_utime_to_utc(l):
    utc = [datetime.utcfromtimestamp(int(utime)).strftime(UTC_FORMAT)
           for utime in l.strip().split()]
    utc = '\t'.join(utc).rstrip()
    return utc

def process_uttr(uttr):
    return " ".join([w for w in uttr.split() if w]).strip()

def cut_turn(l, num_turns, delim):
    l = [process_uttr(uttr) for uttr in l.split(delim)]
    if num_turns:
        l = l[:(num_turns+1)]
    res = delim.join(l)
    return res

def cut_context(dialog):
    return EOT.join(dialog.split(EOT)[:-1])

def cut_response(dialog):
    return dialog.split(EOT)[-1]

@timewatch()
def create_train_dev_dataset(target_dir, source_headers, year, profiles):
    extracted_items = ['tids', 'dialogs'] + list(args.extracted_items)
    existing_items = []
    for i, ext in enumerate(extracted_items):
        if not args.overwrite and os.path.exists(target_dir + '/train.%d.%s' % (year, ext)):
            print(target_dir + '/train.%d.%s'  % (year, ext) + ' already exists.')
            existing_items.append(ext)

    extracted_items = [item for item in extracted_items if item not in existing_items] 
    if not extracted_items:
        return

    # if 'utime' in extracted_items:
    #     extracted_items.append('utc')

    data = [[] for _ in extracted_items]

    for i, ext in enumerate(extracted_items):
        delim = EOT if ext == 'dialogs' else '\t'
        for header in source_headers:
            if ext == 'utc':
                data[i] += [cut_turn(convert_from_utime_to_utc(l), args.num_turns, delim)
                            for j, l in enumerate(open(header + '.utime'))
                            if (not args.max_rows or j < args.max_rows)]

            else:
                data[i] += [cut_turn(l.rstrip(), args.num_turns, delim)
                            for j, l in enumerate(open(header + '.' + ext))
                            if (not args.max_rows or j < args.max_rows)]


    train_idxs = set(range(len(data[0])))
    valid_idxs = set(random.sample(train_idxs, args.ndev))
    train_idxs -= valid_idxs

    for i, ext in enumerate(extracted_items):
        f = open(target_dir + '/train.%d.%s' % (year, ext), 'w')
        for idx in train_idxs:
            print(data[i][idx], file=f)

    for i, ext in enumerate(extracted_items):
        f = open(target_dir + '/dev.%d.%s' % (year, ext), 'w')
        for idx in valid_idxs:
            print(data[i][idx], file=f)


    if not 'dialogs' in extracted_items:
        return

    # Splitted dataset for nmt format.
    dialog_idx = extracted_items.index('dialogs')
    fs = open(target_dir + '/train.%d.%s' % (year, 'src'), 'w')
    ft = open(target_dir + '/train.%d.%s' % (year, 'tgt'), 'w')
    for idx in train_idxs:
        print(cut_context(data[dialog_idx][idx]), file=fs)
        print(cut_response(data[dialog_idx][idx]), file=ft)

    fs = open(target_dir + '/dev.%d.%s' % (year, 'src'), 'w')
    ft = open(target_dir + '/dev.%d.%s' % (year, 'tgt'), 'w')
    for idx in valid_idxs:
        print(cut_context(data[dialog_idx][idx]), file=fs)
        print(cut_response(data[dialog_idx][idx]), file=ft)


# def check_id_trees(header):
#     id_tree = defaultdict(list)
#     dialog_tree = defaultdict(list)
#     id_f = open(header + '.tids')
#     dialog_f = open(header + '.dialogs')
#     for i, (ids, dialog) in enumerate(zip(id_f, dialog_f)):
#         ids = ids.strip().split('\t')
#         dialog = dialog.strip().split(EOT)
#         for j in range(1, len(ids)):
#             if ids[j] not in id_tree[tuple(ids[:j])]:
#                 id_tree[tuple(ids[:j])].append(ids[j])
#                 dialog_tree[tuple(dialog[:j])].append(dialog[j])
#     print(header, len(dialog_tree))
#     cnt = []
#     for k, v in dialog_tree.items():
#         cnt.append(len(v))
#         if len(v) >= 3 and len(k) >= 1:
#             print('Uttr:', EOT.join(k))
#             for r in v:
#                 print(' - Res:', r)
#     cnt = Counter(cnt)
#     print(i)
#     print(len(id_tree))
#     print(len(dialog_tree))
#     print(cnt)

@timewatch()
def create_dialog_tree(source_headers, ext, min_num_responses, delim):
    context = defaultdict(str)
    responses = defaultdict(list)
    num_turns = args.num_turns if args.num_turns else 100
    for header in source_headers:
    # for header in source_headers[:2]: # debug
        id_f = open(header + '.tids')
        if ext == 'utc':
            data_f = open(header + '.utime')
            process_f = lambda x: [datetime.utcfromtimestamp(int(utime)).strftime(UTC_FORMAT) for utime in x]
        elif ext == 'dialogs':
            data_f = open(header + '.%s' % ext)
            process_f = lambda x: [process_uttr(uttr) for uttr in x]
        else:
            data_f = open(header + '.%s' % ext)
            process_f = lambda x: x
        for i, (tids, data) in enumerate(zip(id_f, data_f)):
            tids = tids.strip().split('\t')
            data = process_f(data.strip().split(delim))
            for j in range(1, min(len(tids), num_turns+1)):
                context[tuple(tids[:j])] = delim.join(data[:j])
                responses[tuple(tids[:j])].append(data[j])
    responses = {k:v for k, v in responses.items() if len(v) >= min_num_responses}
    context = {k:v for k, v in context.items() if k in responses}

    return context, responses

@timewatch()
def create_test_dataset(target_dir, source_headers, year, profiles):
    extracted_items = list(args.extracted_items)
    existing_items = []
    if os.path.exists(target_dir + '/test.%d.dialogs' % year) and not args.overwrite:
        print(target_dir + '/test.%d.dialogs' % year, 'already exists.')
        return
    for i, ext in enumerate(extracted_items):
        if not args.overwrite and os.path.exists(target_dir + '/test.%d.%s' % (year, ext)):
            print(target_dir + '/test.%d.%s'  % (year, ext) + ' already exists.')
            existing_items.append(ext)

    extracted_items = [item for item in extracted_items if item not in existing_items or item == 'tids'] 
    extracted_items = ['tids', 'dialogs'] + extracted_items

    # Choose dialogs with multiple responses
    candidate_idx = 0 # Use the first response to a context.
    context_items = []  # context_items[item_type][tid_tuple]
    response_items = [] # response_items[item_type][tid_tuple]
    for ext in extracted_items:
        delim = EOT if ext == 'dialogs' else '\t'
        context, responses = create_dialog_tree(source_headers, ext, 
                                                args.min_responses_in_test, delim)
        context_items.append(context)
        response_items.append(responses)
    test_tids = random.sample(context_items[0].keys(), args.ntest)

    for i, ext in enumerate(extracted_items):
        if os.path.exists(target_dir + '/test.%d.%s' % (year, ext)) and not args.overwrite:
            continue
        delim = EOT if ext == 'dialogs' else '\t'
        f = open(target_dir + '/test.%d.%s' % (year, ext), 'w')
        for tids in test_tids:
            print(delim.join([context_items[i][tids], response_items[i][tids][candidate_idx]]), file=f)


    # Split dialogs into a context and a response.
    tid_idx = extracted_items.index('tids')
    dialog_idx = extracted_items.index('dialogs')
    if not os.path.exists(target_dir + '/test.%d.%s' % (year, 'src')) or args.overwrite:
        fs = open(target_dir + '/test.%d.%s' % (year, 'src'), 'w')
        ft = open(target_dir + '/test.%d.%s' % (year, 'tgt'), 'w')
        for tids in test_tids:
            print(context_items[dialog_idx][tids], file=fs)
            print(response_items[dialog_idx][tids][candidate_idx], file=ft)

    # Save multiple response candidates for advanced evaluation.
    if not os.path.exists(target_dir + '/test.%d.%s' % (year, 'mulres')) or args.overwrite:
        f = open(target_dir + '/test.%d.%s' % (year, 'mulres'), 'w')
        for tids in test_tids:
            print('\t'.join(response_items[dialog_idx][tids]), file=f)

    if not os.path.exists(target_dir + '/test.%d.%s' % (year, 'mulres.tids')) or args.overwrite:
        f = open(target_dir + '/test.%d.%s' % (year, 'mulres.tids'), 'w')
        for tids in test_tids:
            print('\t'.join(response_items[tid_idx][tids]), file=f)
    return

def read_profile(path):
    profiles = dict([tuple(l.strip().split('\t')) for i, l in enumerate(open(path)) if len(l.strip().split('\t')) == 2 and (not args.max_rows or i <= args.max_rows)])
    return profiles


def main(args):
    # profiles = read_profile(args.profile_path) if args.profile_path else {}
    profiles = {}
    target_dir = args.target_dir_basename 
    target_dir += '.%s' % (args.lang)
    if args.num_turns:
        target_dir += '.%dturn' % (args.num_turns)

    os.makedirs(target_dir, exist_ok=True)
    train_dev_headers = []
    test_headers = []
    for year in args.train_dev_years:
        print("Processing dialogs in %d as train/dev data..." % year)
        train_dev_headers = [os.path.splitext(x)[0] for x in sorted(glob.glob(args.source_dir + '/%d-*-*.%s.tids' % (year, args.lang)))]
        create_train_dev_dataset(target_dir, train_dev_headers, year, profiles)
        # create_train_dev_dataset(target_dir, train_dev_headers[:2], year, profiles) # debug

    for year in args.test_years:
        print("Processing dialogs in %d as test data..." % year)
        test_headers = [os.path.splitext(x)[0] for x in sorted(glob.glob(args.source_dir + '/%d-*-*.%s.tids' % (year, args.lang)))]
        create_test_dataset(target_dir, test_headers, year, profiles)
        # create_test_dataset(target_dir, test_headers[:2], year, profiles) # debug


# 2016-ja: 9278792
# 2017-ja: 9684948
# 2018-ja: 9406811
# 2019-ja: 6270691

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('lang', type=str, choices=['en', 'ja'])
    parser.add_argument('--train-dev-years', metavar='N',
                        default=(2017, 2018,), 
                        type=int, nargs='+', help=' ')
    parser.add_argument('--test-years', metavar='N',
                        default=(2019,),
                        type=int, nargs='+', help=' ')
    parser.add_argument('--ndev', type=int, default=100000, help=' ')
    parser.add_argument('--ntest', type=int, default=100000, help= ' ')
    parser.add_argument('--extracted-items', type=str, 
                        default=ITEM_CANDIDATES, help=' ')
    parser.add_argument('-src', '--source-dir', type=str, 
                        default='original.dialogs', 
                        help=' ')
    parser.add_argument('-tgt', '--target-dir-basename', type=str, 
                        default='dataset', 
                        help=' ')
    parser.add_argument('--random-seed', type=int, default=0, help= ' ')
    parser.add_argument('--overwrite', action='store_true', help= ' ')
    parser.add_argument('--profile-path', default='profile/prof.gathered/user.prof.description.joined')
    parser.add_argument('-mr', '--max-rows', type=int, default=0, help= ' ')
    parser.add_argument('-nt', '--num-turns', type=int, default=0, help= ' ')
    parser.add_argument('-minr', '--min-responses-in-test', 
                        type=int, default=3, help= ' ')
    global args
    args = parser.parse_args()
    for item in args.extracted_items:
        assert item in ITEM_CANDIDATES

    random.seed(args.random_seed)
    main(args)
