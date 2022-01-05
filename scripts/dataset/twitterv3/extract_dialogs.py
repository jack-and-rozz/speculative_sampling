# coding: utf-8
import sys, os, random, argparse, re, time
sys.path.append(os.getcwd())
import glob
from collections import defaultdict, OrderedDict, Counter
import emoji

from common import timewatch,  flatten

USERNAME_PATTERN1 = re.compile("^((@[0-9a-zA-Z_]+)\s*)+\s(.+)") # usernames at the beginning of a tweet.
# HASHTAG_PATTERN1 = re.compile("^(.+?)((#\S+\s*)*)$") # hashtags at the end of a tweet.
HASHTAG_PATTERN1 = re.compile("^(.+?)((#[^#\s]+\s*)*)$") # hashtags at the end of a tweet.
TIME, MID, UID, TEXT, HASHTAGS, EMOJIS = list(range(6))

INVALID_CHARS = set(['@', '{', '}', '#'])
EOT = ' <eot> '

ALL_EMOJIS = set(emoji.UNICODE_EMOJI)

def normalize_ja(l):
    # l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
    # l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
    l = ' '.join([neologdn.normalize(x, repeat=3) for x in l.split() if x])
    return l

def preprocess_uttr(uttr, keep_noisy_text=False):
    uttr = uttr.strip()

    # Remove usernames at the beginning of a message.
    m = USERNAME_PATTERN1.search(uttr)  
    if m:
        uttr = m.groups()[-1]

    # Parse and remove hashtags at the end of a message.
    m_hashtags = HASHTAG_PATTERN1.match(uttr)
    uttr, hashtags = m_hashtags.groups()[:2]
    if not hashtags:
        hashtags = '-'
    else:
        hashtags = ' '.join([x for x in hashtags.strip().split() if x])

    # Parse and remove emoticons.
    emojis = emoji.demojize(' '.join(list(set(uttr).intersection(ALL_EMOJIS))))
    if not emojis:
        emojis = '-'
    uttr = ''.join([c if c not in emoji.UNICODE_EMOJI else ' ' for c in uttr])


    if args.lang == 'ja':
        # if uttr != normalize_ja(uttr):
        #     print('Normalized (before):', uttr)
        #     print('Normalized (after) :', normalize_ja(uttr))
        uttr = normalize_ja(uttr)
    elif args.lang == 'en':
        pass

    # Normalize spaces between words.
    uttr = ' '.join([x for x in uttr.strip().split() if x])

    # Empty messages are discarded.
    if not uttr.strip():
        return None, None, None

    # Discarded if a message contains a screenname ('@XXXX') or a hashtag ('#XXXX') even after preprocessing above.
    if not keep_noisy_text and INVALID_CHARS.intersection(set(uttr)):
        return None, None, None

    # Discarded if the same character appears too frequently.
    char_freq = Counter(uttr).values()
    if max(char_freq) / len(uttr) > args.max_char_rate:
        return None, None, None

    return uttr, hashtags, emojis

@timewatch()
def read_data(file_path, skip_first=True, max_rows=0):
    data = {}
    is_leaf = {}
    text2ids = defaultdict(list)
    root_ids = set()
    # for i, l in enumerate(open(file_path)):
    for i, l in enumerate(open(file_path, encoding="utf8", errors='ignore')):
        # epochtime, tweet-id, mention-target-id, user-id, text
        if skip_first and i == 0:
            continue
        if max_rows and i >= max_rows + int(skip_first):
            break
        try:
            tweet_type, epoch_time, tid, mid, uid, uname, sname, text = l.strip().split('\t')
        except:
            continue

        text, hashtags, emojis = preprocess_uttr(
            text, 
            keep_noisy_text=args.keep_noisy_text)
        if not text:
            continue

        text2ids[text].append(tid)
        data[tid] = (epoch_time, mid, uid, text, hashtags, emojis)

        # 親をleafから除外
        is_leaf[mid] = False

        if mid == '-':
            root_ids.add(tid)

        # 既に自身を親とするnodeが存在していなければ仮に葉とする
        if tid not in is_leaf:
            is_leaf[tid] = True

    before_dup_len = len(data)
    for text, ids in text2ids.items():
        if len(ids) == 1:
            continue
        for _id in ids:
            del data[_id]
    after_dup_len = len(data)
    print('Removing duplicated (before, after):', before_dup_len, after_dup_len, file=sys.stdout)
    leaf_ids = [k for k,v in is_leaf.items() if v is True and k in data]
    return data, leaf_ids, root_ids


def is_root(d):
    if not d[MID] or d[MID] == '-':
        return True
    return False


@timewatch()
def trace_from_leaves_to_root(data, leaf_ids, max_chains=0):
  def _trace(data, tid, depth=0, max_depth=0):
    parent_id = data[tid][MID]
    if is_root(data[tid]):
        return [tid]
    elif  max_depth and depth == max_depth:
        return [tid]
    elif parent_id != tid and parent_id in data:
        return _trace(data, parent_id, depth=depth+1, max_depth=max_depth) + [tid]
    else:
        return [tid]

  id_seqs = []
  for i, tid in enumerate(leaf_ids):
    id_seq = _trace(data, tid, max_depth=max_chains)
 
    # Exclude stand-alone tweets.
    if len(id_seq) < 2:
        continue

    # Exclude conversations with missing contexts.
    if not args.keep_reply_to_missing_contexts and not is_root(data[id_seq[0]]):
        continue

    # Exclude monologues and multi-party conversations.
    if len(set([data[tid][UID] for tid in id_seq])) == 1:
        continue

    if (not args.keep_multi_party) and len(set([data[tid][UID] for tid in id_seq])) > 2:
        continue

    id_seqs.append(id_seq)
  return id_seqs


def save_dialogs(_target_path, data, id_seqs, leaf_ids, root_ids):
    # Save concatenated contexts. 
    target_path = _target_path + '.dialogs'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                print(EOT.join([data[tid][TEXT] for tid in id_seq]), file=f)

    # Save epoch-time.
    target_path = _target_path + '.utime'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
        exit(1)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                print('\t'.join([data[tid][TIME] for tid in id_seq]), file=f)

    # Save tweet-ids. 
    target_path = _target_path + '.tids'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                print('\t'.join(id_seq), file=f)

    # Save speaker-ids.
    target_path = _target_path + '.uids'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                print('\t'.join([data[tid][UID] for tid in id_seq]), file=f)

    target_path = _target_path + '.hashtags'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                # hashtags = [','.join(data[tid][HASHTAGS]) for tid in id_seq]
                hashtags = [data[tid][HASHTAGS] for tid in id_seq]
                print('\t'.join(hashtags), file=f)

    target_path = _target_path + '.emojis'
    if os.path.exists(target_path) and not args.overwrite:
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        with open(target_path, 'w') as f:
            for id_seq in id_seqs:
                # emojis = [','.join(data[tid][EMOJIS]) for tid in id_seq]
                emojis = [data[tid][EMOJIS] for tid in id_seq]
                print('\t'.join(emojis), file=f)


    # Save unchained tweets (as distractors).
    target_path = _target_path + '.distractors'
    if os.path.exists(target_path) and (not args.overwrite):
        print("\'%s\' already exists. Set --overwrite." % target_path)
    else:
        dist_ids = set(data.keys()) - set(flatten(id_seqs))
        with open(target_path, 'w') as f:
            for tid in dist_ids:
                print(data[tid][TEXT], file=f)

        print('#dialogs, #distractors =', len(id_seqs), len(dist_ids), file=sys.stdout)

def main(args):
    source_path = args.source_path
    source_filename = source_path.split('/')[-1].split('.')[0]
    target_header = args.target_dir + '/' + source_filename + '.' + args.lang
    if not os.path.exists(args.source_path):
        print("\'%s\' does not exist." % args.source_path, file=sys.stderr)
        exit(-1)
    if (not args.overwrite) and os.path.exists(target_header + '.dialogs'):
        print("\'%s\' already exists. Set --overwrite." % (target_header + '.dialogs'), file=sys.stderr)
        exit(-1)

    print(source_path)
    os.makedirs(args.target_dir, exist_ok=True)

    data, leaf_ids, root_ids = read_data(source_path, 
                                         max_rows=args.max_rows)
    print('#data, #leaves, #roots =', (len(data), len(leaf_ids), len(root_ids)),
          file=sys.stdout)
    id_seqs = trace_from_leaves_to_root(data, leaf_ids, 
                                        max_chains=args.max_chains)

    save_dialogs(target_header, data, id_seqs, leaf_ids, root_ids)

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('source_path', type=str, help ='')
    parser.add_argument('lang', type=str, help = '', choices=['ja', 'en'])
    parser.add_argument('--target_dir', default='original.dialogs', type=str, help ='')
    parser.add_argument('-mc', '--max_chains', type=int, default=5)
    parser.add_argument('-mr', '--max_rows', type=int, default=0)
    parser.add_argument('-mcr', '--max_char_rate', type=float, default=0.5, 
                        help='If a character appears more frequently than this percentage in a message, delete the message.')

    parser.add_argument('--keep-reply-to-missing-contexts', action='store_true')
    parser.add_argument('--keep-multi-party', action='store_true')
    parser.add_argument('--keep-noisy-text', action='store_true')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='if True, overwrite them even if there already exists preprocessed data.')
    global args
    args = parser.parse_args()
    if args.lang == 'ja':
        # import emoji, neologdn, mojimoji
        import neologdn

    main(args)

