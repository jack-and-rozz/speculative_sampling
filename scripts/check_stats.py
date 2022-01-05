# coding: utf-8
import argparse, random, sys, re, os
import glob
import subprocess as sp
from collections import defaultdict 

pattern = re.compile('wps=([0-9]+),')
sys.path.append(os.getcwd())
from common import RED, BLUE, RESET, UNDERLINE, modelname_converter

import pandas as pd
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 101)
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


output_pattern=re.compile("\| ([a-zA-Z_]+) ([e0-9\.\-]+)")

DNAME_DICT = {
    'twitterv3ja': 'Tw-Ja',
    'twitterv3en': 'Tw-En',

}
def domainname_converter(modelname):
    domainname = modelname.split('_')[0]
    return DNAME_DICT[domainname] if domainname in DNAME_DICT else domainname

def check_log(path):
    best_updates = None
    max_updates = None

    try:
        cmd1 = ["cat", path]
        cmd2 = ['grep', '-a', 'valid on']

        res1 = sp.Popen(cmd1, stdout=sp.PIPE)
        res2 = sp.check_output(cmd2, stdin=res1.stdout)
    except Exception as e:
        return 

    valid_stats_strs = res2.decode('utf-8').strip().split('\n')

    if not valid_stats_strs or not valid_stats_strs[0]:
        return

    valid_stats = []

    def _format_val(k, v):
        if '.' in v:
            v = float(v)
            if k in args.exp_cols:
                return '{:.3e}'.format(v)
            else:
                return "%.3f" % v
            # if v < 0.01:
            #     return '{:.3e}'.format(v)
            # else:
            #     return "%.3f" % float(v)

        return v

    for stats_str in valid_stats_strs:
        m = output_pattern.findall(stats_str)
        stats = dict([(k, _format_val(k, v)) for k, v in m])
        valid_stats.append(stats)

    losses = [float(stats['loss']) for stats in valid_stats]
    best_epoch_idx = losses.index(min(losses))

    best_stats = defaultdict(lambda : '-')

    for k, v in valid_stats[best_epoch_idx].items():
        best_stats[change_colname(k)] = v
    best_stats['ups(total)'] = valid_stats[-1]['num_updates']
    return best_stats

def change_colname(name):
    if name == 'num_updates':
        return 'ups(best)'
    elif name == 'current_updates':
        return 'ups(total)'
    if name == 'ppl':
        return 'valid ppl'
    return name

def main(args):
    log_file_paths= sorted(glob.glob("%s/*/train.log" % (args.models_root)))

    summary = defaultdict(dict)

    all_stats = []
    for i, path in enumerate(log_file_paths):
        if '.bak' in path:
            continue
        stats = check_log(path)
        if not stats:
            continue
        name = path.split('/')[-2]
        data_size = name.split('.')[-1]
        name = '.'.join(name.split('.')[:-1])
        domain = domainname_converter(name)
        name = modelname_converter(name)
        stats['Model'] = domain + ':' + name
        stats['#data'] = data_size
        all_stats.append(stats)
    all_keys = set()
    for stats in all_stats:
        all_keys = all_keys | set(stats.keys())
    print('Available colmuns: ', sorted(list(all_keys)), file=sys.stderr)

    # Do not display columns of which no model has the value.
    columns = ['Model']
    for i, col in enumerate(args.columns):
        exists = [stats for stats in all_stats if col in stats]
        if len(exists) != 0:
            columns.append(col)

    data = []
    for stats in all_stats:
        data.append(list(stats[k] for k in columns))

    df = pd.DataFrame(data, columns=columns).set_index('Model')
    print()
    print(df)


if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-rt', '--models_root', default='checkpoints/latest', 
                        type=str)
    parser.add_argument('-col', '--columns', type=str, nargs='+',
                        # default=['kld', 'prior_mu', 'post_mu', 'prior_std', 'post_std'])
                        default=['epoch', 'ups(best)', 'ups(total)', 'loss', 'nll_loss', 'valid ppl', 'kld', 'prior_mu', 'post_mu', 'prior_std', 'post_std', 'bow_loss', 'fuse_loss'])
    parser.add_argument('-exp', '--exp_cols', type=str, nargs='+',
                        default=[])
                        # default=['kld'])

    args = parser.parse_args()
    main(args)
