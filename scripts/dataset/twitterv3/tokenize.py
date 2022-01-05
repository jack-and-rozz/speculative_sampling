# coding: utf-8
import sys, os, random, argparse, re, time
sys.path.append(os.getcwd())
import glob

# @timewatch()
def tokenize_by_mecab(file_paths, target_dir):
    import MeCab
    mecab = MeCab.Tagger('-Owakati')

    os.makedirs(target_dir, exist_ok=True)
    for source_path in file_paths:
        dirname = os.path.dirname(source_path).split('/')[-1]
        basename = os.path.basename(source_path)
        suffix = basename.split('.')[-1]
        target_path = target_dir + '/' + basename
        if os.path.exists(target_path) and not args.overwrite:
            continue
        if suffix == 'mulres':
            # One line has multiple sentences and thus needs to be decomposed before tokenization.
            ft = open(target_path, 'w')
            for l in open(source_path):
                print('\t'.join([mecab.parse(uttr).rstrip() for uttr in l.rstrip().split('\t')]), file=ft)

            pass
        elif suffix in args.target_suffixes:
            command = ['mecab', '-Owakati', '<', source_path, '>', target_path]
            command = ' '.join(command)
            print(command)
            os.system(command)
            pass
        else:
            command = ['ln', '-sf', '../%s/%s' % (dirname, basename), target_path]
            command = ' '.join(command)
            os.system(command)

def main(args):
    target_dir = args.source_dir + '.' + args.tokenizer
    files = glob.glob(args.source_dir + '/*')

    if args.lang == 'ja' and args.tokenizer == 'mecab':
        tokenize_by_mecab(files, target_dir)
    else:
        raise NotImplementedError('Invalid combination: %s and %s' % (args.lang, args.tokenizer))

if __name__ == '__main__':
    desc = ''
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_dir', type=str)
    parser.add_argument('lang', type=str, choices=['ja', 'en'])
    parser.add_argument('tokenizer', type=str, choices=['mecab'])
    parser.add_argument('--target-suffixes', nargs='+', 
                        default=['dialogs', 'src', 'tgt', 'mulres'],
                        help=' ')
    parser.add_argument('--eot-token', default='<eot>', help=' ')

    parser.add_argument('--overwrite', action='store_true', help= ' ')

    global args
    args = parser.parse_args()
    main(args)
