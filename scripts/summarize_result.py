# coding: utf-8
import os, re, sys, time, argparse, subprocess
import glob
from collections import defaultdict
sys.path.append(os.getcwd())
import pandas as pd
pd.set_option("display.max_colwidth", 80)

from common import RED, BLUE, RESET, UNDERLINE, flatten, modelname_converter

N_SPECIAL_WORDS = 4



def get_model_type_and_size(output_path):
    model_name = output_path.split('/')[-3]
    model_size = model_name.split('.')[-1]

    model_type = '.'.join(model_name.split('.')[:-1])

    # if size == 'fixed':
    #     model_type = '.'.join(model_name.split('.')[:-2] + [model_name.split('.')[-1]])
    #     size = model_name.split('.')[-2]
    # exit(1)
    model_type = modelname_converter(model_type)
    return model_type, model_size

def read_data(file_path, max_rows):
    return [l.strip() for i, l in enumerate(open(file_path)) if max_rows <= 0 or i < max_rows]

def decorate_unk(text, 
                 src_vocab=None, src_unk_color=None,
                 tgt_vocab=None, tgt_unk_color=None):
  if (not src_vocab or not src_unk_color) and \
     (not tgt_vocab or not tgt_unk_color):
    return text

  def color(_w):
      w = _w
      if src_vocab and _w not in src_vocab and src_unk_color:
          w = src_unk_color + w + RESET
      if tgt_vocab and _w not in tgt_vocab and tgt_unk_color:
          w = tgt_unk_color + w + RESET
      return w

  #return ' '.join([unk_color + w + RESET if not w in vocab else w for w in text.strip().split()])
  return ' '.join([color(w) for w in text.strip().split()])

def print_summary(inputs, references, outputs, 
                  src_enc_vocab, src_dec_vocab,
                  tgt_enc_vocab, tgt_dec_vocab,
                  target_words=None):

    num_models = len(outputs)
    num_unk_in_res = [0 for _ in range(num_models)]
    num_words_in_res = [0 for _ in range(num_models)]
    for i in range(len(inputs)):
        print('[%05d]' % i)
        inp = decorate_unk(inputs[i], 
                           src_vocab=src_enc_vocab, 
                           tgt_vocab=tgt_enc_vocab,
                           src_unk_color=RED,
                           tgt_unk_color=UNDERLINE)

        ref = decorate_unk(references[i], 
                           src_vocab=src_dec_vocab, 
                           tgt_vocab=tgt_dec_vocab,
                           src_unk_color=BLUE,
                           tgt_unk_color=UNDERLINE)
        if target_words:
            print('- Target word:', target_words[i])
        print('- Input:', inp)
        print('- Reference:', ref)
        for m_idx, (name, out) in enumerate(outputs.items()):
            try:
                o = decorate_unk(out[i],
                                 src_vocab=src_dec_vocab, 
                                 tgt_vocab=tgt_dec_vocab,
                                 src_unk_color=BLUE,
                                 tgt_unk_color=UNDERLINE)
            except:
                print(name, 'is an empty file!', file=sys.stderr)
                o = ''
                exit(1)

            # o = decorate_unk(out[i],
            #                  src_vocab=src_dec_vocab, 
            #                  tgt_vocab=tgt_dec_vocab,
            #                  src_unk_color=BLUE,
            #                  tgt_unk_color=UNDERLINE)

            # model_prefix = '.'.join(name.split('.')[:-1])
            # model_size = name.split('.')[-1]
            # name = modelname_converter(model_prefix) + '.%s' % model_size
            print('- Output [%s]:' % name, o)
        print()


def read_vocab(path, max_rows=0):
    if not os.path.exists(path):
        return None
    vocab = set()
    for i, l in enumerate(open(path)):
        if max_rows and i == max_rows:
            break
        if l.strip():
            vocab.add(l.strip().split()[0])
    return vocab

def calc_corpus_bleu(hyp_filepath, ref_filepath, 
                     script_path, case_insensitive=False):

    if case_insensitive:
        if not os.path.exists(hyp_filepath + '.lower'):
            with open(hyp_filepath + '.lower', 'w') as f:
                for l in open(hyp_filepath):
                    f.write(l.lower())
        if not os.path.exists(ref_filepath + '.lower'):
            with open(ref_filepath + '.lower', 'w') as f:
                for l in open(ref_filepath):
                    f.write(l.lower())
        hyp_filepath = hyp_filepath + '.lower'
        ref_filepath = ref_filepath + '.lower'
    print("Path", hyp_filepath, ref_filepath, file=sys.stderr)
    with open(hyp_filepath, 'r') as hyp_f:
        bleu_cmd = ['perl', script_path] + [ref_filepath, '2>/dev/null']
        bleu_out = subprocess.check_output(bleu_cmd, stdin=hyp_f, 
                                           stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode('utf-8').strip()
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        bleu_score = float(bleu_score)
        m = re.search("BLEU = ([0-9]+\.[0-9\.]+),", bleu_out)
        breu_score = m.group(1)
    return bleu_score, bleu_out

def calc_word_overlap(hyp_filepath, ref_filepath):
  ref_words = set(flatten([l.strip().split() for l in open(ref_filepath)]))
  hyp_words = set(flatten([l.strip().split() for l in open(hyp_filepath)]))
  both_words = ref_words.intersection(hyp_words)
  rate = len(both_words) / len(ref_words)
  return "%.3f" % (rate)

def read_target_word(input_path):
    target_word_path = '.'.join(input_path.split('.')[:-1]) + '.word'
    return [l.strip().split('%')[0] for l in open(target_word_path)]

def main(args):
    output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename))
    output_paths = [path for path in sorted(output_paths) if not re.search('backtranslation_aug', path)]

    inputs = read_data(args.input_file, args.max_rows)
    references = read_data(args.reference_file, args.max_rows)
    outputs = {}
    for output_path in output_paths:
        model_type, size = get_model_type_and_size(output_path)
        if size not in args.target_sizes:
            continue
        model_name = model_type + '.' + size
        # model_name= output_path.split('/')[-3]
        # model_name = modelname_converter(model_name)
        out = read_data(output_path, args.max_rows)
        if not out or len(out) < len(inputs):
            continue
        outputs[model_name] = out

    target_words = read_target_word(args.input_file) if args.task == 'descgen' else None

    max_vocabs = args.n_vocab - N_SPECIAL_WORDS if args.n_vocab else 0
    src_enc_vocab = read_vocab(args.src_enc_vocab, max_vocabs) if args.src_enc_vocab else None
    src_dec_vocab = read_vocab(args.src_dec_vocab, max_vocabs) if args.src_dec_vocab else None
    tgt_enc_vocab = read_vocab(args.tgt_enc_vocab, max_vocabs) if args.tgt_enc_vocab else None
    tgt_dec_vocab = read_vocab(args.tgt_dec_vocab, max_vocabs) if args.tgt_dec_vocab else None

    if not args.src_enc_vocab:
        src_dec_vocab = src_dec_vocab if src_dec_vocab else src_enc_vocab
    if not args.src_dec_vocab:
        src_enc_vocab = src_enc_vocab if src_enc_vocab else src_dec_vocab
    if not args.tgt_enc_vocab:
        tgt_dec_vocab = tgt_dec_vocab if tgt_dec_vocab else tgt_enc_vocab
    if not args.tgt_dec_vocab:
        tgt_enc_vocab = tgt_enc_vocab if tgt_enc_vocab else tgt_dec_vocab

    if not args.disable_output_all:
        print_summary(inputs, references, outputs, 
                      src_enc_vocab, src_dec_vocab,
                      tgt_enc_vocab, tgt_dec_vocab,
                      target_words)

    bleu_summary = defaultdict(dict)
    word_overlap_summary = defaultdict(dict)
    target_sizes = args.target_sizes


    for output_path in output_paths:
        model_type, size = get_model_type_and_size(output_path)
        if size not in target_sizes:
            continue
        bleu_score, bleu_out = calc_corpus_bleu(output_path, args.reference_file,
                                                args.corpus_bleu_path, args.case_insensitive)
        if bleu_score:
            bleu_summary[model_type][size] = bleu_score

        word_overlap_rate = calc_word_overlap(output_path, args.reference_file)
        word_overlap_summary[model_type][size] = word_overlap_rate
        print(model_type, size, '\t', bleu_out, file=sys.stderr)
    max_chars=30
    direction_tok='@'
    header = ['Model'] + list(target_sizes)

    # Show word overlap rates.
    data = [
        [model_name.split(direction_tok)[-1]] + [scores[size] if size in scores else '' for size in target_sizes] 
        for model_name, scores in word_overlap_summary.items()
    ]
    df = pd.DataFrame(data, columns=header).set_index('Model')

    print('<Word overlap rate>')
    print(df)
    print()

    # Show BLEU scores.
    data = [
        [model_name.split(direction_tok)[-1]] + [scores[size] if size in scores else '' for size in target_sizes] 
        for model_name, scores in bleu_summary.items()
    ]
    df = pd.DataFrame(data, columns=header).set_index('Model')
    print('<corpus-BLEU>')
    print(df)
    print()
    # print(df.to_csv(), file=sys.stderr)





if __name__ == "__main__":
  parser = argparse.ArgumentParser( 
    add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('models_root', type=str)
  parser.add_argument('output_filename', type=str)
  parser.add_argument('input_file', type=str)
  parser.add_argument('reference_file', type=str)
  parser.add_argument('--output_filename', default='dialogue_test.best', 
                      type=str, help='')
  parser.add_argument('--task', default='translation', type=str)
  parser.add_argument('--src_enc_vocab', default=None, type=str)
  parser.add_argument('--src_dec_vocab', default=None, type=str)
  parser.add_argument('--tgt_enc_vocab', default=None, type=str)
  parser.add_argument('--tgt_dec_vocab', default=None, type=str)
  parser.add_argument('--share_enc_dec_vocab', action='store_true', default=False)
  parser.add_argument('--disable_output_all', action='store_true', default=False)
  parser.add_argument('--n_vocab', default=50000, type=int)
  parser.add_argument('--max_rows', default=0, type=int, help='if zero, all examples are read.')
  parser.add_argument('--corpus_bleu_path', 
                      default='tools/mosesdecoder/scripts/generic/multi-bleu.perl')
  parser.add_argument('--target_sizes', default=['all'], nargs='+')

  parser.add_argument('--case-insensitive', action='store_true', default=False)
  
  args = parser.parse_args()
  main(args)
