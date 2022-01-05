# coding: utf-8
import os, re, sys, time, argparse, subprocess
import glob
import pandas as pd
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 101)
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sys.path.append(os.getcwd())

from common import RED, BLUE, RESET, UNDERLINE, modelname_converter

def read_data(file_path):
    return [l.strip() for l in open(file_path)]

def calc_dist(hypotheses, dist_n):
    '''
    https://www.aclweb.org/anthology/N16-1014
    '''
    assert dist_n >= 1
    assert type(hypotheses[0]) == str

    n_total_words = 0
    uniq_words = set()
    for hypothesis in hypotheses:
        words_in_hyp = [x for x in hypothesis.split() if x]
        ngrams = [tuple(words_in_hyp[i:i+dist_n]) for i in range(len(words_in_hyp)- dist_n+1)]
        for ngram in ngrams:
            uniq_words.add(ngram)
        n_total_words += len(ngrams)
    dist = 100.0 * len(uniq_words) / n_total_words if n_total_words else 0
    dist = "%.2f" % dist
    return dist


def calc_length(hypotheses):
    lens = [len(l.strip().split()) for l in hypotheses]
    avg_len = sum(lens) / len(lens)
    avg_len = "%.2f" % avg_len
    return avg_len

def main(args):
    output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename))
    output_paths = sorted(output_paths)
    outputs = {}
    for path in output_paths:
        model_name = path.split('/')[-3]
        model_prefix = '.'.join(model_name.split('.')[:-1])
        model_size = model_name.split('.')[-1]
        model_name = modelname_converter(model_prefix) + '.%s' % model_size
        if args.case_insensitive:
            path += '.lower'
        print(path, file=sys.stderr)
        outputs[model_name] = read_data(path)

    reference_file = args.reference_file
    if args.case_insensitive:
        reference_file += '.lower'

    print(reference_file, file=sys.stderr)
    # inputs = read_data(args.input_file)
    references = read_data(reference_file)

    # word_overlap_with_inputs = calc_overlap(inputs, references)
    dist1 =  calc_dist(references, 1)
    dist2 =  calc_dist(references, 2)
    average_length = calc_length(references)

    data = []
    model_name = 'Reference'
    data.append([model_name, dist1, dist2, average_length])

    header = ['Model', 'Dist-1', 'Dist-2', 'Avg. Length']
    for output_path in output_paths:
        #header = '/'.join(output_path.split('/')[:-2])
        model_name = output_path.split('/')[-3]
        model_prefix = '.'.join(model_name.split('.')[:-1])
        model_size = model_name.split('.')[-1]
        model_name = modelname_converter(model_prefix) + '.%s' % model_size

        dist1 = calc_dist(outputs[model_name], 1)
        dist2 = calc_dist(outputs[model_name], 2)
        average_length = calc_length(outputs[model_name])
        data.append([model_name, dist1, dist2, average_length])
    df = pd.DataFrame(data, columns=header).set_index('Model') 
    print('<Dist, Avg.len>')
    print(df)
    print()



if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('models_root', type=str)
    parser.add_argument('output_filename', type=str)
    parser.add_argument('input_file', type=str)
    parser.add_argument('reference_file', type=str)
    parser.add_argument('--case-insensitive', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
