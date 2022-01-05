
# coding: utf-8
import os, re, sys, time, argparse, subprocess, random
import glob
import pandas as pd
from pprint import pprint
pd.set_option("display.max_colwidth", 80)

sys.path.append(os.getcwd())

from common import RED, BLUE, RESET, UNDERLINE, modelname_converter, flatten

EOT = '<eot>'

def read_references(path):
    refs = []
    for l in open(path):
        l = l.rstrip()
        idx, dialog = l.split('\t')
        idx = int(idx)
        context, response = dialog.split(EOT)
        context = context.replace('&amp;', '&')
        response = response.replace('&amp;', '&')
        refs.append([idx, context, response])
    idxs, inputs, outputs = list(zip(*refs))
    return idxs, inputs, outputs

def read_sampled_outputs(path, idxs):
    idxs = set(idxs)
    return [''.join(l.strip().split()).replace('&amp;', '&') for i, l in enumerate(open(path)) if i in idxs]

def get_model_type_and_size(output_path):
    model_name = output_path.split('/')[-3]
  
    model_type = '.'.join(model_name.split('.')[:-1])
    size = model_name.split('.')[-1]
    if size == 'fixed':
        model_type = '.'.join(model_name.split('.')[:-2] + [model_name.split('.')[-1]])
        size = model_name.split('.')[-2]
    model_type = modelname_converter(model_type)

    return model_type, size


def main(args):
    ref_idxs, ref_inputs, ref_outputs = read_references(args.ref_file)
    output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename))
    output_paths = [path for path in sorted(output_paths) if not re.search('backtranslation_aug', path)]

    outputs = {}
    for output_path in output_paths:
        model_type, size = get_model_type_and_size(output_path)
        if size not in args.target_model_sizes:
            continue
        model_name = output_path.split('/')[-3]
        model_prefix = '.'.join(model_name.split('.')[:-1])
        model_size = model_name.split('.')[-1]
        model_prefix = modelname_converter(model_prefix)
        model_name = model_prefix  + '.' + model_size
        if model_prefix not in args.target_model_types:
            continue
        outputs[model_prefix] = read_sampled_outputs(output_path, ref_idxs)

    _outputs = {}
    _outputs['Reference'] = ref_outputs

    model_names = args.target_model_types

    for name in model_names:
        if name in outputs:
            _outputs[name] = outputs[name]
    outputs = _outputs

    num_models = len(outputs)
    num_dialogs = len(ref_idxs)
    model_names = list(outputs.keys())


    output_idx_path = args.ref_file + '.idx'

    with open(output_idx_path, 'w') as f:
        for name in model_names:
            print(name, file=f)

    for i, name in enumerate(outputs):
        output_csv_path = args.ref_file + '.%d.csv' % i
        output_to_csv(output_csv_path, outputs[name], 
                      ref_idxs, ref_inputs)

    # output_csv_path = args.ref_file + '.ref.csv' % i
    # output_to_csv(output_csv_path, ref_outputs, 
    #               ref_idxs, ref_inputs)

def output_to_csv(output_csv_path, outputs, ref_idxs, ref_inputs):
    header = ['ID', '発話', '応答', 'Sensibleness', 'Specificity']
    data = []


    for i in range(len(ref_idxs)):
        l = [
            ref_idxs[i], 
            ref_inputs[i], 
            outputs[i],
            '',
            '',
        ]
        data.append(l)

    df = pd.DataFrame(data, columns=header).set_index('ID')
    with open(output_csv_path, 'w') as f:
        print(df.to_csv(), file=f)


if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--models_root', default='checkpoints/latest', type=str)
    parser.add_argument('--ref_file', default='dataset/twitter-v3/ja/processed.1turn.mecab.sp16000/human_eval/latest/0-1000/test.2019.dialogs.0-1000.sampled', type=str)
    parser.add_argument('--output_filename', default='twitterv3ja.outputs', 
                        type=str, help='')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--target_model_sizes', default=['all'], nargs='+')
    parser.add_argument('--target_model_types', default=[ 
        "Reference", 
        "Transformer",
        "SPACEFUSION", 
        "TCVAE", 
        "TCVAE + monotonic annealing", 
        "TCVAE + cyclic annealing", 
        "TCVAE + BoW loss",
        "TCVAE + speculative sampling (5)",
        "TCVAE + Monte Carlo (5)" ,
    ], nargs='+')
    args = parser.parse_args()
    random.seed(args.random_seed)
    main(args)
