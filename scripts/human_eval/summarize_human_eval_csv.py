# coding: utf-8
import os, re, sys, time, argparse, subprocess, random
import numpy as np
from scipy import stats
import glob
import pandas as pd
from collections import defaultdict
import itertools
import openpyxl
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 101)
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format
sys.path.append(os.getcwd())

SENSIBLENESS=0
SPECIFICITY=1
SSA=2

from common import RED, BLUE, RESET, UNDERLINE, modelname_converter, flatten

def check_agreement(anno_results, num_conversations, num_models, col=SSA):
    def concat_anno_result(result, col):
        res = []
        for model, r in result.items():
            res += r[:, col].tolist()
        return np.array(res)
    annotators = list(anno_results.keys())
    num_annotators = len(annotators)
    annotator2id = {annotator:i for i, annotator in enumerate(annotators)}

    scores_by_annotator = {anno_name: concat_anno_result(result, col) 
                           for anno_name, result in anno_results.items()}

    result = [[0 for name in annotators] for name in annotators]
    for a1, a2 in itertools.combinations(annotators, 2):
        a1id = annotator2id[a1]
        a2id = annotator2id[a2]
        scores1 = scores_by_annotator[a1]
        scores2 = scores_by_annotator[a2]
        coef = np.corrcoef(scores1, scores2)[0, 1]
        result[a1id][a2id] = coef
        result[a2id][a1id] = coef
    np_result = np.array(result)
    result = [[name] + r for name, r in zip(annotators, result)]
    header = annotators
    df = pd.DataFrame(result, columns=['Annotator'] + header).set_index('Annotator')
    print(df)
    print()
    avg_coef = np_result.sum() / (num_annotators^2 - num_annotators)
    print('Avg. coef=', avg_coef)



def read_excel(path, idx2modelname):
    wb = openpyxl.load_workbook(path)
    num_sheets = len(wb.worksheets)
    evals_by_model = {}
    for model_idx in range(num_sheets - 3):
        sheet = wb['%d' % model_idx]
        evals = sheet["D2:E101"]
        # evals = list(zip(*evals))
        evals = np.array([[float(cell.value) if cell.value and cell.value >= 1 else 0 for cell in row] for row in evals])

        ave = evals.mean(axis=1, keepdims=True)
        evals = np.append(evals, ave, -1)
        modelname = idx2modelname[model_idx]
        evals_by_model[modelname] = evals
    return evals_by_model

        # sense = [float(cell.value) if cell.value else 0 for cell in evals[0] ]
        # spec = [float(cell.value) if cell.value else 0 for cell in evals[1]]
    #     sensibleness.append(sense)
    #     specificity.append(spec)
    # return sensibleness, specificity


def main(args):
    modelname_path = "%s/%s.idx" % (args.data_dir, args.test_file_prefix)
    idx2modelname = modelnames = [l.rstrip() for l in open(modelname_path)]

    anno_paths = glob.glob("%s/%s.*.xlsx" % (args.data_dir,
                                             args.anno_file_prefix))
    anno_results_by_annotator = {}
    for path in anno_paths:
        anno_name = path.split('/')[-1].split('.')[1]
        anno_results_by_annotator[anno_name] = read_excel(path, idx2modelname)

    num_models = len(modelnames)
    num_annotators = len(anno_paths)
    num_conversations = len(anno_results_by_annotator[anno_name][modelnames[0]])

    print('# models, workers, dialogs', 
          num_models, num_annotators, num_conversations)


    eval_mean = np.zeros((num_models, 3))
    for anno_name, scores_by_model in anno_results_by_annotator.items():
        df = compute_scores_by_model(scores_by_model, modelnames, num_conversations)
        print("<%s>" % anno_name)
        print(df)
        print()

        eval_mean += df[["Sensibleness", "Specificity", "SSA"]] / num_annotators


    # average scores by annotator
    print('<avg>')
    print(eval_mean)

    check_agreement(anno_results_by_annotator, num_conversations, num_models)


    

def compute_scores_by_model(scores_by_model, modelnames, num_conversations):
    num_models = len(modelnames)

    summarized_scores_by_model = {modelname:np.zeros(shape=(num_conversations, 3)) for modelname in modelnames}

    header = ['Model', 'Sensibleness', 'Specificity', 'SSA']
    data = []
    

    for modelname in modelnames:
        # for _, scores_by_model in anno_results.items():
        summarized_scores_by_model[modelname] += scores_by_model[modelname]

        mean_sensibleness = summarized_scores_by_model[modelname][:, 0].mean()
        mean_specificity = summarized_scores_by_model[modelname][:, 1].mean()
        mean_ssa = summarized_scores_by_model[modelname][:, 2].mean()
        # data.append([name_converter(modelname), mean_sensibleness, mean_specificity, mean_ssa])
        data.append([modelname, mean_sensibleness, mean_specificity, mean_ssa])

    df = pd.DataFrame(data, columns=header).set_index('Model')
    return df

    # anno_results = dict([read_anno(path, idx2modelname) for path in anno_paths])

    # modelnames = set(idx2modelname)
    # num_models = len(modelnames)
    # num_annotators = len(anno_paths)
    # num_conversations = len(list(list(anno_results.values())[0].values())[0])

    # print('#models, #annotators, #dialogs =', num_models, num_annotators, num_conversations)
    

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', default='dataset/twitter-v3/ja/processed.1turn.mecab.sp16000/human_eval/latest/0-1000', )
    parser.add_argument('--test_file_prefix', 
                        default='test.2019.dialogs.0-1000.sampled')
    parser.add_argument('--anno_file_prefix', 
                        default='20210130')
    args = parser.parse_args()
    main(args)


def check_sig_diff(summarized_scores_by_model, col=SSA, sig_level=None):
    modelnames = list(summarized_scores_by_model.keys())
    header = [name_converter(name) for name in modelnames]

    modelname2id = {modelname:i for i, modelname in enumerate(modelnames)}
    # result = [[name_converter(name)] + ['-' for name in modelnames] 
    #           for name in modelnames]
    result = [['-' for name in modelnames] for name in modelnames]

    for m1, m2 in itertools.combinations(modelnames, 2):
        scores1 = summarized_scores_by_model[m1][:, col]
        scores2 = summarized_scores_by_model[m2][:, col]
        m1id = modelname2id[m1]
        m2id = modelname2id[m2]
        res = stats.ttest_rel(scores1, scores2).pvalue
        # res = stats.wilcoxon(scores1, scores2).pvalue
        if sig_level is None or sig_level <= 0:
            res = '%.3f' % res
        elif res < sig_level:
            res = 'o'
        else:
            res = 'x'
        # res = 'o' if res.pvalue < 0.05 else 'x'
        result[m1id][m2id] = res
        result[m2id][m1id] = res
    result = [[name_converter(name)] + r for name, r in zip(modelnames, result)]
    df = pd.DataFrame(result, columns=['Model'] + header).set_index('Model')
    print(df)
    print()

