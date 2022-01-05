# coding: utf-8
import os, re, sys, time, argparse, subprocess, random
import datetime
import glob
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns


from icecream import ic
from typing import List, Dict, Union

sys.path.append(os.getcwd())
from common import modelname_converter, flatten

SCRIPT_NAME=os.path.basename(__file__)

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
spm_path = "dataset/twitter-v3/ja/processed.1turn.mecab.sp16000/spm.src.mecab.model"
sp.Load(spm_path)

Array = np.array

# plt.rcParams['axes.titlesize'] = 'medium'
print(plt.rcParams)
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.linewidth'] = 0.5

# # # font埋め込み
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = '\usepackage{sfmath}'

# plt.rcParams['xtick.major.width'] = 0.5
# plt.rcParams['ytick.major.width'] = 0.5
# plt.rcParams['xtick.major.width'] = 0.0
# plt.rcParams['ytick.major.width'] = 0.0

# exit(1)

def load_vectors(source_dir, suffix) -> Dict[str, Array]:
    file_paths = glob.glob(source_dir + '/*' + suffix)
    data = {}
    for fpath in file_paths:
        data_id = os.path.basename(fpath).split('.')[0]
        data[data_id] = np.loadtxt(fpath)
    return data

def load_scores(source_dir, suffix) -> Dict[str, Array]:
    return load_vectors(source_dir, suffix=suffix)

def load_conversations(conv_path):
    data = {}
    def convert2sent(sp, sp_list: List):
        return ''.join(sp.DecodePieces(sp_list).split())

    for l in open(conv_path):
        data_id, uttr, resp = l.strip().split('\t')
        data[data_id] = (
            convert2sent(sp, uttr.split()),
            convert2sent(sp, resp.split()),
        )
    return data

def normalize_scores_by_val(scores, *args):
    cmax = max(scores)
    cmin = min(scores)
    normed = [(c - cmin)/(cmax - cmin) * 2 - 1 for c in scores] # map [cmin, cmax] -> [-1, 1]
    return normed

def normalize_scores_by_val_among_models(scores, scores_among_models):
    
    cmax = max(flatten(scores_among_models))
    cmin = min(flatten(scores_among_models))
    normed = [(c - cmin)/(cmax - cmin) * 2 - 1 for c in scores] # map [cmin, cmax] -> [-1, 1]
    return normed


def normalize_scores_by_rank(scores, *args):
    rank = np.argsort(scores)
    normed = np.zeros(shape=scores.shape)
    for i, idx in enumerate(rank):
        normed[idx] = i/len(rank)
    normed = normalize_scores_by_val(normed)
    return normed


def main(args):
    model_paths = [
        'twitterv3ja_sp16000.baseline.tcvae.all',
        # 'twitterv3ja_sp16000.baseline.tcvae.kla9387.cycle18775.all',
        # 'twitterv3ja_sp16000.baseline.t-spacefusion.all',
        'twitterv3ja_sp16000.baseline.tcvae.bow1.all',
        # 'twitterv3ja_sp16000.baseline.tcvae.ls5.mean.all',
        'twitterv3ja_sp16000.baseline.tcvae.ls5.all',
    ]
    model_paths = [args.ckpt_root + '/' + mn for mn in model_paths]
    model_names = [modelname_converter('.'.join(os.path.basename(m).split('.')[:-1])) for m in model_paths]
    conversations = load_conversations(model_paths[0] + '/' + args.latent_dirname + "/input.txt")
    ic(model_names)


    num_points_suff = "." + str(args.num_points_prior) if args.num_points_prior else '.all'
    if args.without_conv:
        num_points_suff += '.noconv'

    output_dir = args.output_dir + '/' + args.domainname + '.both.scoreNorm' + num_points_suff
    draw_both_heatmap(args, model_names, model_paths, output_dir, conversations, 
                      score2color_f=normalize_scores_by_val)

    # output_dir = args.output_dir + '/' + args.domainname + '.both.scoreNorm.amongmodels' + num_points_suff
    # draw_both_heatmap(args, model_names, model_paths, output_dir, conversations, 
    #                   score2color_f=normalize_scores_by_val_among_models)

    # output_dir = args.output_dir + '/' + args.domainname + '.prior.scoreNorm' + num_points_suff
    # draw_prior_heatmap(args, model_names, model_paths, output_dir, conversations,
    #                    score2color_f=normalize_scores_by_val)

    # output_dir = args.output_dir + '/' + args.domainname + '.both.rankNorm' + num_points_suff
    # draw_both_heatmap(args, model_names, model_paths, output_dir, conversations, 
    #                   score2color_f=normalize_scores_by_rank)

    # output_dir = args.output_dir + '/' + args.domainname + '.prior.rankNorm' + num_points_suff
    # draw_prior_heatmap(args, model_names, model_paths, output_dir, conversations,
    #                    score2color_f=normalize_scores_by_rank)




def draw_both_heatmap(args, model_names, model_paths, output_dir, conversations, score2color_f) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load concatenated data. The first half are latent variables sampled from priors, the second half are latent variables sampled from posteriors.
    data = {}
    for i, (model_name, model_path) in enumerate(zip(model_names, model_paths)):
        latent_dir = model_path + '/' + args.latent_dirname
        data[model_name] = {}
        vecs = load_vectors(latent_dir, '.both.latent.tsne')
        scores = load_scores(latent_dir, '.both.score')
        n_vecs = vecs[list(vecs.keys())[0]].shape[0]

        prior_latent = {}
        prior_score = {}
        post_latent = {}
        post_score = {}
        for key in vecs.keys():
            prior_end_idx = int(n_vecs/2)
            prior_latent[key] = vecs[key][:prior_end_idx, :]
            prior_score[key] = scores[key][:prior_end_idx]
            post_latent[key] = vecs[key][prior_end_idx:, :]
            post_score[key] = scores[key][prior_end_idx]

        data[model_name]['prior_latent'] = prior_latent
        data[model_name]['prior_score'] =  prior_score
        data[model_name]['post_latent'] = post_latent
        data[model_name]['post_score'] =  post_score


    # Make a heatmap visualization for each example.
    data_ids = list(data[list(data.keys())[0]]['prior_score'].keys())
    num_examples = len(data_ids)
    model_names = list(data.keys())

    for example_idx in range(num_examples):
        data_id = data_ids[example_idx]
        # print(data_id)
        if data_id != '66':
            continue
        prior_latent_by_models = np.array([data[model_name]['prior_latent'][data_id] for model_name in model_names]) # (num_models, num_points/2, 2)

        all_prior_scores = [data[model_name]['prior_score'][data_id] for model_name in model_names]
        prior_color_by_models = np.array([
            score2color_f(
                data[model_name]['prior_score'][data_id], 
                all_prior_scores)
            for model_name in model_names])  # # (num_models, num_points)

        post_latent_by_models = np.array([data[model_name]['post_latent'][data_id] for model_name in model_names]) # (num_models, num_points/2, 2)
        target_path = output_dir + '/' + data_id

        plot_all_models(
            target_path,
            model_names, 
            prior_latent_by_models, 
            prior_color_by_models, 
            post_latent=post_latent_by_models,
            conv=conversations[data_id], 
            num_points_prior=args.num_points_prior,
            num_points_post=args.num_points_post
        )
        plt.clf() # Clear saved figures from memory. 
        # if example_idx > 5:
        #     break # debug


def draw_prior_heatmap(args, model_names, model_paths, output_dir, conversations, score2color_f) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load prior-only data.
    data = {}
    for i, (model_name, model_path) in enumerate(zip(model_names, model_paths)):
        latent_dir = model_path + '/' + args.latent_dirname
        data[model_name] = {}
        data[model_name]['prior_latent'] = load_vectors(latent_dir, '.prior.latent.tsne')
        data[model_name]['prior_score'] = load_scores(latent_dir, '.prior.score')


    # Make a heatmap visualization for each example.
    data_ids = list(data[list(data.keys())[0]]['prior_score'].keys())
    num_examples = len(data_ids)
    model_names = list(data.keys())

    for example_idx in range(num_examples):
        data_id = data_ids[example_idx]
        prior_latent_by_models = np.array([data[model_name]['prior_latent'][data_id] for model_name in model_names]) # (num_models, num_points, 2)
        prior_color_by_models = np.array([score2color_f(data[model_name]['prior_score'][data_id]) for model_name in model_names])  # # (num_models, num_points)
        target_path = output_dir + '/' + data_id
        plot_all_models(
            target_path,
            model_names, 
            prior_latent_by_models, 
            prior_color_by_models, 
            conv=conversations[data_id], 
            num_points_prior=args.num_points_prior)
        plt.clf() # Clear saved figures from memory. 
        # if example_idx > 5:
        #     break # debug


def plot_all_models(target_path, titles, prior_latent, prior_color, 
                    post_latent=None, 
                    num_points_prior=-1, num_points_post=-1,
                    cmap=cm.seismic, conv=None):
    '''
    - prior_latent: np.array[num_models, num_points, 2]
    - prior_color : np.array[num_models, num_points]
    - titles : list[num_models]
    '''
    # NROW = 2
    NROW = 1
    NCOL = 3
    # fig, axes = plt.subplots(NROW, NCOL, figsize=(6.4, 4.8))
    # fig, axes = plt.subplots(NROW, NCOL, figsize=(7.2, 2.4))
    fig, axes = plt.subplots(NROW, NCOL, figsize=(7.2, 2.1))
    base_ax = fig.add_subplot(111)
    base_ax.axis('off')

    def plot_to_ax(fig, ax, prior_latent, prior_color, post_latent=None,
                   size=16):
        prior_color = prior_color[:num_points_prior]
        post_color = (0, 1, 0, 1)
        # post_color = (1, 1, 0, 1)
        # post_color = 'yellow'
        # edgecolor = (0.6, 0.6, 0.6, 1)
        edgecolor = (0, 0, 0, 1)
        if num_points_prior:
            x = prior_latent[:num_points_prior, 0]
            y = prior_latent[:num_points_prior, 1]
        else:
            x = prior_latent[:, 0]
            y = prior_latent[:, 1]
        # sc = ax.scatter(x, y, vmin=-1, vmax=1, c=prior_color, cmap=cmap, 
        #                 s=size, marker='o', edgecolors=edgecolor)
        sc = ax.scatter(x, y, vmin=-1, vmax=1, c=prior_color, cmap=cmap, 
                        s=int(size * 0.5), marker='o')

        if post_latent is not None:
            x = post_latent[:num_points_post, 0]
            y = post_latent[:num_points_post, 1]
            ax.scatter(x, y, vmin=-1, vmax=1, color=post_color,
                       s=size, marker='^', edgecolors=edgecolor)
        return sc

    for i in range(NROW * NCOL):
        # ax = axes[int(i/3)][i%3]
        ax = axes[i%3]
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i < len(titles):
            ax.set_title(titles[i])
            if post_latent is not None:
                sc = plot_to_ax(fig, ax, prior_latent[i], prior_color[i], 
                                post_latent=post_latent[i])
            else:
                sc = plot_to_ax(fig, ax, prior_latent[i], prior_color[i])
        # fig.colorbar(sc)

    if args.lang == 'ja':
        cmd = "fc-list | grep 'IPA'"
        out = subprocess.getoutput(cmd)
        if not out.strip():
            print("IPAexGothic font was not found. Run 'apt install fonts-ipaexfont'.", file=sys.stderr)
        #     print("Config file path: ", matplotlib.matplotlib_fname())
        #     exit(1)


        # from matplotlib.font_manager import FontProperties
        # font_path = "/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf"
        # font_prop = FontProperties(fname=font_path)
        # matplotlib.rcParams["font.family"] = font_prop.get_name()

    # 会話の表示
    if conv and not args.without_conv:
        # text = str(datetime.datetime.now())
        yoffset = 0.00
        height = 0.060
        fsize=9
        # fig.text(0, yoffset + height * 2, "Data id: " + str(data_id), fontsize=fsize)
        # fig.text(0, yoffset + height * 1, "Uttr: " + conv[0], fontsize=fsize)
        # fig.text(0, yoffset + height * 0, "Gold Resp: " + conv[1], fontsize=fsize)
        uttr_en = " (I'm looking for delicious ways to eat vegetables.)"
        resp_en = " (Cut and wash vegetables. Put dressing. Enjoy.)"
        fig.text(0, yoffset + height * 1, "Uttrance: " + conv[0] + uttr_en, 
                 fontsize=fsize)
        fig.text(0, yoffset + height * 0, "Response: " + conv[1] + resp_en,
                 fontsize=fsize)


        # fig.text(0, 0, text)
        # fig.text(0, 0.025, text + ' 0.025')
        # fig.text(0, 0.05, text + ' 0.05')

    # カラーバーの設定
    axpos = base_ax.get_position()
    rect = [0.87, axpos.y0, 0.02, axpos.height]
    cbar_ax = fig.add_axes(rect)

    norm = colors.Normalize(vmin=prior_color[0].min(),vmax=prior_color[0].max())
    # norm = colors.Normalize(vmin=-1,vmax=1)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable._A = []
    cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=[-1, 1], 
                        aspect=10)
    # cbar.ax.set_yticklabels(['Low prob.', 'High prob.'])
    cbar.ax.set_yticklabels(['Low probability', 'High probability'])


    # cbar = fig.colorbar(mappable)
    # cbar.set_ticks([])
    # fig.text(0.8925, axpos.y0, "Low prob.", fontsize=10, weight='bold')
    # fig.text(0.8925, axpos.y0 + axpos.height - 0.01, "High prob.", fontsize=10, weight='bold')



    # 余白の調整
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(wspace=0.1)

    # plt.savefig(target_path + '.eps', bbox_inches='tight', pad_inches=0)
    # plt.savefig(target_path + '.png')


    plt.savefig(target_path + '.eps', bbox_inches='tight', pad_inches=0)
    # plt.savefig(target_path + '.eps', bbox_inches='tight')
    # plt.savefig(target_path + '.png', bbox_inches='tight')





if __name__ == "__main__":
    desc = ''
    # parser = argparse.ArgumentParser(description=desc)
    parser = argparse.ArgumentParser(
        # add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--domainname', default='twitterv3ja', type=str)
    parser.add_argument('--ckpt-root', default='checkpoints/latest', help=' ')
    parser.add_argument('-indir', '--latent-dirname', 
                        default='/analyses/fairseq.analysis.500', help=' ')
    parser.add_argument('-outdir', '--output-dir', 
                        default='logs/heatmap/', help=' ')
    parser.add_argument('--without-conv', action='store_true', default=False)

    parser.add_argument('--lang', default='ja')
    parser.add_argument('-npri', '--num_points_prior', default=300, help=' ', type=int)
    parser.add_argument('-npost', '--num_points_post', default=50, help=' ', type=int)
    # parser.add_argument('--input-suffix', default='.latent', help=' ')
    # parser.add_argument('--output-suffix', default='.latent.tsne', help=' ')
    # parser.add_argument('--ndim', default=2, type=int, help=' ')
    # parser.add_argument('--overwrite', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
