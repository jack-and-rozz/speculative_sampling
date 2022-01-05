# coding: utf-8
import os, re, sys, time, argparse, subprocess, random
import glob
from fastTSNE import TSNE
import numpy as np
sys.path.append(os.getcwd())


SCRIPT_NAME=os.path.basename(__file__)
def apply_tsne(tsne, input_path, output_path):
    latents = np.loadtxt(input_path)
    latents_tsne = tsne.fit(latents)
    np.savetxt(output_path, latents_tsne)

def main(args):
    tsne = TSNE(n_components=args.ndim)
    latent_files = glob.glob('%s/*%s' % (args.latent_root, args.input_suffix))

    exist_flag=False
    for input_path in latent_files:
        output_path = '.'.join(input_path.split('.')[:-1]) + args.output_suffix
        if not args.overwrite and os.path.exists(output_path):
            exist_flag=True
            continue
        apply_tsne(tsne, input_path, output_path)
    if exist_flag:
        print(SCRIPT_NAME, ":", "Some output files already existed and were skipped. Manually delete them or set --overwrite if needed.")

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('latent_root', type=str)
    parser.add_argument('--input-suffix', default='.latent', help=' ')
    parser.add_argument('--output-suffix', default='.latent.tsne', help=' ')
    parser.add_argument('--ndim', default=2, type=int, help=' ')
    parser.add_argument('--overwrite', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
