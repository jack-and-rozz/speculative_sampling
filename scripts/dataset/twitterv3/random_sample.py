import argparse, random, glob, subprocess, os
from orderedset import OrderedSet


# todo
def main(args):
    random.seed(args.seed)

    src_header = args.source_header
    src_fnames = glob.glob(src_header + '.*') 
    src_suffixs = ["." + fname.split('.')[-1] for fname in src_fnames]

    size_suffix = "." + str(args.N)
    tgt_header = src_header + size_suffix
    tgt_fnames = [src_header + size_suffix + suf for suf in src_suffixs]
    assert src_header + '.tids' in src_fnames
    num_lines = int(subprocess.getoutput("wc -l %s.tids" % src_header).split()[0])
    print(tgt_fnames)
    sampled_idxs = OrderedSet(list(sorted(random.sample(range(num_lines), args.N))))

    
    tgt_fname = tgt_header + '.idx'
    if not os.path.exists(tgt_fname or args.overwrite):   
        tf = open(tgt_fname, 'w')
        for idx in sampled_idxs:
            print(idx, file=tf)

    for i, src_fname in enumerate(src_fnames):
        tgt_fname = tgt_fnames[i]
        if not os.path.exists(tgt_fname or args.overwrite):
            sf = open(src_fname)
            tf = open(tgt_fname, 'w')
            for idx, line in enumerate(sf):
                if idx in sampled_idxs:
                    tf.write(line)
        
    
    
    # src = [l for l in open(args.src_file)]
    # output_header = '.'.join(args.src_file.split('.')[:-1])
    # src_suffix = '.' + args.src_file.split('.')[-1]
    # if args.N < 1000:
    #     raise ValueError('args.N cannot be less than 1000.')
    # elif args.N < 1000 * 1000:
    #     size_suffix = '.' + str(int(args.N / 1000)) + 'k'
    # else:
    #     size_suffix = '.' + str(int(args.N / 1000 / 1000)) + 'm'
    
    # n_samples = len(src)
    # picked_indices = random.sample(range(n_samples), args.N)

    # if args.tgt_file:
    #     tgt = [l for l in open(args.tgt_file)]
    #     assert len(src) == len(tgt)
    #     tgt_suffix = '.' + args.tgt_file.split('.')[-1]
    #     with open(output_header + size_suffix + tgt_suffix, 'w') as f:
    #         for idx in picked_indices:
    #             f.write(tgt[idx])

    # with open(output_header + size_suffix + src_suffix, 'w') as f:
    #     for idx in picked_indices:
    #         f.write(src[idx])
    
if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help=' ')
    parser.add_argument('source_header', type=str)
    parser.add_argument('N', type=int)
    parser.add_argument('--overwrite', action='store_true', help= ' ')
    args = parser.parse_args()
    main(args)
