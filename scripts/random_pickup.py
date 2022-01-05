import argparse, random, sys, re

def main(args):
    random.seed(args.seed)
    src = [l for l in open(args.src_file)]
    src_suffix = '.' + args.src_suffix

    if args.src_file != args.tgt_file:
        tgt = [l for l in open(args.tgt_file)]
        tgt_suffix = '.' + args.tgt_suffix
    m = re.search("(.+)%s" % src_suffix ,args.src_file)
    output_header = m.group(1)
    print(output_header, file=sys.stderr)


    if args.N[-1].lower() == 'k':
        N = int(args.N[:-1]) * 1000
    else:
        N = int(args.N)

    if N < 1000:
        raise ValueError('N cannot be less than 1000.')
    else:
        size_suffix = '.' + str(int(N / 1000)) + 'k'

    if args.src_file != args.tgt_file:
        assert len(src) == len(tgt)

    n_samples = len(src)
    
    if N > n_samples:
        print('args.N must be smaller than or equal to the size of the original file.', file=sys.stderr)
        exit(1)

    picked_indices = random.sample(range(n_samples), N)

    with open(output_header + size_suffix + src_suffix, 'w') as f:
        for idx in picked_indices:
            f.write(src[idx])

    with open(output_header + size_suffix + '.idx', 'w') as f:
        for idx in picked_indices:
            f.write(str(idx) + '\n')

    if args.src_file != args.tgt_file:
        with open(output_header + size_suffix + tgt_suffix, 'w') as f:
            for idx in picked_indices:
                f.write(tgt[idx])


if __name__ == "__main__":
    desc = 'if you want to pick up examples from only one file, make args.src_file and args.tgt_file identical.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_file', type=str)
    parser.add_argument('tgt_file', type=str)
    parser.add_argument('src_suffix', type=str)
    parser.add_argument('tgt_suffix', type=str)
    parser.add_argument('N', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args)
 
# python scripts/random_pickup.py dataset/ubuntu-dialog/processed.moses.nourl/train.3turns.src dataset/ubuntu-dialog/processed.moses.nourl/train.3turns.tgt 100000
