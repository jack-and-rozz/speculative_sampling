import torch
import sys

def parse_embedding(embed_path):
    embed_dict = {}
    with open(embed_path) as f_embed:
        line = f_embed.readline()
        while line:
            pieces = line.strip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
            try:
                line = f_embed.readline()
            except Exception as e:
                print(e, file=sys.stderr)
                continue

    return embed_dict
