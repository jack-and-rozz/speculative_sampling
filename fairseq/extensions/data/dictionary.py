import numpy as np
import torch

from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    Dictionary
)

class DictionaryWithInvFreqWeight(Dictionary):
    def __init__(self, *args, **kwargs):
        super(DictionaryWithInvFreqWeight, self).__init__(*args, **kwargs)
        self.weights = None # Weights for the losses of imbalanced training data.

    # def finalize(self, *args, **kwargs):
    #     super(DictionaryWithInvFreqWeight, self).finalize(*args, **kwargs)
    #     print(self.symbols[:10])
    #     print(self.count[:10])
    #     exit(1)
    def add_from_file(self, *args, **kwargs):
        '''
        self.weight[i] sould be inversely proportinal to normal_token_counts[i].
        sum(normal_token_counts) / len(normal_token_counts) is a scale factor to make the expected value of the weight be 1.
        '''
        super(DictionaryWithInvFreqWeight, self).add_from_file(*args, **kwargs)
        n_special_tokens = 4
        n_normal_tokens = len([s for s in self.symbols[n_special_tokens:] 
                               if s[:10] != 'madeupword'])
        n_padding_factor = len(self.symbols) - n_special_tokens - n_normal_tokens

        normal_token_counts = self.count[n_special_tokens:n_special_tokens+n_normal_tokens]
        sum_normal_token_counts = sum(normal_token_counts)
        normal_token_weights = [sum_normal_token_counts/(len(normal_token_counts) * c) for c in normal_token_counts] 

        self.weights = [0, 1, 1, 1] + normal_token_weights + [1 for _ in range(n_padding_factor)]
        self.weights = torch.from_numpy(np.array([float(w) for w in self.weights])).float()

