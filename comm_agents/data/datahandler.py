import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd

PATH = 'data/reference_experiment_dat.csv'


class RefExpDataset(Dataset):

    def __init__(self):
        # read with pandas
        df = pd.read_csv(PATH, dtype=np.float32)
        self.n_samples = len(df)

        # define attributes
        self.hidden_states = torch.from_numpy(
            df[['m0', 'm1', 'q0', 'q1']].values)

        self.observations = torch.from_numpy(
            df[[c for c in df.columns if 'o' in c]].values)

        self.questions = torch.from_numpy(
                df[['m_ref_a', 'v_ref_a', 'm_ref_b', 'v_ref_b']].values)

        self.opt_answers = torch.from_numpy(
            df[['alpha_star0', 'alpha_star1',
                'phi_star0', 'phi_star1']].values)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return (self.hidden_states[index], self.observations[index],
                self.questions[index], self.opt_answers[index])

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    # create dataset
    dataset = RefExpDataset()
