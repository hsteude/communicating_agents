import torch
from torch.utils.data import Dataset
import pandas as pd
import glob

TRAINING_DATA_PATH = r'data/training/'


class RefExpDataset(Dataset):

    def __init__(self):
        # read with pandas
        df = self.read_files_to_df()
        self.n_samples = len(df)

        # scale obs with max val of each eperiment and zero
        self.scale_observations(df)

        # define attributes
        self.hidden_states = torch.tensor(
            df[['m0', 'm1', 'q0', 'q1']].values,
            dtype=torch.float32)

        self.observations = torch.tensor(
            df[[c for c in df.columns if 'o' in c]].values,
            dtype=torch.float32)

        self.questions = torch.tensor(
            df[['m_ref_a', 'v_ref_a', 'm_ref_b', 'v_ref_b']].values,
            dtype=torch.float32)

        self.opt_answers = torch.tensor(
            df[['alpha_star0', 'alpha_star1',
                'phi_star0', 'phi_star1']].values,
            dtype=torch.float32)

    def __getitem__(self, index):
        return (self.hidden_states[index], self.observations[index],
                self.questions[index], self.opt_answers[index])

    def __len__(self):
        return self.n_samples

    @staticmethod
    def read_files_to_df():
        all_files = glob.glob(TRAINING_DATA_PATH + "/*.csv")
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)

    def scale_observations(self, df):
        # observation scalling
        for match_str in ['o_a_0', 'o_a_1', 'o_b_0', 'o_b_1']:
            cols = [c for c in df.columns if match_str in c]
            df.loc[:, cols] = df[cols] / df[cols].abs().max().max()
        return df

if __name__ == '__main__':
    # create dataset
    dataset = RefExpDataset()
