import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob

TRAINING_DATA_PATH = r'data/training/'


class RefExpDataset(Dataset):
    """Pytorch data set holding hidden states, observations, questions and answers

    Attributes
    ----------
    hidden_states : torch.tensor (Nx4)
        Holds: m0, m1, q0 and q1
    observations : torch.tensor (Nx40)
        Holds: o_a_0_0..9, o_a_1_0..9, o_b_0_0..9, o_b_1_0..9,
    questions : torch.tensor (Nx4)
        Holds: m_ref_a, v_ref_a, m_ref_b, v_ref_b
    opt_answers : torch.tensor (Nx4)
        Holds: alpha_star0, alpha_star1, phi_star0, phi_star1
        """

    def __init__(self, oversample=True):
        # read with pandas
        df = self.read_files_to_df()

        # scale obs with max val of each experiment and zero
        df = self.scale_observations(df)

        if oversample:
            df = self.oversample(
                df=df,
                cols=['alpha_star0', 'phi_star0'],
                ranges=[(0, .25*np.pi), (.5 * np.pi, .75 * np.pi)],
                nbins=10)
            df = self.oversample(
                df=df,
                cols=['alpha_star1', 'phi_star1'],
                ranges=[(0, .25*np.pi), (.5 * np.pi, .75 * np.pi)],
                nbins=10)

        self.n_samples = len(df)

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
        """Read csv file with samples into pandas data frame

        Returns
        -------
        pd.DataFrame
        """
        all_files = glob.glob(TRAINING_DATA_PATH + "/*.csv")
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)

    def scale_observations(self, df):
        """Scale observations with the abs max val of each experiment and zero

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        for match_str in ['o_a_0', 'o_a_1', 'o_b_0', 'o_b_1']:
            cols = [c for c in df.columns if match_str in c]
            df.loc[:, cols] = df[cols] / df[cols].abs().max().max()
        return df

    @staticmethod
    def oversample(df, cols, ranges, nbins, frac=1):
        """Performs frequency based oversampling

        The distribution of the target variables is highly skewed. This is why
        we oversample those based on the inverse frequency distribution.

        Parameters
        ----------
        df : pd.DataFrame
            df holding data set
        cols : list of strings
            Column names holding the target variables
        ranges : list of tuples
            Holding the min an max value of the target variable (for histogram)
        nbins : int
            Number of bins of the histogram
        frac : float
            fraction of samples. If 1, the distribution of the target variable
            will get close to uniform
        """
        edges_col0 = np.linspace(*ranges[0], nbins)
        edges_col1 = np.linspace(*ranges[1], nbins)
        H, ec0, ec1 = np.histogram2d(df[cols[0]].values,
                                     df[cols[1]].values,
                                     bins=(edges_col0, edges_col1))
        sample_factors = (H.max() - H) * frac  # don't over do it --> .5

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                cond = df[cols[0]].between(ec0[i], ec0[i+1])\
                    & df[cols[1]].between(ec1[j], ec1[j+1])
                if not df[cond].empty:
                    df_over = df[cond].sample(int(sample_factors[i, j]),
                                              replace=True)
                    df = pd.concat([df, df_over],
                                   axis=0, ignore_index=True)
        return df


if __name__ == '__main__':
    # create dataset
    dataset = RefExpDataset()
