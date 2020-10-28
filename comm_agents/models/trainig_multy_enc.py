from comm_agents.data.data_handler import RefExpDataset
from comm_agents.models.model_multy_enc import MultyEncModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from loguru import logger

# we also decided not the scale the input data since the ranges are similar
VALIDATION_SPLIT = .01

# model related params
ENC_NUM_HIDDEN_LAYERS = 10
ENC_HIDDEN_SIZE = 100
DEC_NUM_HIDDEN_LAYERS = 10
DEC_HIDDEN_SIZE = 100
NUM_DEC_AGENTS = 4
NUM_ENC_AGENTS = 2
QUESTION_SIZE = 2

# trainng related params
LEARNING_RATE = 0.0001
INITIAL_LOG_VAR = -10
EPOCHS = 200
BATCH_SIZE = 521
INITIAL_BETA = 0.0
BETA = .0001
SHUFFLE = False
PRETRAIN_LOSS_THRESHOLD = .014
PRETRAIN = True


def train():
    breakpoint()
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--beta', default=1e-4, type=float)
    # parser.add_argument('--max_epochs', default=2, type=float)
    args = parser.parse_args()

    logger.debug('Loading data set')
    dataset = RefExpDataset(oversample=True)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler,
                              shuffle=SHUFFLE)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            sampler=val_sampler,
                            shuffle=SHUFFLE)

    args_dct = vars(args)
    args_dct.update(dict(
        observantion_size=int(dataset.observations.shape[1]/2), # make nicer!
        lat_space_size=dataset.hidden_states.shape[1],
        question_size=QUESTION_SIZE,
        enc_num_hidden_layers=ENC_NUM_HIDDEN_LAYERS,
        enc_hidden_size=ENC_HIDDEN_SIZE,
        dec_num_hidden_layers=DEC_NUM_HIDDEN_LAYERS,
        dec_hidden_size=DEC_HIDDEN_SIZE,
        num_encoding_agents=NUM_ENC_AGENTS,
        num_decoding_agents=NUM_DEC_AGENTS,
        initial_log_var=INITIAL_LOG_VAR,
        beta=BETA))

    # Initialize model
    model = MultyEncModel(**args_dct)

    breakpoint()
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=2)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    train()
