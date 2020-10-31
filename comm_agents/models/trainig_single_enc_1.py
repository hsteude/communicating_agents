from comm_agents.data.data_handler import RefExpDataset
from comm_agents.models.model_single_enc_1 import SingleEncModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import Callback
from loguru import logger
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

MODEL_PATH_PRE = f'./models/single_enc_model_pretrain_{str(datetime.now())[:-16]}.ckpt'
MODEL_PATH = f'./models/single_enc_model_{str(datetime.now())[:-16]}'

# data loader related
VALIDATION_SPLIT = .01
NUM_DL_WORKERS = 8

# model related params
ENC_NUM_HIDDEN_LAYERS = 10
ENC_HIDDEN_SIZE = 100
DEC_NUM_HIDDEN_LAYERS = 10
DEC_HIDDEN_SIZE = 100
NUM_DEC_AGENTS = 4
NUM_ENC_AGENTS = 1
QUESTION_SIZE = 2

# trainng related params
LEARNING_RATE = 0.001
INITIAL_LOG_VAR = -10
EPOCHS = 1000
BATCH_SIZE = 512*15
INITIAL_BETA = 0.0
BETA = .0001
SHUFFLE = False
PRETRAIN_LOSS_THRES = .0005
PRETRAIN = False
GPUS = 1
BACKEND = None


checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_PATH,
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min')


class Callbacks(Callback):

    def on_init_start(self, trainer):
        logger.debug('Starting to init trainer!')

    def on_epoch_end(self, trainer, model):
        loss = model.current_train_loss
        thres = model.pretrain_loss_thres
        if loss < thres and model.pretrain:
            trainer.model.pretrain = False
            logger.debug('Started fitting selection bias')
            trainer.save_checkpoint(MODEL_PATH_PRE)
            logger.debug(f'Saved pretrained model at: {MODEL_PATH_PRE}')

    def on_train_end(self, trainer, model):
        logger.debug('Training fiished!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
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
                              shuffle=SHUFFLE,
                              num_workers=NUM_DL_WORKERS)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            sampler=val_sampler,
                            shuffle=SHUFFLE,
                            num_workers=NUM_DL_WORKERS)

    args_dct = vars(args)
    args_dct.update(dict(
        observantion_size=int(dataset.observations.shape[1] / NUM_ENC_AGENTS),
        lat_space_size=dataset.hidden_states.shape[1] - 1,
        question_size=QUESTION_SIZE,
        enc_num_hidden_layers=ENC_NUM_HIDDEN_LAYERS,
        enc_hidden_size=ENC_HIDDEN_SIZE,
        dec_num_hidden_layers=DEC_NUM_HIDDEN_LAYERS,
        dec_hidden_size=DEC_HIDDEN_SIZE,
        num_encoding_agents=NUM_ENC_AGENTS,
        num_decoding_agents=NUM_DEC_AGENTS,
        initial_log_var=INITIAL_LOG_VAR,
        beta=BETA,
        max_epochs=EPOCHS,
        pretrain_loss_thres=PRETRAIN_LOSS_THRES,
        pretrain=PRETRAIN,
        gpus=GPUS,
        distributed_backend=BACKEND,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE))

    if PRETRAIN:

        # Initialize model
        logger.debug('Initializing model and trainer')
        model = SingleEncModel(hparams=args_dct)

    else:
        pretrained_model_path = 'models/single_enc_model_pretrain_2020-10-31.ckpt'
        logger.debug(f'Loading pretrained model from {pretrained_model_path}')
        model = SingleEncModel.load_from_checkpoint(
            pretrained_model_path)
        breakpoint()
        model.beta = 0.0002
        model.pretrain = False

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[Callbacks()],
        checkpoint_callback=checkpoint_callback)

    logger.debug('Started fitting model')
    trainer.fit(model, train_loader, val_loader)
