import torch
from comm_agents.data.data_handler import RefExpDataset
from comm_agents.models.model_single_enc import SingleEncModel
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from datetime import datetime
from comm_agents.utils import plot_learning_curve
from tqdm import tqdm
torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device('cuda:0')
logger.info(f'Training script running on {device}')
# data related params:
# create data loaders
# we also decided not the scale the input data since the ranges are similar
VALIDATION_SPLIT = .01
MODEL_PATH_PRE = f'./models/single_enc_model_pre{str(datetime.now())[:-16]}.pt'
MODEL_PATH = f'./models/single_enc_model_post{str(datetime.now())[:-16]}.pt'
LEARNING_CURVE_FIGURE_PATH = './figures/training/'\
    f'learning_curve_single_env_{str(datetime.now())[:-16]}.html'

# model related params
ENC_NUM_HIDDEN_LAYERS = 3 
ENC_HIDDEN_SIZE = 100
DEC_NUM_HIDDEN_LAYERS = 3
DEC_HIDDEN_SIZE = 100
NUM_DEC_AGENTS = 4
QUESTION_SIZE = 2

# trainng related params
LEARNING_RATE = 0.01
EPOCHS = 50
BATCH_SIZE = 5120*2# 128 #5120 * 2
INITIAL_BETA = 0.0
BETA = .001
SHUFFLE = False
PRETRAIN_LOSS_THRESHOLD = .021

# initialize dataset
logger.info('Loading data set and data loader')
dataset = RefExpDataset(oversample=True)
# send to gpu if available
# BATCH_SIZE = len(dataset)

# Creating data indices for training and validation splits:
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


# Initialize model
model = SingleEncModel(observantion_size=dataset.observations.shape[1],
                       lat_space_size=3,
                       question_size=QUESTION_SIZE,
                       enc_num_hidden_layers=ENC_NUM_HIDDEN_LAYERS,
                       enc_hidden_size=ENC_HIDDEN_SIZE,
                       dec_num_hidden_layers=DEC_NUM_HIDDEN_LAYERS,
                       dec_hidden_size=DEC_HIDDEN_SIZE,
                       num_decoding_agents=NUM_DEC_AGENTS,
                       device=device)

# send to gpu if available
model.to(device)


# # debug model forward pass
answers, lat_spaces, selection_biases = model(
    dataset.observations[0:2].to(device), dataset.questions[0:2].to(device))


# define loss function
def loss_fn(answers, opt_answers, log_vars, beta):
    mse_angle = torch.mean(torch.sum((answers - opt_answers)**2, axis=1))
    filter_loss = torch.mean(-torch.sum(log_vars, axis=1))
    return mse_angle + beta * filter_loss


# Define loss and optimizer
optimizer_adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE*.0001)


# Training loop
train_loss_ls = []
val_loss_ls = []
epoch_ls = []
sel_biases_ls = []
beta = INITIAL_BETA
optimizer = optimizer_adam


for epoch in range(EPOCHS):
    for _, observations, questions, opt_answers in tqdm(train_loader):
        # send data set to gpu if available
        observations, questions, opt_answers = (observations.to(device),
                                                questions.to(device),
                                                opt_answers.to(device))
        # predict = forward pass with our model
        answers, lat_spaces, selection_biases = model(observations,
                                                      questions)

        # loss
        train_loss = loss_fn(answers, opt_answers, selection_biases, beta)

        # calculate gradients = backward pass
        train_loss.backward()

        # update weights
        optimizer.step()

        # zero the gradients after updating
        optimizer.zero_grad()

    # Validation
    with torch.set_grad_enabled(False):
        for hidden_states, observations, questions, opt_answers in val_loader:
            observations, questions, opt_answers = (observations.to(device),
                                                    questions.to(device),
                                                    opt_answers.to(device))

            # predict = forward pass with our model
            answers, lat_spaces, selection_biases = model(observations,
                                                          questions)


            # loss
            val_loss = loss_fn(answers, opt_answers, selection_biases, beta)
    
    if train_loss < PRETRAIN_LOSS_THRESHOLD:
        beta = BETA 
        logger.debug(f'Turning on filter optimization with beta = {beta}')
        torch.save(model.state_dict(), MODEL_PATH_PRE)

        # optimizer = optimizer_sgd

    # save learning stats
    train_loss_ls.append(train_loss)
    val_loss_ls.append(val_loss)
    epoch_ls.append(epoch)
    sel_biases_ls.append(model.selection_bias.detach().cpu().numpy())

    if epoch % 1 == 0:
        logger.debug(f'epoch {epoch+1} of {EPOCHS}, train_loss = {train_loss},'
                     f' val_loss = {val_loss}')

torch.save(model.state_dict(), MODEL_PATH)
# plot_learning_curve(epoch_ls, train_loss_ls, val_loss_ls,
                    # LEARNING_CURVE_FIGURE_PATH)
