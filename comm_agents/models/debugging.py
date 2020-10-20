import torch
from comm_agents.data.data_handler import RefExpDataset
from comm_agents.models.model_single_enc import SingleEncModel
import pandas as pd
import plotly.graph_objects as go


# define loss function
def loss_fn(answers, opt_answers, log_vars, beta):
    mse_angle = torch.mean(torch.sum(torch.abs(answers - opt_answers), axis=1))
    filter_loss = torch.mean(-torch.sum(log_vars, axis=1))
    return mse_angle + beta * filter_loss


MODEL_PATH = './models/single_enc_model_2020-10-19.pt'

dataset = RefExpDataset()
model = SingleEncModel(observantion_size=dataset.observations.shape[1],
                       lat_space_size=3,
                       question_size=2,
                       enc_num_hidden_layers=20,
                       enc_hidden_size=100,
                       dec_num_hidden_layers=20,
                       dec_hidden_size=100,
                       num_decoding_agents=4)
model.load_state_dict(torch.load(MODEL_PATH))
breakpoint()
# model.selection_bias = torch.diag(torch.tensor([-1]*4))

N = 3
answers, lat_spaces, selection_biases = model(dataset.observations[0:N], dataset.questions[0:N])
opt_answers = dataset.opt_answers[0:N]

loss = loss_fn(answers, opt_answers, log_vars=model.selection_bias, beta=1) 
