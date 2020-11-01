import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl


class SingleEncModel(pl.LightningModule):
    """Pytorch lightning model for single encoder setup

    Parameters 
    ----------
    observation_size : int
        Number of values per encoder
    lat_space_size : int
        Number of latent neurons
    question_size : int
        Number of values per question
    enc_num_hidden_layers : int
        Number of hidden layers in each encoder
    enc_hidden_size : int
        Number of neurons in each hidden layers in each encoder
    dec_num_hidden_layers : int
        Number of hidden layers in each decoder
    dec_hidden_size : int
        Number of neurons in each hidden layers in each decoder
    num_encoding_agents : int 
        Number of encoding agents
    num_decoding_agents : int
        Number of decoding agents
    learning_rate : float
        Initial learning rate for optimizer (Adam) 
    beta : float
        Weight for filter function loss
    initial_log_var : float
        Initial value for all selection bias model parameter
        for pre-training phase
    pretrain : bool
        True if training starts without adjusting selection bias
    pretrain_loss_thres : float
        Loss value at which selection biases learning starts
    """

    def __init__(self, hparams, **kwargs):
        super().__init__()

        breakpoint()
        self.hparams = hparams
        self.observantion_size = hparams['observation_size']
        self.lat_space_size = hparams['lat_space_size']
        self.question_size = hparams['question_size']
        self.enc_num_hidden_layers = hparams['enc_num_hidden_layers']
        self.enc_hidden_size = hparams['enc_hidden_size']
        self.dec_num_hidden_layers = hparams['dec_num_hidden_layers']
        self.dec_hidden_size = hparams['dec_hidden_size']
        self.num_encoding_agents = hparams['num_encoding_agents']
        self.num_decoding_agents = hparams['num_decoding_agents']
        self.learning_rate = hparams['learning_rate']
        self.beta = hparams['beta']
        self.initial_log_var = hparams['initial_log_var']
        self.pretrain_loss_thres = hparams['pretrain_loss_thres']
        self.pretrain = hparams['pretrain']
        self.current_train_loss = torch.tensor([[100]])

        # Encoding Agent layers
        # TODO: make pretty and generic!
        self.enc1_in, self.enc1_h, self.enc1_out = self._get_encoder_agent()

        # 4 Decoding agents
        # TODO: make pretty and generic!
        self.a0_in, self.a0_h, self.a0_out = self._get_decoder_agent()
        self.a1_in, self.a1_h, self.a1_out = self._get_decoder_agent()
        self.b0_in, self.b0_h, self.b0_out = self._get_decoder_agent()
        self.b1_in, self.b1_h, self.b1_out = self._get_decoder_agent()

        # selection bias, shape: num decoding agents x num latent neurons
        self.selection_bias = nn.Parameter(torch.tensor(
            np.array([self.initial_log_var]*(
                self.lat_space_size*self.num_decoding_agents))
            .reshape(self.num_decoding_agents, self.lat_space_size),
            dtype=torch.float32))

    def _get_encoder_agent(self):
        """Creates one encoder agent

        Returns
        -------
        enc_in : torch.Linear
            Encoder input layer
        enc_h : torch.nn.ModuleList
            Encoder input layers
        enc_out : torch.Linear
            Encoder output layer
        """
        enc_in = nn.Linear(self.observantion_size, self.enc_hidden_size)
        enc_h = nn.ModuleList(
            [nn.Linear(self.enc_hidden_size, self.enc_hidden_size)
             for i in range(self.enc_num_hidden_layers)])
        enc_out = nn.Linear(
            self.enc_hidden_size,
            int(self.lat_space_size / self.num_encoding_agents))
        return enc_in, enc_h, enc_out

    def _get_decoder_agent(self):
        """Creates one decoding agent

        Returns
        -------
        a_in : torch.Linear
            Encoder input layer
        a_h : torch.nn.ModuleList
            Encoder input layers
        a_out : torch.Linear
            Encoder output layer
        """
        a_in = nn.Linear(in_features=self.lat_space_size + self.question_size,
                         out_features=self.dec_hidden_size)
        a_h = nn.ModuleList(
            [nn.Linear(self.dec_hidden_size, self.dec_hidden_size)
             for i in range(self.dec_num_hidden_layers)])
        a_out = nn.Linear(in_features=self.dec_hidden_size,
                          out_features=1)
        return a_in, a_h, a_out

    def loss_function(self, answers, opt_answers, log_vars, beta):
        """Computes the weighted average of the MSE and the filter loss for a batch

        Parameters
        ----------
        answers : torch.tensor
        opt_answers : torch.tensor
        log_vars : torch.tensor
        beta : float

        Returns
        -------
        float
        """
        mse_angle = torch.mean(torch.sum((answers - opt_answers)**2, axis=1))
        filter_loss = torch.mean(-torch.sum(log_vars, axis=1))
        return mse_angle + beta * filter_loss

    def encode(self, observations):
        """Performs the encoding step for a batch (all encoders)

        Parameters
        ----------
        observations : torch.tensor

        Returns
        -------
        torch.tensor
        """
        # TODO: Make generic for more encoders
        # decoder 1
        lat_space_enc1 = torch.tanh(self.enc1_in(observations))
        for e1h in self.enc1_h:
            lat_space_enc1 = torch.relu(e1h(lat_space_enc1))
        lat_space_enc1 = self.enc1_out(lat_space_enc1)

        return lat_space_enc1

    def filter(self, mu, log_var):
        """Implements the filter function (see chap. 4.2 of paper)

        Parameters
        ----------
        mu : torch.tensor
            latent space activations (means)
        log_var : torch.tensor
            selection bias activations

        Returns
        -------
        s : list of tensors
            One tensor for each decoding agent, where noise has been added
            according to the selection_bias parameters.
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn(mu.shape[0], *std.shape, device=self.device)
        s = [mu + std[i, :] * eps[:, i, :] for i in range(std.shape[0])]
        return s

    def decode(self, s0, s1, s2, s3, questions):
        """Performs the decoding step for a batch (all decoders)

        Parameters
        ----------
        s0 : torch.tensor
            filtered version of latent space activations for agent 0
        s1, s2, s3 : torch.tensor
            just like s0 but for the other decoding agents
        question : torch.tensor
            Holding the question values

        Returns
        -------
        torch.tensor
            Holding the answers of all decoding agents
        """
        # TODO: Make pretty and generic!
        # decoding
        a0_in = torch.cat((s0, questions[:, 0:2]), axis=1)
        a0_out = torch.tanh(self.a0_in(a0_in))
        for a0h in self.a0_h:
            a0_out = torch.tanh(a0h(a0_out))
        a0_out = self.a0_out(a0_out)

        a1_in = torch.cat((s1, questions[:, 0:2]), axis=1)
        a1_out = torch.tanh(self.a1_in(a1_in))
        for a1h in self.a1_h:
            a1_out = torch.tanh(a1h(a1_out))
        a1_out = self.a1_out(a1_out)

        b0_in = torch.cat((s2, questions[:, 2:4]), axis=1)
        b0_out = torch.tanh(self.b0_in(b0_in))
        for b0h in self.b0_h:
            b0_out = torch.tanh(b0h(b0_out))
        b0_out = self.b0_out(b0_out)

        b1_in = torch.cat((s3, questions[:, 2:4]), axis=1)
        b1_out = torch.tanh(self.b1_in(b1_in))
        for b1h in self.b1_h:
            b1_out = torch.tanh(b1h(b1_out))
        b1_out = self.b1_out(b1_out)

        return torch.cat((a0_out, a1_out, b0_out, b1_out), axis=1)

    def forward(self, observations):
        """Implements forward pass (decoding only)

        Parameters
        ---------
        observations : torch.tensor
        """
        return self.encode(observations)

    def training_step(self, batch, batch_idx):
        """Implements the training step for one batch"""

        # get data from data loader
        _, observantions, questions, opt_answers = batch

        # compute forward pass
        lat_space = self.encode(observantions)

        # filter
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)

        # decode
        answers = self.decode(s0, s1, s2, s3, questions)

        # set beta to 0 and force selection bias to initial value
        # if within pre-training phase
        if self.pretrain:
            beta = 0
            with torch.no_grad():
                self.selection_bias[:, :] = \
                    torch.empty(*self.selection_bias.shape).fill_(
                            self.initial_log_var)
        # else allow selection_bias training and reset beta
        else:
            beta = self.beta

        self.current_train_loss = self.loss_function(answers, opt_answers,
                                                     self.selection_bias, beta)
        self._log_selection_biases()

    def validation_step(self, batch, batch_idx):
        _, observantions, questions, opt_answers = batch
        lat_space = self.encode(observantions)
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)
        answers = self.decode(s0, s1, s2, s3, questions)
        beta = 0 if self.pretrain else self.beta
        val_loss = self.loss_function(answers, opt_answers,
                                      self.selection_bias, beta)
        self.log('val_loss', val_loss, prog_bar=True)
        self.logger.experiment.add_scalars(
            'loss',
            {'train_loss': self.current_train_loss,
             'val_loss': val_loss},
            global_step=self.global_step)

    def configure_optimizers(self):
        """Configures optimizer"""
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def _log_selection_biases(self):
        """Logs the selection bias for each agent to tensorboard"""
        self.logger.experiment.add_scalars(
            'sel_bias_a0',
            {'lat_neu0': self.selection_bias[0, 0],
             'lat_neu1': self.selection_bias[0, 1],
             'lat_neu2': self.selection_bias[0, 2]},
            global_step=self.global_step)
        self.logger.experiment.add_scalars(
            'sel_bias_a1',
            {'lat_neu0': self.selection_bias[1, 0],
             'lat_neu1': self.selection_bias[1, 1],
             'lat_neu2': self.selection_bias[1, 2]},
            global_step=self.global_step)
        self.logger.experiment.add_scalars(
            'sel_bias_b0',
            {'lat_neu0': self.selection_bias[2, 0],
             'lat_neu1': self.selection_bias[2, 1],
             'lat_neu2': self.selection_bias[2, 2]},
            global_step=self.global_step)

        self.logger.experiment.add_scalars(
            'sel_bias_b1',
            {'lat_neu0': self.selection_bias[3, 0],
             'lat_neu1': self.selection_bias[3, 1],
             'lat_neu2': self.selection_bias[3, 2]},
            global_step=self.global_step)
        return self.current_train_loss
