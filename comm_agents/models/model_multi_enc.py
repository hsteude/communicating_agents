import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl


class MultiEncModel(pl.LightningModule):
    """Write me!"""

    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.hparams = hparams
        self.observantion_size = hparams['observantion_size']
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

        # Encoding Angent layers
        self.enc1_in, self.enc1_h, self.enc1_out = self._get_encoder_agent()
        self.enc2_in, self.enc2_h, self.enc2_out = self._get_encoder_agent()

        # 4 Decoding agents
        self.a1_in, self.a1_h, self.a1_out = self._get_decoder_agent()
        self.a2_in, self.a2_h, self.a2_out = self._get_decoder_agent()
        self.b1_in, self.b1_h, self.b1_out = self._get_decoder_agent()
        self.b2_in, self.b2_h, self.b2_out = self._get_decoder_agent()

        self.selection_bias = nn.Parameter(torch.tensor(
            np.array([self.initial_log_var]*(
                self.lat_space_size*self.num_decoding_agents))
            .reshape(self.num_decoding_agents, self.lat_space_size),
            dtype=torch.float32))

    def _get_encoder_agent(self):
        """Write me!"""
        enc_in = nn.Linear(self.observantion_size, self.enc_hidden_size)
        enc_h = nn.ModuleList(
            [nn.Linear(self.enc_hidden_size, self.enc_hidden_size)
             for i in range(self.enc_num_hidden_layers)])
        enc_out = nn.Linear(
            self.enc_hidden_size,
            int(self.lat_space_size / self.num_encoding_agents))
        return enc_in, enc_h, enc_out

    def _get_decoder_agent(self):
        """Write me!"""
        a_in = nn.Linear(in_features=self.lat_space_size + self.question_size,
                         out_features=self.dec_hidden_size)
        a_h = nn.ModuleList(
            [nn.Linear(self.dec_hidden_size, self.dec_hidden_size)
             for i in range(self.dec_num_hidden_layers)])
        a_out = nn.Linear(in_features=self.dec_hidden_size,
                          out_features=1)
        return a_in, a_h, a_out

    def loss_function(self, answers, opt_answers, log_vars, beta):
        """Write me!"""
        mse_angle = torch.mean(torch.sum((answers - opt_answers)**2, axis=1))
        filter_loss = torch.mean(-torch.sum(log_vars, axis=1))
        return mse_angle + beta * filter_loss

    def encode(self, observantions):
        """Write me!"""
        # decoder 1
        lat_space_enc1 = torch.tanh(self.enc1_in(observantions[:, :20]))
        for e1h in self.enc1_h:
            lat_space_enc1 = torch.relu(e1h(lat_space_enc1))
        lat_space_enc1 = self.enc1_out(lat_space_enc1)

        # decoder 2
        lat_space_enc2 = torch.tanh(self.enc2_in(observantions[:, 20:]))
        for e2h in self.enc2_h:
            lat_space_enc2 = torch.relu(e2h(lat_space_enc2))
        lat_space_enc2 = self.enc2_out(lat_space_enc2)
        return torch.cat((lat_space_enc1, lat_space_enc2), axis=1)

    def filter(self, mu, log_var):
        """Write me!"""
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn(mu.shape[0], *std.shape, device=self.device)
        s = [mu + std[i, :] * eps[:, i, :] for i in range(std.shape[0])]
        return s

    def decode(self, s0, s1, s2, s3, questions):
        # decoding
        a1_in = torch.cat((s0, questions[:, 0:2]), axis=1)
        a1_out = torch.tanh(self.a1_in(a1_in))
        for a1h in self.a1_h:
            a1_out = torch.tanh(a1h(a1_out))
        a1_out = self.a1_out(a1_out)

        a2_in = torch.cat((s1, questions[:, 0:2]), axis=1)
        a2_out = torch.tanh(self.a2_in(a2_in))
        for a2h in self.a2_h:
            a2_out = torch.tanh(a2h(a2_out))
        a2_out = self.a2_out(a2_out)

        b1_in = torch.cat((s2, questions[:, 2:4]), axis=1)
        b1_out = torch.tanh(self.b1_in(b1_in))
        for b1h in self.b1_h:
            b1_out = torch.tanh(b1h(b1_out))
        b1_out = self.b1_out(b1_out)

        b2_in = torch.cat((s3, questions[:, 2:4]), axis=1)
        b2_out = torch.tanh(self.b2_in(b2_in))
        for b2h in self.b2_h:
            b2_out = torch.tanh(b2h(b2_out))
        b2_out = self.b2_out(b2_out)

        return torch.cat((a1_out, a2_out, b1_out, b2_out), axis=1)

    def forward(self, observantions):
        return self.encode(observantions)

    def training_step(self, batch, batch_idx):
        """not mine yet"""
        _, observantions, questions, opt_answers = batch

        # compute forward pass
        lat_space = self.encode(observantions)

        # filter
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)

        # decode
        answers = self.decode(s0, s1, s2, s3, questions)

        beta = 0 if self.pretrain else self.beta
        self.current_train_loss = self.loss_function(answers, opt_answers,
                                                     self.selection_bias, beta)
        return self.current_train_loss

    def validation_step(self, batch, batch_idx):
        _, observantions, questions, opt_answers = batch
        lat_space = self.encode(observantions)
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)
        answers = self.decode(s0, s1, s2, s3, questions)
        beta = 0 if self.pretrain else self.beta
        loss = self.loss_function(answers, opt_answers,
                                  self.selection_bias, beta)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)
