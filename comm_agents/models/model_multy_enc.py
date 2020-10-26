import torch
from torch.nn import functional as F
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torchvision import datasets, transforms
from PIL import Image
import torchvision


class MultyEncModel(pl.LightningModule):
    def __init__(self, observantion_size, lat_space_size, question_size,
                 enc_num_hidden_layers, enc_hidden_size, dec_num_hidden_layers,
                 dec_hidden_size, num_decoding_agents, initial_log_var device):
        super().__init__()

        self.observantion_size = observantion_size
        self.lat_space_size = lat_space_size
        self.question_size = question_size
        self.enc_num_hidden_layers = enc_num_hidden_layers
        self.enc_hidden_size = enc_hidden_size
        self.dec_num_hidden_layers = dec_num_hidden_layers
        self.dec_hidden_size = dec_hidden_size
        self.num_decoding_agents = num_decoding_agents
        self.cuda = device.type != 'cpu'

        # Encoding Angent layers
        self.enc1_in, self.enc1_h, self.enc1_out = self.get_encoder_agent()
        self.enc2_in, self.enc2_h, self.enc2_out = self.get_encoder_agent()

        # 4 Decoding agents
        self.a1_in, self.a1_h, self.a1_out = self.get_decoder_agent()
        self.a2_in, self.a2_h, self.a2_out = self.get_decoder_agent()
        self.b1_in, self.b1_h, self.b1_out = self.get_decoder_agent()
        self.b2_in, self.b2_h, self.b2_out = self.get_decoder_agent()

        self.selection_bias = nn.Parameter(torch.tensor(
            np.array([initial_log_var]*(
                self.lat_space_size*self.num_decoding_agents))
            .reshape(self.num_decoding_agents, self.lat_space_size),
            dtype=torch.float32))

    def get_encoder_agent(self):
        """Write me!"""
        enc_in = nn.Linear(self.observantion_size, self.enc_hidden_size)
        enc_h = nn.ModuleList(
            [nn.Linear(self.enc_hidden_size, self.enc_hidden_size)
             for i in range(self.enc_num_hidden_layers)])
        enc_out = nn.Linear(self.enc_hidden_size, self.lat_space_size)
        return enc_in, enc_h, enc_out

    def get_decoder_agent(self):
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
        lat_space_enc1 = torch.tanh(self.enc1_in(observantion[:, :20]))
        for e1h in self.enc1_h:
            lat_space_enc1 = torch.relu(e1h(lat_space))
        lat_space_enc1 = self.enc1_out(lat_space)

        # decoder 2
        lat_space_enc2 = torch.tanh(self.enc2_in(observantion[:, 20:]))
        for e2h in self.enc2_h:
            lat_space_enc2 = torch.relu(e2h(lat_space_enc2))
        lat_space_enc2 = self.enc2_out(lat_space_enc2)
        return torch.cat((lat_space_enc1, lat_space_enc2), axis=1)

    def filter(self, mu, log_var):
        """Write me!"""
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn(mu.shape[0], *std.shape)
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

        return torch.cat((a1_out, a2_out, b1_out, b2_out)

    def forward(self, observantions, questions):

        # compute forward pass
        lat_space = self.encode(observantions)

        # filter
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)

        # decode
        answers = self.decode(s0, s1, s2, s3, questions)
        return answers, lat_space



    def training_step(self, batch, batch_idx, beta):
        """not mine yet"""
        _, observantions, questions, opt_answers = batch

        # compute forward pass
        lat_space = self.encode(observantions)

        # filter
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)

        # decode
        answers = self.decode(s0, s1, s2, s3, questions)

        loss = self.loss_function(answers, opt_answers, log_vars, beta)

        log = {'train_loss': loss}
        return {'loss':loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        """not mine yet"""
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        val_loss = self.loss_function(x_hat, x, mu, logvar)

        return {'val_loss':val_loss, 'x_hat': x_hat}

    def validation_epoch_end(
            self,
            outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']

        grid = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid, 0)

        log = {'avg_val_loss': val_loss}
        return {'log': log, 'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                transform=transforms.ToTensor()),
            batch_size=self.hparams.batch_size)
        return val_loader


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)

    args = parser.parse_args()

    mem = MultyEncModel(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(mem)
