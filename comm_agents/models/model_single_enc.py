import torch.nn as nn
import torch
import numpy as np


class SingleEncModel(nn.Module):
    def __init__(self, observantion_size, lat_space_size, question_size,
                 enc_num_hidden_layers, enc_hidden_size, dec_num_hidden_layers,
                 dec_hidden_size, num_decoding_agents, device):
        super(SingleEncModel, self).__init__()
        

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

        # 4 Decoding agents
        self.a1_in, self.a1_h, self.a1_out = self.get_decoder_agent()
        self.a2_in, self.a2_h, self.a2_out = self.get_decoder_agent()
        self.b1_in, self.b1_h, self.b1_out = self.get_decoder_agent()
        self.b2_in, self.b2_h, self.b2_out = self.get_decoder_agent()

        self.selection_bias = nn.Parameter(torch.tensor(
            np.array([-10.0]*(self.lat_space_size*self.num_decoding_agents))
            .reshape(self.num_decoding_agents, self.lat_space_size),
            dtype=torch.float32
        ))

    def get_encoder_agent(self):
        enc_in = nn.Linear(self.observantion_size, self.enc_hidden_size)
        enc_h = nn.ModuleList(
            [nn.Linear(self.enc_hidden_size, self.enc_hidden_size)
             for i in range(self.enc_num_hidden_layers)])
        enc_out = nn.Linear(self.enc_hidden_size, self.lat_space_size)
        return enc_in, enc_h, enc_out

    def get_decoder_agent(self):
        a_in = nn.Linear(in_features=self.lat_space_size + self.question_size,
                         out_features=self.dec_hidden_size)
        a_h = nn.ModuleList(
            [nn.Linear(self.dec_hidden_size, self.dec_hidden_size)
             for i in range(self.dec_num_hidden_layers)])
        a_out = nn.Linear(in_features=self.dec_hidden_size,
                          out_features=1)
        return a_in, a_h, a_out

    def filter(self, mu, log_var):
        """
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        if self.cuda:
            eps_shape = torch.Size((mu.shape[0], *std.shape))
            eps = torch.cuda.FloatTensor(eps_shape)
            torch.randn(eps_shape, out=eps)
        else:
            eps = torch.randn(mu.shape[0], *std.shape)

        s = [mu + std[i, :] * eps[:, i, :] for i in range(std.shape[0])]
        return s 

    def forward(self, observantion, questions):
        # encoding
        lat_space = torch.tanh(self.enc1_in(observantion))
        for e1h in self.enc1_h:
            lat_space = torch.relu(e1h(lat_space))
        lat_space = self.enc1_out(lat_space)

        # filter
        s0, s1, s2, s3 = self.filter(lat_space, self.selection_bias)

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

        return torch.cat((a1_out, a2_out, b1_out, b2_out), axis=1), lat_space, self.selection_bias
