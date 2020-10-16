import torch.nn as nn
import torch
import torch.nn.functional as F


class SingleEncModel(nn.Module):
    def __init__(self, observantion_size, lat_space_size, question_size,
                 num_enc_hidden, num_dec_hidden, num_decoding_agents):
        super(SingleEncModel, self).__init__()

        # Encoding Angent layers
        self.enc11 = nn.Linear(in_features=observantion_size,
                               out_features=num_enc_hidden)
        self.enc12 = nn.Linear(in_features=num_enc_hidden,
                               out_features=lat_space_size)

        # 4 Decoding agents with two layers each
        self.a11 = nn.Linear(in_features=lat_space_size + question_size,
                             out_features=num_dec_hidden)
        self.a12 = nn.Linear(in_features=num_dec_hidden,
                             out_features=1)

        self.a21 = nn.Linear(in_features=lat_space_size + question_size,
                             out_features=num_dec_hidden)
        self.a22 = nn.Linear(in_features=num_dec_hidden,
                             out_features=1)

        self.b11 = nn.Linear(in_features=lat_space_size + question_size,
                             out_features=num_dec_hidden)
        self.b12 = nn.Linear(in_features=num_dec_hidden,
                             out_features=1)

        self.b21 = nn.Linear(in_features=lat_space_size + question_size,
                             out_features=num_dec_hidden)
        self.b22 = nn.Linear(in_features=num_dec_hidden,
                             out_features=1)

        self.selection_bias = torch.randn(lat_space_size, num_decoding_agents,
                                          dtype=torch.float32,
                                          requires_grad=True)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def filter(self, lat_space):
        return self.reparameterize(lat_space, self.selection_bias)

    def forward(self, observantion, questions):
        # encoding
        lat_space = F.relu(self.enc11(observantion))
        lat_space = self.enc12(lat_space)

        # filter
        filt_out = self.filter(lat_space)

        # decoding
        a1_in = torch.cat((filt_out[0, :], questions[0:2]))
        a1_out = F.relu(self.a11(a1_in))
        a1_out = self.a12(a1_out)

        a2_in = torch.cat((filt_out[1, :], questions[0:2]))
        a2_out = F.relu(self.a21(a2_in))
        a2_out = self.a22(a2_out)

        b1_in = torch.cat((filt_out[2, :], questions[2:4]))
        b1_out = F.relu(self.b11(b1_in))
        b1_out = self.b12(b1_out)

        b2_in = torch.cat((filt_out[3, :], questions[2:4]))
        b2_out = F.relu(self.b21(b2_in))
        b2_out = self.b22(b2_out)

        return a1_out, a2_out, b1_out, b2_out, lat_space
