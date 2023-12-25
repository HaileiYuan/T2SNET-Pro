import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.nn.utils import weight_norm
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        '''
        Args:
            x: 3D [B, D, T]
        Returns:

        '''
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation),dim=1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation),dim=1)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        '''

        :param x: [B, D, T]
        :return:
        '''
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Args:
            x: [B, T, D]
        Returns:

        '''
        batch_size, T, D = x.shape
        x = x.view(batch_size, T, D).transpose(2, 1)    # [B, D, T]
        # dilated convolution
        x = self.network(x)
        return x.transpose(2, 1)  # [B, T, D]

class Embedding(nn.Module):
    '''
    spatio-temporal embedding
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, in_channel, out_channel):
        super(Embedding, self).__init__()
        self.FC = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, bias=True),
                                nn.ReLU(),
                                nn.Conv1d(out_channel, out_channel, kernel_size=1, padding=0, stride=1, bias=True))
    def forward(self, x):
        '''
        Args:
            x: [B, T, D]
        Returns: [B, T, D]

        '''
        # temporal embedding
        x = self.FC(x.transpose(2, 1))
        return x.transpose(2, 1)


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, D, num_step]
    HT:     [batch_size, D, num_step]
    D:      output dims
    return: [batch_size, D, num_step]
    '''

    def __init__(self, D, bn_decay=0.2):
        super(gatedFusion, self).__init__()

        self.FC_xs = nn.Conv1d(in_channels=D, out_channels=D, kernel_size=1, bias=True)
        self.FC_xt = nn.Conv1d(in_channels=D, out_channels=D, kernel_size=1, bias=True)
        self.FC_h = nn.Sequential(nn.Conv1d(in_channels=D, out_channels=D, kernel_size=1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=D, out_channels=D, kernel_size=1, bias=True))

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H

class T2SNT(nn.Module):
    def __init__(self, energy_in_channel, weather_in_channel, time_in_channel, seq_len, pre_len, emb_size, device=None):
        '''

        :param energy_in_channel:
        :param seq_len:
        :param pre_len:
        :param emb_size:
        :param num_of_target_time_feature:
        :param num_of_graph_feature:
        '''
        super(T2SNT, self).__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.emb_size = emb_size
        self.device =device
        self.timeEmbedding = Embedding(in_channel=time_in_channel, out_channel=emb_size)

        self.conv_1 = nn.Sequential(nn.Conv1d(energy_in_channel, emb_size, kernel_size=1, padding=0, stride=1, bias=True),
                                    nn.ReLU(),
                                    nn.Conv1d(emb_size, emb_size, kernel_size=1, padding=0, stride=1, bias=True))
        self.conv_2 = nn.Sequential(nn.Conv1d(weather_in_channel-1, emb_size, kernel_size=1, padding=0, stride=1, bias=True),
                                    nn.ReLU(),
                                    nn.Conv1d(emb_size, emb_size, kernel_size=1, padding=0, stride=1, bias=True))

        self.energy_tcn = TemporalConvNet(num_inputs=emb_size, num_channels=[emb_size, emb_size, emb_size], kernel_size=3, dropout=0.0)
        self.weather_tcn = TemporalConvNet(num_inputs=emb_size, num_channels=[emb_size, emb_size, emb_size], kernel_size=3, dropout=0.0)

        self.gate_f = gatedFusion(emb_size)

        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1),
            nn.Conv1d(in_channels=emb_size, out_channels=1, kernel_size=1))

    def forward(self, energy_input, weather_input, time_input):
        '''
        :param energy_input:
        :param weather_input:
        :param time_feature:
        :return:
        '''
        time_emb = self.timeEmbedding(time_input)  # [B, T, D]
        energy_output = self.conv_1(energy_input.transpose(2, 1)).transpose(2, 1) # [B, T, D]
        energy_output = self.energy_tcn(energy_output+time_emb) # [B, T, D]

        weather_input = self.conv_2(weather_input.transpose(2, 1)).transpose(2, 1) # [B, T, D]
        weather_output = self.weather_tcn(weather_input+ time_emb) # [B, T, D]
        final_output = self.gate_f(energy_output.transpose(2,1), weather_output.transpose(2,1)).transpose(2,1) # [B, T, D]
        final_output = self.output_layer(final_output[:,-1:].transpose(2,1))
        return torch.squeeze(final_output, dim=2)