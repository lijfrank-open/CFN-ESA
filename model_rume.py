import torch
import copy
import torch.nn as nn
from torch.nn import functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RUME(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(RUME, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, features, seq_lengths): 
        for mod in self.layers:
            features = mod(features, seq_lengths)
        return features
    
class RUMELayer(torch.nn.Module):
    def __init__(self, feature_size, dropout=0.3, rnn_type='gru', use_vanilla=False, use_rnnpack=True, no_cuda=False):
        super(RUMELayer, self).__init__()
        self.no_cuda = no_cuda
        self.use_rnnpack = use_rnnpack
        self.use_vanilla = use_vanilla
        if rnn_type=='gru':
            self.rnn = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, bidirectional=True)
        elif rnn_type=='lstm':
            self.rnn = nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=1, bidirectional=True)
        self.linear_rnn = nn.Linear(2*feature_size, feature_size)
        self.drop_rnn = nn.Dropout(dropout)
        self.norm_rnn = nn.LayerNorm(feature_size)

        self.fc = nn.Linear(feature_size, 2*feature_size)
        self.drop_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2*feature_size, feature_size)
        self.drop_fc1 = nn.Dropout(dropout)
        self.norm_fc = nn.LayerNorm(feature_size)

    def forward(self, features, seq_lengths):
        feat = features
        if self.use_vanilla:
            feat_ = self._rnn(feat, seq_lengths)
        else:
            feat_rnn = self._rnn(feat, seq_lengths)
            feat_rnn = self.norm_rnn(feat + feat_rnn)
            feat_fc = self._fc(feat_rnn)
            feat_ = self.norm_fc(feat + feat_rnn + feat_fc)

        return feat_

    def _rnn(self, features, seq_lengths):
        feat = features
        if self.use_rnnpack:
            feat = nn.utils.rnn.pack_padded_sequence(feat, seq_lengths.cpu(), enforce_sorted=False)
            self.rnn.flatten_parameters()
            feat_rnn = self.rnn(feat)[0]
            feat_rnn = nn.utils.rnn.pad_packed_sequence(feat_rnn)[0]
        else:
            feat_rnn = self.rnn(feat)[0]
        feat_rnn = self.linear_rnn(feat_rnn)

        return self.drop_rnn(feat_rnn)

    
    def _fc(self, features):
        feat = features
        feat_fc = self.fc1(self.drop_fc(F.relu(self.fc(feat))))
        
        return self.drop_fc1(feat_fc)