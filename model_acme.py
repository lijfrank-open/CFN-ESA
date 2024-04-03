import torch
import copy
import torch.nn as nn
from torch.nn import functional as F

from utils import LearnedPositionalEmbedding, RelativeSinusoidalPositionalEmbedding, SinusoidalPositionalEmbedding

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ACME(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(ACME, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, features_t, features_v, features_a, key_padding_mask): 
        for mod in self.layers:
            features_t, features_v, features_a = mod(features_t, features_v, features_a, key_padding_mask)
        return features_t, features_v, features_a
    
class ACMELayer(torch.nn.Module):
    def __init__(self, feature_size, nheads=4, dropout=0.3, no_cuda=False):
        super(ACMELayer, self).__init__()
        self.no_cuda = no_cuda
        self.self_att_t = nn.MultiheadAttention(feature_size, nheads)
        self.self_att_v = nn.MultiheadAttention(feature_size, nheads)
        self.self_att_a = nn.MultiheadAttention(feature_size, nheads)
        self.cross_att_tv = nn.MultiheadAttention(feature_size, nheads)
        self.cross_att_ta = nn.MultiheadAttention(feature_size, nheads)
        self.cross_att_v = nn.MultiheadAttention(feature_size, nheads)
        self.cross_att_a = nn.MultiheadAttention(feature_size, nheads)
        self.dropout_t = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_a = nn.Dropout(dropout)
        self.dropout_t1 = nn.Dropout(dropout)
        self.dropout_t11 = nn.Dropout(dropout)
        self.dropout_t12 = nn.Dropout(dropout)
        self.dropout_v1 = nn.Dropout(dropout)
        self.dropout_a1= nn.Dropout(dropout)
        self.dropout_t2 = nn.Dropout(dropout)
        self.dropout_v2 = nn.Dropout(dropout)
        self.dropout_a2 = nn.Dropout(dropout)
        self.dropout_t3 = nn.Dropout(dropout)
        self.dropout_v3 = nn.Dropout(dropout)
        self.dropout_a3 = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(feature_size)
        self.norm_v = nn.LayerNorm(feature_size)
        self.norm_a = nn.LayerNorm(feature_size)
        self.norm_t1 = nn.LayerNorm(feature_size)
        self.norm_v1 = nn.LayerNorm(feature_size)
        self.norm_a1 = nn.LayerNorm(feature_size)
        self.norm_t2 = nn.LayerNorm(feature_size)
        self.norm_v2 = nn.LayerNorm(feature_size)
        self.norm_a2 = nn.LayerNorm(feature_size)
        self.linear_cat = nn.Linear(2*feature_size, feature_size)
        self.fc_t = nn.Linear(feature_size, 2*feature_size)
        self.fc_v = nn.Linear(feature_size, 2*feature_size)
        self.fc_a = nn.Linear(feature_size, 2*feature_size)
        self.fc_t1 = nn.Linear(2*feature_size, feature_size)
        self.fc_v1= nn.Linear(2*feature_size, feature_size)
        self.fc_a1 = nn.Linear(2*feature_size, feature_size)

    def forward(self, features_t, features_v, features_a, key_padding_mask):
        key_padding_mask = key_padding_mask.transpose(0,1)
        feat_t, feat_v, feat_a = features_t, features_v, features_a
        self_feat_t, self_feat_v, self_feat_a = self._self_att(feat_t, feat_v, feat_a, key_padding_mask)
        self_feat_t, self_feat_v, self_feat_a = self.norm_t(feat_t + self_feat_t), self.norm_v(feat_v + self_feat_v), self.norm_a(feat_a + self_feat_a)

        cross_feat_t, cross_feat_v, cross_feat_a = self._cross_att(self_feat_t, self_feat_v, self_feat_a, key_padding_mask)
        cross_feat_t, cross_feat_v, cross_feat_a = self.norm_t1(feat_t + self_feat_t + cross_feat_t), self.norm_v1(feat_v + self_feat_v + cross_feat_v), self.norm_a1(feat_a + self_feat_a + cross_feat_a)

        full_feat_t, full_feat_v, full_feat_a = self._full_con(cross_feat_t, cross_feat_v, cross_feat_a)
        full_feat_t, full_feat_v, full_feat_a = self.norm_t2(feat_t + cross_feat_t + full_feat_t), self.norm_v2(feat_v + cross_feat_v + full_feat_v), self.norm_a2(feat_a + cross_feat_a + full_feat_a)

        return full_feat_t, full_feat_v, full_feat_a

    def _self_att(self, features_t, features_v, features_a, key_padding_mask):
        feat_t = self.self_att_t(features_t,features_t,features_t,key_padding_mask)[0]
        feat_v = self.self_att_v(features_v,features_v,features_v,key_padding_mask)[0]
        feat_a = self.self_att_a(features_a,features_a,features_a,key_padding_mask)[0]
        return self.dropout_t(feat_t), self.dropout_v(feat_v), self.dropout_a(feat_a)

    def _cross_att(self, features_t, features_v, features_a, key_padding_mask):
        feat_t1 = self.cross_att_tv(features_t,features_v,features_v, key_padding_mask)[0]
        feat_t2 = self.cross_att_ta(features_t,features_a,features_a, key_padding_mask)[0]
        feat_t = torch.concat([self.dropout_t11(feat_t1), self.dropout_t11(feat_t2)], dim=-1)
        feat_t = F.relu(self.linear_cat(feat_t))
        feat_v = self.cross_att_v(features_v,features_t,features_t, key_padding_mask)[0]
        feat_a = self.cross_att_a(features_a,features_t,features_t, key_padding_mask)[0]
        return self.dropout_t1(feat_t), self.dropout_v1(feat_v), self.dropout_a1(feat_a)
    
    def _full_con(self, features_t, features_v, features_a):
        feat_t = self.fc_t1(self.dropout_t2(F.relu(self.fc_t(features_t))))
        feat_v = self.fc_v1(self.dropout_v2(F.relu(self.fc_v(features_v))))
        feat_a = self.fc_a1(self.dropout_a2(F.relu(self.fc_a(features_a))))
        return self.dropout_t3(feat_t), self.dropout_v3(feat_v), self.dropout_a3(feat_a)