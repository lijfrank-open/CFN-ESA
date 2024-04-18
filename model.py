from model_acme import ACME, ACMELayer
from shift import EmoShift
import torch.nn as nn
import torch
from torch.nn import functional as F

from model_rume import RUME, RUMELayer

class CFNESA(nn.Module):
    def __init__(self, args, embedding_dims, n_classes):
        super(CFNESA, self).__init__()
        self.textfeature_mode = args.textfeature_mode

        if args.textfeature_mode == 'concat4':
            self.linear_t = nn.Linear(4*embedding_dims[0], args.cross_hidden_dim)
        elif args.textfeature_mode == 'concat2':
            self.linear_t = nn.Linear(2*embedding_dims[0], args.cross_hidden_dim)
        else:
            self.linear_t = nn.Linear(embedding_dims[0], args.cross_hidden_dim)
        self.linear_v = nn.Linear(embedding_dims[1], args.cross_hidden_dim)
        self.linear_a = nn.Linear(embedding_dims[2], args.cross_hidden_dim)

        rumeLayer = RUMELayer(feature_size=args.cross_hidden_dim, dropout=args.rnn_drop, rnn_type=args.rnn_type, use_vanilla=args.use_vanilla, use_rnnpack=args.use_rnnpack, no_cuda=args.no_cuda)
        self.rume = RUME(rumeLayer,num_layers=args.rnn_n_layers)

        acmeLayer = ACMELayer(feature_size=args.cross_hidden_dim, nheads=args.cross_num_head,dropout=args.cross_drop,no_cuda=args.no_cuda)
        self.acme = ACME(acmeLayer, num_layers=args.cross_n_layers)

        self.linear_cat = nn.Linear(3*args.cross_hidden_dim, args.cross_hidden_dim)
        self.drop_cat = nn.Dropout(args.cross_drop)

        self.emotion_shift = EmoShift(d_model=args.cross_hidden_dim,output_dim=args.shift_output_dim,dropout=args.shift_drop)
        
        self.smax_fc = nn.Linear(args.cross_hidden_dim, n_classes)
        
    def forward(self, features_t1, features_t2, features_t3, features_t4, features_v, features_a, umask, qmask, seq_lengths):
        if self.textfeature_mode == 'concat4':
            features_t = torch.cat([features_t1, features_t2, features_t3, features_t4], dim=-1)
        elif self.textfeature_mode == 'sum4':
            features_t = features_t1 + features_t2 + features_t3 + features_t4
        elif self.textfeature_mode == 'concat2':
            features_t = torch.cat([features_t1, features_t2], dim=-1)
        elif self.textfeature_mode == 'sum2':
            features_t = features_t1 + features_t2
        elif self.textfeature_mode == 'textf1':
            features_t = features_t1
        elif self.textfeature_mode == 'textf2':
            features_t = features_t2
        elif self.textfeature_mode == 'textf3':
            features_t = features_t3
        elif self.textfeature_mode == 'textf4':
            features_t = features_t4
        featlinear_t = self.linear_t(features_t)
        featlinear_v,featlinear_a = self.linear_v(features_v),self.linear_a(features_a)
        
        featsingle_t,featsingle_v,featsingle_a = self.rume(featlinear_t, seq_lengths),self.rume(featlinear_v, seq_lengths),self.rume(featlinear_a, seq_lengths)
        
        featcross_t,featcross_v,featcross_a = self.acme(featsingle_t,featsingle_v,featsingle_a,umask)
        featcross_cat = torch.concat([featcross_t,featcross_v,featcross_a], dim=-1)
        featcross = self.drop_cat(F.relu(self.linear_cat(featcross_cat)))

        featcross_tcmp,featcross_vcmp,featcross_acmp = self.acme(featsingle_t,featsingle_v,featsingle_a,umask)
        featcross_cmp_cat = torch.concat([featcross_tcmp,featcross_vcmp,featcross_acmp], dim=-1)
        featcross_cmp = self.drop_cat(F.relu(self.linear_cat(featcross_cmp_cat)))
        logitshift = self.emotion_shift(featcross, umask, qmask, featcross_cmp)

        logit = self.smax_fc(featcross)

        return logit, logitshift