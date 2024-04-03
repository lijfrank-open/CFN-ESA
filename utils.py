from time import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels, mask=None):
        if mask != None:
            mask_ = mask.view(-1,1)
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = logits
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
        
class MaskFocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(MaskFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels, mask):
        if mask != None:
            mask_ = mask.view(-1,1)
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        if mask != None:
            log_p = logits*mask_
        else:
            log_p = logits
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
        
class Poly1_Cross_Entropy(nn.Module):
    def __init__(self, weight, num_classes=5, epsilon=1.0, size_average=True):
        super(Poly1_Cross_Entropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.size_average = size_average
        self.ce_loss_func = nn.CrossEntropyLoss(weight)
    def forward(self, preds, labels):
        poly1 = torch.sum(F.one_hot(labels, self.num_classes).float() * F.softmax(preds,dim=-1), dim=-1)
        ce_loss = self.ce_loss_func(preds, labels)
        poly1_ce_loss = ce_loss + self.epsilon * (1 - poly1)
        if self.size_average:
            poly1_ce_loss = poly1_ce_loss.mean()
        else:
            poly1_ce_loss = poly1_ce_loss.sum()
        return poly1_ce_loss

class Poly1_Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, epsilon=1.0, size_average=True):
        super(Poly1_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.size_average = size_average
        self.focal_loss_func = Focal_Loss(self.alpha, self.gamma, self.num_classes, self.size_average)
    def forward(self, preds, labels):
        focal_loss = self.focal_loss_func(preds, labels)
        p = torch.sigmoid(preds)
        labels = F.one_hot(labels, self.num_classes)
        poly1 = labels * p + (1 - labels) * (1 - p)
        poly1_focal_loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
        if self.size_average:
            poly1_focal_loss = poly1_focal_loss.mean()
        else:
            poly1_focal_loss = poly1_focal_loss.sum()
        return poly1_focal_loss

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):

        super(Focal_Loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):

        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) 
        preds_softmax = torch.exp(preds_logsoft)    
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
       
        return loss

class MaskFocal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):
        super(MaskFocal_Loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels, mask):
        mask_ = mask.view(-1, 1)
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = preds * mask_
        preds_softmax = torch.exp(preds_logsoft)    
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask=None):
        if mask!=None:
            mask_ = mask.view(-1, 1)
            if type(self.weight) == type(None):
                loss = self.loss(pred * mask_, target) / torch.sum(mask)
            else:
                loss = self.loss(pred * mask_, target) \
                    / torch.sum(self.weight[target] * mask_.squeeze())
        else:
            if type(self.weight) == type(None):
                loss = self.loss(pred, target)
            else:
                loss = self.loss(pred, target) \
                    / torch.sum(self.weight[target])
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')
    def forward(self, pred, target, mask):
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def make_positions(tensor, padding_idx):
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        return int(1e5) 

class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
    def forward(self, input):
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)

class RelativeEmbedding(nn.Module):
    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(int(-seq_len/2), round(seq_len/2 + 1e-5)).to(input.device).long() + self.origin_shift 
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed

class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb
    
if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())