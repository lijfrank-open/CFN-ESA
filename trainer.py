import os
import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from utils import AutomaticWeightedLoss

seed = 2023
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_or_eval_model(model, loss_function, loss_function_shift, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP',loss_type='',lambd=0,epochs=100):
    losses, preds, labels, masks = [], [], [], []
    predshifts, labelshifts, maskshifts = [], [], []
    scores, vids = [], []
    initial_feats, probs = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths0 = []
        for j, umask_ in enumerate(umask.transpose(0,1)):
            lengths0.append((umask.transpose(0,1)[j] == 1).nonzero()[-1][0] + 1)
        seq_lengths = torch.stack(lengths0)
        logit, logitshift = model(textf1, textf2, textf3, textf4, visuf, acouf, umask, qmask, seq_lengths)
        prob = F.log_softmax(logit,-1)
        prob_ = prob.view(-1, prob.size()[-1])
        label_ = label.view(-1)
        loss_ = loss_function(prob_, label_, umask)
        _, _, batch_size, n_cls = logitshift.shape 
        probshift = F.log_softmax(logitshift,-1)
        probshift_ = probshift.view(-1, n_cls)
        labelshift = (label[None, :, :] != label[:, None, :]).long()
        labelshift_ = labelshift.view(-1)
        umaskshift_cat = umask[None, :, :].long() & umask[:, None, :].long()
        umaskshift = umaskshift_cat.view(-1, batch_size)
        loss_shift = loss_function_shift(probshift_, labelshift_, umaskshift)
        if loss_type=='auto_loss':
            awl = AutomaticWeightedLoss(2)
            loss = awl(loss_, loss_shift)
 
        elif loss_type=='sum_class_shift_loss':
            loss = loss_+ lambd*loss_shift

        elif loss_type=='class_loss':
            loss = loss_
        else:
            NotImplementedError

        preds.append(torch.argmax(prob_, 1).cpu().numpy())
        labels.append(label_.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        predshifts.append(torch.argmax(probshift_, 1).cpu().numpy())
        labelshifts.append(labelshift_.cpu().numpy())
        maskshifts.append(umaskshift.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()
        if epoch == 0: 
            initial_feature = torch.cat([textf1,visuf,acouf],dim=-1)
            initial_feature_ = initial_feature.view(-1, initial_feature.size()[-1])
            initial_feats.append(initial_feature_.cpu().numpy())
        probs.append(prob_.cpu().detach().numpy())

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        if epoch == 0:
            initial_feats = np.concatenate(initial_feats)
        probs = np.concatenate(probs)
        predshifts  = np.concatenate(predshifts)
        labelshifts = np.concatenate(labelshifts)
        maskshifts  = np.concatenate(maskshifts)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    labelshifts = np.array(labelshifts)
    predshifts = np.array(predshifts)

    vids = np.array(vids)
    if epoch == 0:
        initial_feats = np.array(initial_feats)
    probs = np.array(probs)
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    avg_shiftaccuracy = round(accuracy_score(labelshifts,predshifts, sample_weight=maskshifts)*100, 2)
    avg_shiftfscore = round(f1_score(labelshifts,predshifts, sample_weight=maskshifts, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, vids, ei, et, en, el, avg_shiftaccuracy, avg_shiftfscore, initial_feats, probs