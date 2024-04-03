import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import pickle as pk
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss, FocalLoss, MaskedNLLLoss
from model import CFNESA
from sklearn.metrics import confusion_matrix, classification_report
from trainer import train_or_eval_model, seed_everything
from dataloader import IEMOCAPDataset, MELDDataset
from torch.utils.data import DataLoader
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
parser.add_argument('--classify', default='emotion', help='sentiment, emotion')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')
parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')
parser.add_argument('--modals', default='avl', help='modals')
parser.add_argument('--Dataset', default='MELD', help='dataset to train and test.MELD/IEMOCAP')
parser.add_argument('--textfeature_mode', default='textf1', help='concat4/sum4/concat2/sum2/textf1/textf2/textf3/textf4')

parser.add_argument('--rnn_n_layers', type=int, default=2, help='rnn_n_layers')
parser.add_argument('--rnn_drop', type=float, default=0.15, metavar='dropout', help='dropout rate')
parser.add_argument('--rnn_type', default='gru', help='gru/lstm')
parser.add_argument('--use_vanilla', action='store_true', default=False, help='does not use vanilla')
parser.add_argument('--use_rnnpack', action='store_true', default=False, help='does not use rnnpack')

parser.add_argument('--cross_hidden_dim', type=int, default=384, help='cross_hidden_dim')
parser.add_argument('--cross_n_layers', type=int, default=3, help='cross_n_layers')
parser.add_argument('--cross_num_head', type=int, default=8, help='cross_num_head')
parser.add_argument('--cross_drop', type=float, default=0.3, metavar='dropout', help='dropout rate')

parser.add_argument('--shift_output_dim', type=int, default=128, help='shift_output_dim')
parser.add_argument('--shift_drop', type=float, default=0.05, metavar='dropout', help='dropout rate')
parser.add_argument('--shift_type', default='concat', help='sub/abs_sub/concat/sub_concat/mul')

parser.add_argument('--loss_type', default='sum_class_shift_loss', help='auto_loss/sum_class_shift_loss/class_loss')
parser.add_argument('--lambd', nargs='+', type=float, default=0.9, help='lambd of shift_cl_loss')
args = parser.parse_args()

save_path = 'save_model/{}/cross_{}_shift_{}.pth'.format(
    args.Dataset,
    args.cross_n_layers,
    args.shift_type)

MELD_path = '/home/lijfrank/code/dataset/MELD_features/meld_multi_features_mmgcn_cosmic.pkl' 
IEMOCAP_path = '/home/lijfrank/code/dataset/IEMOCAP_features/iemocap_multi_features_mmgcn_cosmic.pkl'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12361'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def cleanup():
    dist.destroy_process_group()

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])

def get_MELD_loaders(path, batch_size=32, classify='emotion', valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path, classify=classify)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(path, train=False, classify=classify)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(path, batch_size=32, classify='emotion', valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path, classify=classify)

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path, train=False, classify=classify)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def main(rank, world_size):
    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    today = datetime.datetime.now()
    name_ =args.modals+'_'+args.Dataset
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals 
    if args.Dataset=='IEMOCAP':
        embedding_dims = [1024, 342, 1582] 
    elif args.Dataset=='MELD':
        embedding_dims  = [1024, 342, 300] 
    if args.classify == 'emotion':
        if args.Dataset=='MELD':
            n_classes  = 7 
        elif args.Dataset=='IEMOCAP':
            n_classes  = 6 
    elif args.classify == 'sentiment':
        n_classes  = 3
    seed_everything()
    model = CFNESA(args=args, embedding_dims=embedding_dims, n_classes=n_classes)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    if args.classify == 'emotion':
        if args.Dataset == 'MELD':
            loss_function = FocalLoss()

            loss_weights = torch.FloatTensor([1.0/0.469507, 
                                            1.0/0.119346, 
                                            1.0/0.026116, 
                                            1.0/0.073096, 
                                            1.0/0.168369, 
                                            1.0/0.026335, 
                                            1.0/0.117231])
            
            shift_loss_weights = torch.FloatTensor([1/(2*(1-0.5047)), 1/(2*0.5047)]).cuda()
            loss_function_shift = MaskedNLLLoss(shift_loss_weights)

        elif args.Dataset == 'IEMOCAP':
            loss_weights = torch.FloatTensor([1/0.086747,
                                            1/0.144406,
                                            1/0.227883,
                                            1/0.160585,
                                            1/0.127711,
                                            1/0.252668])
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            shift_loss_weights = torch.FloatTensor([1/(2*(1-0.4364)), 1/(2*0.4364)]).cuda()
            loss_function_shift = MaskedNLLLoss(shift_loss_weights)

    elif args.classify == 'sentiment':
        loss_function = MaskedNLLLoss()
        loss_function_shift = MaskedNLLLoss()

    if args.loss_type=='auto_loss':
        awl = AutomaticWeightedLoss(2)
        optimizer = optim.AdamW([
                    {'params': model.parameters()},
                    {'params': awl.parameters(), 'weight_decay': 0}], lr=args.lr, weight_decay=args.l2, amsgrad=True)
    else:    
        optimizer = optim.AdamW(model.parameters() , lr=args.lr, weight_decay=args.l2, amsgrad=True)

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(path=MELD_path, valid=0.1,
                                                                    batch_size=batch_size,
                                                                    classify=args.classify,
                                                                    num_workers=0,
                                                                    pin_memory=False)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(path=IEMOCAP_path, valid=0.1,
                                                                      batch_size=batch_size,
                                                                      classify=args.classify,
                                                                      num_workers=0,
                                                                      pin_memory=False)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_probs = None
    all_fscore, all_acc, all_loss = [], [], []
    all_fscoreshift, all_accshift = [], []

    for e in range(n_epochs):
        if args.Dataset == 'MELD':
            trainset = MELDDataset(MELD_path)
        elif args.Dataset == 'IEMOCAP':
            trainset = IEMOCAPDataset(IEMOCAP_path)
        train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid=0.1)
        train_sampler.set_epoch(e)
        valid_sampler.set_epoch(e)

        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _, _, _, _, train_accshift, train_fscoreshift, train_initial_feats,train_probs = train_or_eval_model(model, loss_function, loss_function_shift, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset, loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
        valid_loss, valid_acc, _, _, _, valid_fscore, _, _, _, _, _, valid_accshift, valid_fscoreshift, valid_initial_feats, valid_probs = train_or_eval_model(model, loss_function,loss_function_shift, valid_loader, e, cuda, args.modals, dataset=args.Dataset,loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}'.\
            format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore))
        
        if rank == 0:
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, _, _, _, _, _, test_accshift, test_fscoreshift, test_initial_feats, test_probs = train_or_eval_model(model, loss_function, loss_function_shift, test_loader, e, cuda, args.modals, dataset=args.Dataset, loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
            all_fscore.append(test_fscore)
            all_acc.append(test_acc)
            all_fscoreshift.append(test_fscoreshift)
            all_accshift.append(test_accshift)
            print('test_loss: {}, test_acc: {}, test_fscore: {}, test_accsh: {}, test_fscoresh: {}, total time: {} sec, {}'.\
                    format(test_loss, test_acc, test_fscore, test_accshift, test_fscoreshift, round(time.time()-start_time, 2), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
            print('-'*100)
            if best_fscore == None or best_fscore < test_fscore: 
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask

                best_probs = test_probs

                if (e+1)%10 == 0: 
                    torch.save(model.state_dict(), save_path)
                    print("epoch: {}, save the model successfully".format(e+1))
                    print('-'*150)
            if (e+1)%10 == 0:
                np.set_printoptions(suppress=True)
                print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
                print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
                print('-'*100)
                            
        dist.barrier()
        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
    
    if args.tensorboard:
        writer.close()
    if rank == 0:
        print('Test performance..')
        print ('Acc: {}, F-Score: {}'.format(max(all_acc), max(all_fscore)))
        if not os.path.exists("results/record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("results/record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump({}, f)
        with open("results/record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore))
        else:
            record[key_] = [max(all_fscore)]
        if record.get(key_+'record', False):
            record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
        else:
            record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0)]
        with open("results/record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4, zero_division=0))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
        
    cleanup()

if __name__ == '__main__':
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    run_demo(main, n_gpus)