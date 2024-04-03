import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy as np

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True, classify='emotion'):
        '''
        label index mapping = {0:happy, 1:sad, 2:neutral, 3:angry, 4:excited, 5:frustrated}
        '''
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText1, self.videoText2, self.videoText3, self.videoText4,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if classify == 'emotion':
            self.labels = self.videoLabels
        elif classify == 'sentiment':
            sentiment_labels = {}
            for item in self.videoLabels:
                array = []
                for e in self.videoLabels[item]:
                    if e in [1, 3, 5]:
                        array.append(0)
                    elif e == 2:
                        array.append(1)
                    elif e in [0, 4]:
                        array.append(2)
                sentiment_labels[item] = array
            self.labels = sentiment_labels

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText1[vid])),\
               torch.FloatTensor(np.array(self.videoText2[vid])),\
               torch.FloatTensor(np.array(self.videoText3[vid])),\
               torch.FloatTensor(np.array(self.videoText4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid])),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  np.array(self.videoSpeakers[vid])]),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i]) if i<9 else dat[i].tolist() for i in dat]
    
class IEMOCAPDatasetMerge4(Dataset):

    def __init__(self, path, train=True, classify='emotion'):
        '''
        label index mapping = {0:happy, 1:sad, 2:neutral, 3:angry, 4:excited, 5:frustrated}
        '''
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText1, self.videoText2, self.videoText3, self.videoText4,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if classify == 'emotion':
            emotion4_labels = {}
            for item in self.videoLabels:
                array = []
                index_to_remove = 0
                for e in self.videoLabels[item]:
                    if e in [0, 4]:
                        array.append(0)
                    elif e in [1, 5]:
                        array.append(1)
                    elif e == 2:
                        array.append(2)
                    elif e == 3:
                        array.append(3)
                emotion4_labels[item] = array
            self.labels = emotion4_labels

        elif classify == 'sentiment':
            sentiment_labels = {}
            for item in self.videoLabels:
                array = []
                for e in self.videoLabels[item]:
                    if e in [1, 3, 5]:
                        array.append(0)
                    elif e == 2:
                        array.append(1)
                    elif e in [0, 4]:
                        array.append(2)
                sentiment_labels[item] = array
            self.labels = sentiment_labels

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText1[vid])),\
               torch.FloatTensor(np.array(self.videoText2[vid])),\
               torch.FloatTensor(np.array(self.videoText3[vid])),\
               torch.FloatTensor(np.array(self.videoText4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid])),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  np.array(self.videoSpeakers[vid])]),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i]) if i<9 else dat[i].tolist() for i in dat]
    
class MELDDataset(Dataset):

    def __init__(self, path, train=True, classify='emotion'):
        '''
        label index mapping = {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
        '''
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoSentiments, self.videoText1, self.videoText2, self.videoText3, self.videoText4,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if classify == 'emotion':
            self.labels = self.videoLabels
        elif classify == 'sentiment':
            self.labels = self.videoSentiments

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText1[vid])),\
               torch.FloatTensor(np.array(self.videoText2[vid])),\
               torch.FloatTensor(np.array(self.videoText3[vid])),\
               torch.FloatTensor(np.array(self.videoText4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid])),\
               torch.FloatTensor(np.array(self.videoSpeakers[vid])),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i]) if i<9 else dat[i].tolist() for i in dat]