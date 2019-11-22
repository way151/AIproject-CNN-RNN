import sys
import os
import pandas as pd
import string
import numpy as np
import math
import copy
from utils import Logger

from gensim.models import word2vec
import time
import gensim


import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.functional as f


from train import train_epoch
from validation import val_epoch

from temporal_transforms import LoopPadding, TemporalRandomCrop

from layer import GraphConvolution

def extract_words(strr):
    strr = strr.strip(string.punctuation+string.whitespace)
    strr = strr.replace(':',' ').replace('\'',' ').replace('\"',' ').replace(';',' ')
    return strr.split() 


def preprocess(path, run_type):
    labels = []
    sts = []
    data = []
    with open(path,'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            # if not len(line) == 7:
                # print(line)
            labels.append(float(line[4]))
            s1 = extract_words(line[5])
            s2 = extract_words(line[6])
            sts.append([s1,s2])
            data.append(s1)
            data.append(s2)
    s = time.time()

    if run_type == 'train':
       # model = word2vec.Word2Vec(data,size=100,min_count=3)
       # model.save('123.model')
        model = word2vec.Word2Vec.load('100.model')    

    elif run_type == 'validation':
        model = word2vec.Word2Vec.load('100.model')    
    print(time.time()-s)

    dataset = []
    dataset2 = []
    cnt = 0
    sum1 = 0
    sum2 = 0
    labelss = []
    max_tmp = 0
    max_tmp1 = 0
    
    for items in sts: # items = [['1','2'],['3','4']]
        tmp = []
        for word in items[0]:
            if word in model:
                tmp.append(word)
        if len(tmp) == 0:
            cnt += 1
            continue
        sum1 += len(tmp)
        if len(tmp) > max_tmp:
            max_tmp = len(tmp)

        tmp1 = []
        for word in items[1]:
            if word in model:
                tmp1.append(word)
        if len(tmp1) == 0:
            cnt += 1
            continue
        sum2 += len(tmp1)
        if len(tmp1) > max_tmp1:
            max_tmp1 = len(tmp1)

        labelss.append(labels[cnt])
        dataset.append(tmp)
        dataset2.append(tmp1)
        cnt += 1
    
    print('data size:', len(dataset), len(labels))
    print('sts length:', sum1/len(dataset), sum2/len(dataset2))
    print('max length:', max_tmp, max_tmp1)

    return dataset, dataset2, labelss, model # (5479,len,100)

class stsbenchmark(data.Dataset):
    def __init__(self, root_path, file_path, temporal_transform,datatype):
        self.file_path = os.path.join(root_path, file_path)
        self.sts, self.sts2, self.labels, self.w2v_model = preprocess(self.file_path,datatype)
        self.temporal_transform = temporal_transform
        self.word_dict = {}
        self.datatype = datatype
    def set_dict(self, w_dict):
        self.word_dict = w_dict

    def __getitem__(self, index):
        clip = self.sts[index]
        clip2 = self.sts2[index]
        clip = torch.tensor([self.word_dict[w] for w in clip], dtype=torch.long)
        clip2 = torch.tensor([self.word_dict[w] for w in clip2], dtype=torch.long)
        # if self.datatype == 'train':
        #     # clip = np.array(self.sts[index], dtype=float)

        #     # zz = np.zeros((55-clip.shape[0],clip.shape[1]))
        #     # clip = np.concatenate((zz,clip))
            
        #     # clip2 = np.array(self.sts2[index], dtype=float)
        #     # zz = np.zeros((55-clip2.shape[0],clip2.shape[1]))
        #     # clip2 = np.concatenate((zz,clip2))

        #     indices = np.arange(len(clip)).tolist()
        #     indices = self.temporal_transform(indices)
            
        #     indices2 = np.arange(len(clip2)).tolist()
        #     indices2 = self.temporal_transform(indices2)
        #     # print(indices2)
        #     # print(np.array(self.sts2[index], dtype=float)[indices2].shape)
        #     return (clip[indices], clip2[indices2], self.labels[index])
        #     # return (clip, clip2, self.labels[index])
        # elif self.datatype == 'validation':
        pad = torch.ones(55-clip.shape[0], dtype=torch.long)*len(self.word_dict)
        # print(pad.shape)
        clip = torch.cat((pad, clip))
        clip2 = torch.cat((torch.ones(55-clip2.shape[0], dtype=torch.long)*len(self.word_dict), clip2))
        return (clip, clip2, self.labels[index]/5)


    def __len__(self):
        return len(self.labels) 





class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)).cuda()
        self.bias = nn.Parameter(torch.zeros(hidden_size)).cuda()

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)



class lstm(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()# b,128*2,
        self.rnn1 = nn.LSTM(
            input_size = 100,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
            # dropout = 0.5,
            bidirectional = True,
            ).cuda()
        # self.fc1 = nn.Linear(100, 100).cuda()

        # self.rnn2 = nn.LSTM(
        #     input_size = 100,
        #     hidden_size = 32,
        #     num_layers = 2,
        #     batch_first = True,
        #     ).cuda()
        # self.fc2 = nn.Linear(128, 32).cuda()
        # self.fc3 = nn.Linear(128, 1).cuda()
        self.embedding = nn.Embedding.from_pretrained(weights).cuda()
        # print(embedding(torch.LongTensor([word_dict['the']])))
        self.embedding.weight.requires_grad = True
        # self.embedding.weight[len(self.embedding.weight)-1].requires_grad = False
        self.pooling = nn.MaxPool1d(55, stride=2)
        # self.softmax = nn.Softmax()
        # self.conv2 = nn.Conv1d(6, 16, kernel_size=1)
        # self.layers = nn.Sequential(GraphConvolution(),
        #                             GraphConvolution(),
        #                             )
        self.LN = LayerNorm(64*2)

        # self.attention = nn.Parameter(torch.empty(55,1), requires_grad = True).cuda()
        # nn.init.xavier_normal_(self.attention)



    def forward(self, x, x2):
        x = self.embedding(x)
        x2 = self.embedding(x2)
        # x = self.fc1(x)
        # x2 = self.fc1(x2)

        # print(self.embedding.weight)
        # print(x.shape)

        # x = x.float().cuda()
        # x = f.relu(x)
        r_out, (h_n, h_c) = self.rnn1(x)

        r_out = self.LN(r_out)
        # r_out = r_out * torch.exp(-abs(self.attention))
        
        # ot = r_out[:, -1, :]
        # out = torch.mean(r_out, dim=1, keepdim=False)

        out = self.pooling(r_out.permute(0,2,1)).squeeze()

        # out = torch.mean(self.layers(r_out), dim=1)
        

        
        # x2 = x2.float().cuda()
        # x2 = self.fc1(x2)
        # x2 = f.relu(x2)
        r_out, (h_n, h_c) = self.rnn1(x2)
        r_out = self.LN(r_out)
        # r_out = r_out * torch.exp(-abs(self.attention))

        # ot2 = r_out[:, -1, :]


        # out2 = torch.mean(self.layers(r_out), dim=1)
        
        out2 = self.pooling(r_out.permute(0,2,1)).squeeze()
        # out2 = torch.mean(r_out, dim=1, keepdim=False)
        # res = torch.cat((abs(out-out2), out, out2), 1)
        # cos = torch.cosine_similarity(ot, ot2)

        # return torch.exp(-torch.sum(abs(self.fc2(out-out2)),dim=1))
        return torch.exp(-torch.sum(abs(out-out2),dim=1))

        # print(out.shape)
    #bilinear pooling
        # res = (out.unsqueeze(2)*out2.unsqueeze(1)).view(x.shape[0],-1)
        # res = self.fc2(res)
        # res = self.softmax(res)
        # w = torch.arange(6).float().cuda()
        # return torch.sum(res*w, dim=1)


        # return torch.cosine_similarity(out, out2)

if __name__ == '__main__':
    # preprocess('stsbenchmark/sts-train.csv','train')
    # exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(1000)

# for data train:
    temporal_transform = TemporalRandomCrop(16)#todo

    training_data = stsbenchmark('stsbenchmark', 'sts-train.csv', temporal_transform,'train')

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    train_logger = Logger(
        os.path.join('result', 'train.log'),
        ['epoch', 'loss', 'lr','cor'])
    train_batch_logger = Logger(
        os.path.join('result', 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'lr'])


#for data valid:
    temporal_transform = TemporalRandomCrop(16)
    # temporal_transform = LoopPadding(16)

    validation_data = stsbenchmark('stsbenchmark', 'sts-test.csv', temporal_transform, 'validation')
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    val_logger = Logger(
        os.path.join('result', 'val.log'), ['epoch', 'loss', 'cor'])


#for model:
    m = training_data.w2v_model
    weights = torch.zeros([m.wv.vectors.shape[0]+1,m.wv.vectors.shape[1]])

    word_dict = {}
    cnt = 0
    for word in m.wv.index2word:
        word_dict[word] = cnt
        weights[cnt] = torch.FloatTensor(m[word])
        cnt += 1
    weights[cnt] = torch.zeros(len(weights[0]), dtype=torch.float)
    weights = Variable(weights).cuda()
    training_data.set_dict(word_dict)
    validation_data.set_dict(word_dict)
    
    model = lstm(weights)

    criterion = nn.MSELoss().cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
#for optimization:

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001)

# RMSProp
# AdamW
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=0.001,
    #     momentum=0.9,
    #     weight_decay=1e-3,
    # )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=20)

# for resume data (.pth):
    # if opt.resume_path:
    #     print('loading checkpoint {}'.format(opt.resume_path))
    #     checkpoint = torch.load(opt.resume_path)
    #     assert opt.arch == checkpoint['arch']

    #     opt.begin_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if not opt.no_train:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
# running!
    print('run')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    for i in range(1, 220 + 1):
        train_epoch(i, train_loader, model, criterion, optimizer,
                        train_logger, train_batch_logger)
        validation_loss = val_epoch(i, val_loader, model, criterion, val_logger)
        scheduler.step(validation_loss)


