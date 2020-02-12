import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pdb
import torch
import random
import numpy as np
from dataset_sin import *

device = torch.device("cuda")

class Autoencoder(nn.Module):
    """
    Autoencoder consisting of FC Encoder and Decoder
    """
    def __init__(self, input_features, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_features, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_features)

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction

def eval(options, model, test_loader):
    """
    Evaluate model on validation dataset and return reconstruction loss and label
    """
    loss_fn = nn.MSELoss(reduction='none')
    outputs = []
    label_score=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(torch.float)
        sample = data
        # sample = torch.FloatTensor(np.expand_dims(sample, 0))
        if options.device == 'gpu':
            sample = sample.to(device)
        # sample, label = val_data[i]
        # print(sample.size())
        with torch.no_grad():
            # for inp in sample:
            reconstruction = model(sample)
            rec_err = loss_fn(reconstruction, sample)
            rec_err = torch.mean(rec_err, dim = 1)
        # rec_err /= sample.shape[0]
        # pdb.set_trace()
        label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            rec_err.cpu().data.numpy().tolist()))
        
    # Compute AUC
    labels, scores = zip(*label_score)
    labels = 1 - np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    f = open('synthetic_sphere_vs_sphere_ae_devs.txt','a+')
    f.write(" NEG_rad " + str(options.negrad) + " LR " + str(options.lr) + " Enc Dim " + str(options.encoding_dim) + " AUC " + str(test_auc) + "\n")
    f.close()
    print(test_auc)

def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch <= args.epochs:
        lr = args.lr * 0.001
    if epoch <= 0.75*args.epochs:
        lr = args.lr * 0.01
    if epoch <= 0.50*args.epochs:
        lr = args.lr * 0.1
    if epoch <= 0.25*args.epochs:
        lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(options, model, train_loader, test_loader):
    """
    Train model on training dataset and evaluate on validation dataset
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr)
    for epoch in range(options.epochs):
        adjust_learning_rate(optimizer,epoch, options)
        training_loss = 0.0
        rel_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(torch.float)
            batch_x = data
            # import pdb;pdb.set_trace()
            if options.device == 'gpu':
                batch_x = batch_x.to(device)
            # print(batch_x.size())
            batch_x = batch_x.to(torch.float)
            rec_x = model(batch_x)
            loss = loss_fn(input=rec_x, target=batch_x)
            training_loss += loss.mean()
            rel_loss += ((batch_x - rec_x) ** 2).sum() / (batch_x ** 2).sum()
            # pdb.set_trace()
            # print((batch_x ** 2).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss /= (batch_idx + 1)
        rel_loss /= (batch_idx + 1)
        # val_out[:, 0] = (val_out[:, 0] - val_out[:, 0].min()) / (val_out[:, 0].max() - val_out[:, 0].min())
        print('Epoch {}: Train loss: {}, Rel Loss: {}'.format(epoch, training_loss, rel_loss))
    eval(options, model, test_loader)

def main(options):
    # train_loader, mean, std = get_data_sin_ring_noneg_2D() #functionin dataset_sin.py #check the dataloader there
    # test_loader = test_data_2D(mean, std, options.negrad)
    train_loader, mean, std, _ = get_sphere() #functionin dataset_sin.py #check the dataloader there
    test_loader = test_data_sphere_cuboid(mean, std, options.negrad)
    model = Autoencoder(3, options.encoding_dim)
    # model = LSTMAutoencoder(train_data.features, options.encoding_dim, options.batch_size, train_data_.shape[1])
    if options.device == 'gpu':
        model.to(device)
    train(options, model, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--encoding_dim', type=int, default=3)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-nr', '--negrad', type=float, default=1)
    parser.add_argument('-dev', '--device', type=str, default='cpu')
    parser.add_argument('-lr', type=float)
    options = parser.parse_args()

    main(options)
