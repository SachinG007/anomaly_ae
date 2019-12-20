import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pdb

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def features(self):
        return self.data.shape[-1]


class MLP(nn.Module):
    """
    NN consisting of fully connected layers with ReLU activation
    """
    def __init__(self, input_dim, hidden_layers, output_dim, is_decoder=False):
        super(MLP, self).__init__()
        layers = []
        hidden_layers = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(hidden_layers[:-1])):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        if is_decoder:
            layers = layers[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    """
    Autoencoder consisting of FC Encoder and Decoder
    """
    def __init__(self, input_features, encoding_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()
        self.encoder = MLP(input_features, encoder_layers, encoding_dim)
        self.decoder = MLP(encoding_dim, decoder_layers, input_features, is_decoder=True)

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_features, encoding_dim, batch_size, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_features, hidden_size=encoding_dim,
                                num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=encoding_dim,
                                num_layers=1, batch_first=True)
        self.decoder_fc = nn.Linear(encoding_dim, input_features)
        # self.encoding_dim = encoding_dim
        # self.decoder = MLP(encoding_dim, decoder_layers, input_features, is_decoder=True)

    def forward(self, x):
        _, (h, c) = self.encoder(x)

        self.dec_input = torch.zeros(x.size(0), x.size(1), 1, device=torch.device('cuda'))

        intermediate, _ = self.decoder(self.dec_input, (h, c))
        intermediate = torch.transpose(intermediate, 0, 1)
        intermediate = torch.add(intermediate, torch.randn_like(intermediate))
        outputs = torch.zeros_like(x)
        for i, step in enumerate(intermediate):
            outputs[:, i, :] = self.decoder_fc(step)

        return outputs


def eval(options, model, val_data):
    """
    Evaluate model on validation dataset and return reconstruction loss and label
    """
    loss_fn = nn.MSELoss()
    outputs = []
    for sample, label in val_data:
        sample = torch.FloatTensor(sample)
        # sample = torch.FloatTensor(np.expand_dims(sample, 0))
        if options.device == 'gpu':
            sample = sample.to(torch.device('cuda:3'))
        # sample, label = val_data[i]
        # print(sample.size())
        with torch.no_grad():
            # for inp in sample:
            reconstruction = model(sample)
            rec_err = loss_fn(reconstruction, sample).detach().cpu().mean()
        # rec_err /= sample.shape[0]
        outputs.append([rec_err, label])
    outputs = np.array(outputs)
    return outputs

def train(options, model, train_data, val_data):
    """
    Train model on training dataset and evaluate on validation dataset
    """
    data_loader = DataLoader(train_data, batch_size=options.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_iters = len(train_data) // options.batch_size
    for epoch in range(options.epochs):
        training_loss = 0.0
        rel_loss = 0.0
        for _, batch_x in enumerate(data_loader):
            if options.device == 'gpu':
                batch_x = batch_x.to(torch.device('cuda:3'))
            # print(batch_x.size())
            rec_x = model(batch_x)
            loss = loss_fn(input=rec_x, target=batch_x)
            training_loss += loss.mean()
            rel_loss += ((batch_x - rec_x) ** 2).sum() / (batch_x ** 2).sum()
            # pdb.set_trace()
            # print((batch_x ** 2).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss /= num_iters
        rel_loss /= num_iters
        val_out = eval(options, model, val_data)
        val_loss = val_out[:, 0].mean()
        # val_out[:, 0] = (val_out[:, 0] - val_out[:, 0].min()) / (val_out[:, 0].max() - val_out[:, 0].min())
        print('Epoch {}: Train loss: {}, Rel Loss: {}, Validation loss: {}'.format(epoch, training_loss, rel_loss, val_loss))
        auc = roc_auc_score(val_out[:, 1], val_out[:, 0])
        print('AUC: {}'.format(auc))


def load_data(path):
    train_data = np.load(os.path.join(path, 'train.npy'))
    val_data = np.load(os.path.join(path, 'val.npy'), allow_pickle=True)
    mean = np.mean(train_data)
    std = np.std(train_data)
    # mean = np.mean(np.reshape(train_data, [-1, train_data.shape[-1]]), axis=1)
    # std = np.std(np.reshape(train_data, [-1, train_data.shape[-1]]), axis=1)
    train_data = (train_data - mean) / std
    # pdb.set_trace()
    for i, (sample, _) in enumerate(val_data):
        val_data[i][0] = (sample - mean) / std    
    # val_data[:, 0] =  (val_data[:, 0] - mean) / std
    print('Train: ', train_data.shape)
    print('Validation: ', val_data.shape)
    return train_data, val_data

def main(options):
    train_data_, val_data_ = load_data(options.data_path)
    train_data = AudioDataset(train_data_)
    val_data = AudioDataset(val_data_)
    model = Autoencoder(train_data.features, options.encoding_dim, [1024, 512, 128], [128, 512, 1024])

    # model = LSTMAutoencoder(train_data.features, options.encoding_dim, options.batch_size, train_data_.shape[1])
    if options.device == 'gpu':
        model.cuda(3)
    train(options, model, train_data, val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='.')
    parser.add_argument('-s', '--encoding_dim', type=int, default=128)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-dev', '--device', type=str, default='cpu')
    options = parser.parse_args()

    main(options)
