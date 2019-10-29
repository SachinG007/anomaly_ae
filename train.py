import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, input_features, encoding_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()
        self.encoder = MLP(input_features, encoder_layers, encoding_dim)
        self.decoder = MLP(encoding_dim, decoder_layers, input_features, is_decoder=True)

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_features, encoding_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_features, hidden_size=encoding_dim,
                                num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=encoding_dim,
                                num_layers=1, batch_first=True)
        self.decoder_fc = nn.Linear(encoding_dim, input_features)
        # self.dec_input = torch.zeros()
        # self.encoding_dim = encoding_dim
        # self.decoder = MLP(encoding_dim, decoder_layers, input_features, is_decoder=True)

    def forward(self, x: torch.Tensor):
        # init_h = torch.zeros(1, x.size(1), self.encoding_dim)
        _, (h, c) = self.encoder(x)
        intermediate, _ = self.decoder(torch.zeros(1), (h, c))
        output = []
        for step in intermediate:
            output.append(self.decoder_fc(step))
        return torch.FloatTensor(np.array(output))


def eval(options, model, val_data):
    loss_fn = nn.MSELoss()
    outputs = []
    for sample, label in val_data:
        sample = torch.FloatTensor(sample)
        if options.device == 'gpu':
            sample = sample.to(torch.device('cuda'))
        # sample, label = val_data[i]
        with torch.no_grad():
            # for inp in sample:
            reconstruction = model(sample)
            rec_err = loss_fn(reconstruction, sample).detach().cpu().mean()
        # rec_err /= sample.shape[0]
        outputs.append([rec_err, label])
    outputs = np.array(outputs)
    return outputs

def train(options, model, train_data, val_data):
    data_loader = DataLoader(train_data, batch_size=options.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    num_iters = len(train_data) // options.batch_size
    for epoch in range(options.epochs):
        training_loss = 0.0

        for _, batch_x in enumerate(data_loader):
            if options.device == 'gpu':
                batch_x = batch_x.to(torch.device('cuda'))
            rec_x = model(batch_x)
            loss = loss_fn(input=rec_x, target=batch_x)
            training_loss += loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss /= num_iters
        val_out = eval(options, model, val_data)
        val_loss = val_out[:, 0].mean()
        # val_out[:, 0] = (val_out[:, 0] - val_out[:, 0].min()) / (val_out[:, 0].max() - val_out[:, 0].min())
        print('Epoch {}: Train loss: {}, Validation loss: {}'.format(epoch, training_loss, val_loss))
        auc = roc_auc_score(val_out[:, 1], val_out[:, 0])
        print('AUC: {}'.format(auc))


def load_data(path):
    train_data = np.load(os.path.join(path, 'train.npy'))
    val_data = np.load(os.path.join(path, 'val.npy'), allow_pickle=True)

    return train_data, val_data

def main(options):
    train_data_, val_data_ = load_data(options.data_path)
    train_data = AudioDataset(train_data_)
    val_data = AudioDataset(val_data_)

    model = Autoencoder(train_data.features, options.encoding_dim, [64, 64], [64, 64])
    if options.device == 'gpu':
        model.cuda()
    train(options, model, train_data, val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='.')
    parser.add_argument('-s', '--encoding_dim', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-dev', '--device', type=str, default='cpu')
    options = parser.parse_args()

    main(options)