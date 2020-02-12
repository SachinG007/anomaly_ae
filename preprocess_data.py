from __future__ import print_function
import os
import numpy as np
import librosa
import argparse
import random
import pdb;
from tqdm import tqdm

def load_generated_data(options):
    normal_data = np.load(os.path.join(options.dataset_path, 'normal_spec.npy'))
    abnormal_data = np.load(os.path.join(options.dataset_path, 'abnormal_spec.npy'))
    len_ab = len(abnormal_data)
    print(len_ab)
    normal_data = np.transpose(normal_data,(0,2,1))
    abnormal_data = np.transpose(abnormal_data,(0,2,1))
    print(normal_data.shape)
    print(abnormal_data.shape)
    np.random.shuffle(normal_data)
    np.random.shuffle(abnormal_data)
    
    normal_train = normal_data[len_ab:, :, :]
    # pdb.set_trace()
    normal_train = normal_train[:, :-3, :].reshape(-1, 64 * 5)
    
    val_data = []
    for sample in abnormal_data:
        val_data.append((sample[:-3, :].reshape(-1, 64 * 5), 0))
    
    for sample in normal_data[:len_ab]:
        val_data.append((sample[:-3, :].reshape(-1, 64 * 5), 1))
    val_data = np.array(val_data)
    print(normal_train.shape, val_data.shape)
    return normal_train, val_data



def extract_features(file_path):
    data, rate = librosa.load(file_path, sr=None, mono=False) # data = [channels, signal]
    data = data[0, :] # Consider single channel
    data = np.asfortranarray(data)
    # features = [n_mel, timesteps]
    features = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=rate, n_mels=64, win_length=1024, hop_length=512))
    features = features.transpose(1, 0) # features = [timesteps, n_mel]
    if options.type != 'non-ts':
        skip_steps = features.shape[0] % 100
        features = features[:-skip_steps, :]
    else: 
        skip_steps = features.shape[0] % 5
        features = np.reshape(features[:-skip_steps, :], (features.shape[0] // 5, 320)) # features = [timesteps // 5, n_mels * 5]
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-t', '--type', type=str)
    options = parser.parse_args()
    train, val = load_generated_data(options)
    # train, val = load_dataset(options)
    np.save(os.path.join(options.output_path, 'trainv6.npy'), train)
    np.save(os.path.join(options.output_path, 'valv6.npy'), val, allow_pickle=True)

    # print(np.load(os.path.join(options.output_path, 'val2.npy'), allow_pickle=True).shape)
    # print(np.load(os.path.join(options.output_path, 'train2.npy')).shape)
