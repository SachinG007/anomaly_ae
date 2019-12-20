from __future__ import print_function
import os
import numpy as np
import librosa
import argparse
import random
# from tqdm import tqdm

def load_generated_data(options):
    normal_data = np.load(os.path.join(options.dataset_path, 'idct_spec_slider_00_0.005thresh_normal.npy'))
    abnormal_data = np.load(os.path.join(options.dataset_path, 'idct_spec_slider_00_0.005thresh_abnormal.npy'))
    print(normal_data.shape)
    print(abnormal_data.shape)
    np.random.shuffle(normal_data)
    np.random.shuffle(abnormal_data)
    
    normal_train = normal_data[119:, :, :]
    normal_train = normal_train[:, :-8, :].reshape(-1, 512 * 15)
    
    val_data = []
    for sample in abnormal_data:
        val_data.append((sample[:-8, :].reshape(-1, 512 * 15), 1))
    
    for sample in normal_data[:119]:
        val_data.append((sample[:-8, :].reshape(-1, 512 * 15), 0))
    val_data = np.array(val_data)
    print(normal_train.shape, val_data.shape)
    return normal_train, val_data

def load_dataset(options):
    normal_path = os.path.join(options.dataset_path, 'normal')
    abnormal_path = os.path.join(options.dataset_path, 'abnormal')
    normal_files = os.listdir(normal_path)
    abnormal_files = os.listdir(abnormal_path)

    normal_val = random.sample(normal_files, int(len(abnormal_files)))
    normal_train = list(set(normal_files) - set(normal_val))

    train_data = []
    print('Featurzing training data:')
    if options.type == 'non-ts':        
        for train_file in tqdm(normal_train):
            train_data.extend(extract_features(os.path.join(normal_path, train_file)))
        train_data = np.array(train_data)
        train_data = np.reshape(train_data, (-1, 64 * 5))
    else: 
        for train_file in tqdm(normal_train):
            features = extract_features(os.path.join(normal_path, train_file))
            # print(features.shape)
            features = np.split(features, features.shape[0] // 100)
            train_data.extend(features)
        train_data = np.array(train_data)
    
    val_data = []
    print('Featurizing validation data:')
    for val_file in tqdm(normal_val):
        val_data.append((extract_features(os.path.join(normal_path, val_file)), 0))
    for val_file in tqdm(abnormal_files):
        val_data.append((extract_features(os.path.join(abnormal_path,val_file)), 1))
    val_data = np.array(val_data)

    return train_data, val_data


# def load_dataset_time_series(path):
#     normal_path = os.path.join(path, 'normal')
#     abnormal_path = os.path.join(path, 'abnormal')
#     normal_files = os.listdir(normal_path)
#     abnormal_files = os.listdir(abnormal_path)

#     normal_val = random.sample(normal_files, int(len(abnormal_files)))
#     normal_train = list(set(normal_files) - set(normal_val))

#     train_data = []
#     print('Featurzing training data:')
#     for train_file in tqdm(normal_train):
#         train_data.append(extract_features(os.path.join(normal_path, train_file)))
#     train_data = np.array(train_data)
    
#     val_data = []
#     print('Featurizing validation data:')
#     for val_file in tqdm(normal_val):
#         val_data.append((extract_features(os.path.join(normal_path, val_file)), 0))
#     for val_file in tqdm(abnormal_files):
#         val_data.append((extract_features(os.path.join(abnormal_path,val_file)), 1))
#     val_data = np.array(val_data)

#     return train_data, val_data


# def extract_features_time_series(file_path):
#     data, rate = librosa.load(file_path, sr=None, mono=False) # data = [channels, signal]
#     data = data[0, :] # Consider single channel
#     data = np.asfortranarray(data)
#     # features = [n_mel, timesteps]
#     features = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=rate, n_mels=64, win_length=1024, hop_length=512))
#     features = features.transpose(1, 0) # features = [timesteps, n_mel]
#     return features


def extract_features(file_path):
    data, rate = librosa.load(file_path, sr=None, mono=False) # data = [channels, signal]
    data = data[0, :] # Consider single channel
    data = np.asfortranarray(data)
    # features = [n_mel, timesteps]
    features = librosa.feature.melspectrogram(y=data, sr=rate, n_mels=64, win_length=1024, hop_length=512)
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
    np.save(os.path.join(options.output_path, 'train.npy'), train)
    np.save(os.path.join(options.output_path, 'val.npy'), val, allow_pickle=True)

    print(np.load(os.path.join(options.output_path, 'val.npy'), allow_pickle=True).shape)
    print(np.load(os.path.join(options.output_path, 'train.npy')).shape)
