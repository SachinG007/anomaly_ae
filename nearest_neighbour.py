import numpy as np
import os
import librosa
import argparse
import random
from tqdm import tqdm

SAMPLE_RATE = 16000
LABEL_NORMAL = 0
LABEL_ANOMALOUS = 1

def load_dataset(options):
    normal_path = os.path.join(options.dataset_path, 'normal')
    abnormal_path = os.path.join(options.dataset_path, 'abnormal')
    normal_files = os.listdir(normal_path)
    abnormal_files = os.listdir(abnormal_path)

    normal_data = []
    abnormal_data = []
    print('Featurzing data:')
    
    for train_file in tqdm(normal_files):
        features = extract_features(os.path.join(normal_path, train_file))
        normal_data.append(features)        
        
    for train_file in tqdm(abnormal_files):
        features = extract_features(os.path.join(normal_path, train_file))
        abnormal_data.append(features)

    return normal_data, abnormal_data

def extract_features(file_path):
    data, rate = librosa.load(file_path, sr=None, mono=False) # data = [channels, signal]
    data = data[0, :] # Consider single channel
    data = np.asfortranarray(data)
    # features = [n_mel, timesteps]
    features = librosa.feature.melspectrogram(y=data, sr=rate, n_mels=64, win_length=400, n_fft=512, hop_length=512)
    features = features.transpose(1, 0) # features = [timesteps, n_mel]
    return features


def get_average_distance(set1, set2, same=False):
    print(len(set1), len(set2))
    dist = 0
    dist2 = 0
    for point1 in set1: 
        dists = []
        dists2 = []
        for point2 in set2:
            dists.append(np.sqrt(((point1 - point2) ** 2).sum()))
            dists2.append(((point1 - point2) ** 2).sum())
        dists.sort()
        dists2.sort()
        if same:
            dists = dists[1:]
            dists2 = dists2[1:]
        dist += sum(dists[:5]) / 5
        dist2 += sum(dists2[:5]) / 5
    return dist/len(set1), dist2 / len(set1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str)
    options = parser.parse_args()

    normal, abnormal = load_dataset(options)

    print(get_average_distance(abnormal, normal))
    print(get_average_distance(normal, normal, True))
    