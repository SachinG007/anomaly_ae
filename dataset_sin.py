import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as utils
from torch.utils.data import Sampler, Dataset
from dataset import  *

def get_data_sin():

    f = 1
    Fs = 1020#samples will  remain consta nt
    sample = 1020
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave / 2
    x = (x - sample/2)/sample
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)
    neg_points = np.array([[-0.25, 0], [0.25, 0],[0, 0.25], [0, -0.25]])
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    y = np.ones((sample+4,1))
    y[sample:sample + 4] = np.zeros((4,1))
    
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    return train_loader

# def get_data_sin_ring():

#     f = 1
#     Fs = 1020#samples will  remain consta nt
#     sample = Fs
#     x = np.arange(sample)
#     wave = np.sin(2 * np.pi * f * x / Fs)
#     wave = wave / 2
#     x = (x - sample/2)/sample
#     x = np.reshape(x,(sample,1))
#     wave = np.reshape(wave,(sample,1))
#     data_matrix = np.concatenate((x, wave), axis = 1)
#     neg_points = np.array([[-0.25, 0], [0.25, 0],[0, 0.25], [0, -0.25]])
#     data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)


#     sample = 1024
#     neg_points = np.random.rand(1024,2) * 2 - 1
#     norm = np.sqrt(np.sum(np.power(neg_points,2), axis = 1))
#     norm = np.expand_dims(norm, axis  = 1)
#     norm = np.concatenate((norm,norm), axis = 1)
#     # radius = np.random.randint(low = 5, high = 8)/10
#     radius = 2
#     neg_points = neg_points *radius / norm
    
#     data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
#     y = np.ones((sample* 2,1))
#     y[sample-4:2 *sample] = np.zeros((sample+4,1))
#     # vis dataset
#     colormap = np.array(['r', 'b'])
#     y = y.astype(int)
#     plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
#     plt.show()
#     plt.savefig("data_vis.png")
#     plt.close()

#     # import pdb;pdb.set_trace()
#     train_dataset = CustomDataset(data_matrix, y)
#     train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=32,
#             shuffle=False)

#     return train_loader

def get_data_sin_random():

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = 1020
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave / 2
    x = (x - sample/2)/sample
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)
    neg_points = np.array([[-0.25, 0], [0.25, 0],[0, 0.25], [0, -0.25]])
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)


    sample = 1024
    neg_points = np.random.rand(1024,2) * 1 - 0.5

    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    y = np.ones((sample* 2,1))
    y[sample-4:2 *sample] = np.zeros((sample+4,1))
    # vis dataset
    colormap = np.array(['r', 'b'])
    y = y.astype(int)
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
    plt.show()
    plt.savefig("data_vis.png")
    plt.close()

    # import pdb;pdb.set_trace()
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    return train_loader

def test_data_2D(mean, std, neg_radius):

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave 
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)

    sample = 1024
    neg_points = np.random.rand(1024,2) * 2 - 1
    norm = np.sqrt(np.sum(np.power(neg_points,2), axis = 1))
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.concatenate((norm,norm), axis = 1)
    radius = neg_radius
    neg_points = neg_points *radius / norm  
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    y = np.zeros((sample* 2,1))
    y[sample:2 *sample] = np.ones((sample,1))
    # vis dataset
    np.save('/mnt/one_class_results/results_2D/data_train.npy', data_matrix)
    np.save('/mnt/one_class_results/results_2D/label_train.npy', y)
    print(np.shape(data_matrix))
    # import pdb;pdb.set_trace()
    data_matrix = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return train_loader


def get_data_sin_ring_2D(args):

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave 
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)

    sample = 1024
    neg_points = np.random.rand(1024,2) * 2 - 1
    norm = np.sqrt(np.sum(np.power(neg_points,2), axis = 1))
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.concatenate((norm,norm), axis = 1)
    radius = 2
    neg_points = neg_points *radius / norm  
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    y = np.ones((sample* 2,1))
    y[sample:2 *sample] = np.zeros((sample,1))
    # vis dataset
    np.save('/mnt/one_class_results/results_2D/data_train.npy', data_matrix)
    np.save('/mnt/one_class_results/results_2D/label_train.npy', y)
    print(np.shape(data_matrix))
    # import pdb;pdb.set_trace()
    mean_ = np.mean(data_matrix)
    std_ = np.std(data_matrix)
    data_matrix = (data_matrix - mean_)/std_
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False)

    return train_loader, mean_, std_


def get_data_sin_ring_3D():

    f = 1
    Fs = 1020#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave / 2
    x = (x - sample/2)/sample
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    z = np.random.rand(sample,1) * 2 - 1
    data_matrix = np.concatenate((x, wave, z), axis = 1)
    neg_points = np.array([[-0.25, 0, 0.1], [0.25, 0, 0.1],[0, 0.25, -0.1], [0, -0.25, -0.1]])
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)


    sample = 1024
    neg_points = np.random.rand(1024,2) * 2 - 1
    norm = np.sqrt(np.sum(np.power(neg_points,2), axis = 1))
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.concatenate((norm,norm), axis = 1)
    # radius = np.random.randint(low = 5, high = 8)/10
    radius = 2
    neg_points = neg_points *radius / norm
    z = np.random.rand(sample,1) * 2 - 1
    neg_points = np.concatenate((neg_points, z), axis = 1)
    
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    print(np.shape(data_matrix))
    y = np.ones((sample* 2,1))
    y[sample-4:2 *sample] = np.zeros((sample+4,1))
    # vis datase
    y = y.astype(int)    
    np.save('data_train.npy', data_matrix)
    np.save('lab_train.npy', y)
    print(np.shape(data_matrix))
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    return train_loader

# get_data_sin_ball()
# def visualize_dataset(X, y):
#     colormap = np.array(['r', 'b'])
#     y = y.astype(int)
#     plt.scatter(X[:, 0], X[:, 1], c=colormap[np.ravel(y)])
#     return plt

##Data in a sphere of Radius R
def get_data_sphere_noneg():
    print("************No Negative Samples*************")
    sample_p = 128 * 16
    data_matrix = np.random.rand(sample_p,3) * 2 - 1
    # norm = np.sqrt(np.sum(np.power(data_matrix,2), axis = 1))
    # norm = np.expand_dims(norm, axis  = 1)
    # norm = np.concatenate((norm,norm,norm), axis = 1)
    # radius = 1
    # data_matrix = data_matrix * radius / norm
    y = np.ones((sample_p,1))
    np.save('results/data_train_noneg.npy', data_matrix)
    np.save('results/lab_train_noneg.npy', y)
    print(np.shape(data_matrix))
    mean_ = np.mean(data_matrix)
    std_ = np.std(data_matrix)
    data_matrix = (data_matrix - mean_)/std_
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return train_loader, mean_, std_


def get_data_sphere_test(mean, std, neg_rad):
    sample_p = 128 * 8
    sample_n = 128 * 8
    data_matrix = np.random.rand(sample_p,3) * 2 - 1

    neg_points = np.random.rand(sample_p,3) * 2 - 1
    norm = np.sqrt(np.sum(np.power(neg_points,2), axis = 1))
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.concatenate((norm,norm,norm), axis = 1)
    radius = neg_rad
    neg_points = neg_points * radius / norm

    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    data_matrix = (data_matrix - mean)/std
    y = np.zeros((sample_p+sample_n,1))
    y[sample_p:sample_p+sample_n] = np.ones((sample_n,1))
    print(np.shape(data_matrix))
    train_dataset = CustomDataset(data_matrix, y)
    test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return test_loader

def get_data_sin_ring_noneg_2D():

    f = 1
    Fs = 2048#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave 
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)

    y = np.ones((sample,1))
    # vis dataset
    np.save('/mnt/one_class_results/results_2D/data_train_noneg.npy', data_matrix)
    np.save('/mnt/one_class_results/results_2D/label_train_noneg.npy', y)
    print(np.shape(data_matrix))
    # import pdb;pdb.set_trace()
    mean_ = np.mean(data_matrix)
    std_ = np.std(data_matrix)
    data_matrix = (data_matrix - mean_)/std_
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return train_loader, mean_, std_

def test_data_2D(mean, std, disp):

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave 
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)

    wave_up = wave + disp
    wave_down = wave - disp
    neg_points_up = np.concatenate((x, wave_up), axis = 1)
    neg_points_down = np.concatenate((x, wave_down), axis = 1) 
    data_matrix = np.concatenate((data_matrix, neg_points_up, neg_points_down), axis = 0)
    y = np.zeros((sample* 3,1))
    y[sample:3 *sample] = np.ones((2*sample,1))
    # vis dataset
    np.save('/mnt/one_class_results/results_2D/data_train.npy', data_matrix)
    np.save('/mnt/one_class_results/results_2D/label_train.npy', y)
    print(np.shape(data_matrix))
    # import pdb;pdb.set_trace()
    data_matrix = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return train_loader

def get_sphere():
    f = 1
    Fs = 2048#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    # import pdb;pdb.set_trace()
    data_matrix = np.random.multivariate_normal(mean=[0,0,0], cov = np.identity(3), size=(sample))
    norm = np.linalg.norm(data_matrix, ord=2, axis=1)
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.repeat(norm, repeats = 3, axis = 1)
    radius = 1
    data_matrix = data_matrix *radius / norm
    compressor = np.random.rand(sample)
    power = np.ones((sample)) * 1/3
    compressor = np.power(compressor, power)
    compressor = np.expand_dims(compressor, axis  = 1)
    compressor = np.repeat(compressor, repeats = 3, axis = 1)
    data_matrix = data_matrix * compressor
    y = np.ones((sample,1))
    # vis dataset
    np.save('results/data_sphere_garbage.npy', data_matrix)
    np.save('results/label_sphere_garbage.npy', y)
    print(np.shape(data_matrix))
    # import pdb;pdb.set_trace()
    mean_ = np.mean(data_matrix)
    std_ = np.std(data_matrix)
    data_matrix = (data_matrix - mean_)/std_
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)

    return train_loader, mean_, std_, data_matrix

def test_data_sphere_cuboid(mean, std, neg_rad):

    f = 1
    Fs = 2048#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    # import pdb;pdb.set_trace()
    data_matrix = np.random.multivariate_normal(mean=[0,0,0], cov = np.identity(3), size=(sample))
    norm = np.linalg.norm(data_matrix, ord=2, axis=1)
    norm = np.expand_dims(norm, axis  = 1)
    norm = np.repeat(norm, repeats = 3, axis = 1)
    radius = 1
    data_matrix = data_matrix *radius / norm
    compressor = np.random.rand(sample)
    power = np.ones((sample)) * 1/3
    compressor = np.power(compressor, power)
    compressor = np.expand_dims(compressor, axis  = 1)
    compressor = np.repeat(compressor, repeats = 3, axis = 1)
    data_matrix = data_matrix * compressor
    
    
    neg_points = np.random.multivariate_normal(mean=[0,0,0], cov = np.identity(3), size=(sample))
    norm = np.linalg.norm(neg_points, ord=2, axis=1)
    norm = np.expand_dims(norm, axis =1)
    norm = np.repeat(norm, repeats = 3, axis = 1)
    radius = neg_rad
    neg_points = neg_points * radius / norm 

    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    y = np.ones((sample*2))
    y[sample:2*sample] = np.zeros((sample))

    data_matrix = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False)
    return train_loader