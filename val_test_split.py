from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb


for idx in range(0,7,2):
    print(idx)
    val_data = np.load('valve/valv' + str(idx) + '.npy', allow_pickle=True)
    tot_samp = len(val_data)
    split = int(tot_samp/4)
    val_split_neg = val_data[:split]
    test_split_neg = val_data[split:2*split]
    val_split_pos = val_data[-2*split:-split]
    test_split_pos = val_data[-split:]

    val_data = np.concatenate((val_split_neg, val_split_pos), axis = 0)
    test_data = np.concatenate((test_split_neg, test_split_pos), axis = 0)

    np.save('valve/val_split' + str(idx) + '.npy', val_data)
    np.save('valve/test_split' + str(idx) + '.npy', test_data)