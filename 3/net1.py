'''
File:    net1.py
Authors: Tyler Tracy, Carson Molder
Class:   CSCE 4613 Artificial Intelligence
Date:    12/12/19

Neural network that implements the XOR function.

A  |  B  |  Expected
0     0     0
0     1     1
1     0     1
1     1     0
'''

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Xor():
    # first two values in each entry are inputs, third
    # is the expected output (label)
    training_data = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]

    def make_training_data(self):
        np.random.shuffle(self.training_data) # Shuffle data to make training position-agnostic
        print(self.training_data)

x = Xor()
x.make_training_data()

# class Net(nn.Module):

# def train(net):

# def test(net):

# def print_results():