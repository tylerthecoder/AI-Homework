'''
File:    net2.py
Authors: Tyler Tracy, Carson Molder
Class:   CSCE 4613 Artificial Intelligence
Date:    12/12/19

Neural network that implements the function y = x^2.

Input | Expected output
 etc.    etc.
-3      -9
-1.5    -2.25
-1       1
 0       0
 1       1
 1.5     2.25
 3       9
 etc.    etc.
'''

import numpy as np     # Numpy dependency
import math
import torch           # Pytorch dependenices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random          # Python dependency



'''
Constants for the net, training, and testing
'''
TRAIN_BATCH_SZ = 200  # Batch size for train / test sets
TEST_BATCH_SZ  = 10
MAX_X_VALUE    = 50   # Maximum value for x for y = x^2

DIM_IN     = 1     # Input dimension        (1, for the value of x)
DIM_H      = 4     # 1st hidden layer dimension (2nd hidden layer dimension is this, squared)
DIM_OUT    = 1     # Output dimension       (1, for the value of y = x^2)

LEARN_RATE = 0.02  # Learning rate of NN
EPOCHS     = 20    # Maximum allowed number of training iterations for NN



'''
Class definition of the x^2 neural net
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_H)   # Input layer
        self.fc2 = nn.Linear(DIM_H, DIM_H**2)       # Hidden layer
        self.fc3 = nn.Linear(DIM_H**2, DIM_H**2)      # Hidden layer
        self.fc4 = nn.Linear(DIM_H**2, DIM_OUT) # Output layer

    def forward(self, x):
        x = x.view(-1, DIM_IN)
        x = torch.abs(x)
        x = torch.relu(self.fc1(x)) # Forward pass through all the layers
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)             # Don't want to ReLU the last layer
        return x



'''
Train the neural net (with backpropagation)
'''
def train(net, train_set):
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    loss  = torch.tensor([1])
    epoch = 0

    while epoch < EPOCHS:
        epoch = epoch + 1

        for data in train_set:
            sample = data[:1]
            label  = data[1:]

            output = net(sample.view(-1, 1))            # Run the NN on the value
            loss   = F.smooth_l1_loss(output[0], label) # Calculate how incorrect the NN was

            optimizer.zero_grad() # Start gradient at zero
            loss.backward()       # Backward propagate the loss
            optimizer.step()      # Adjust the weights based on the backprop

        print(f'Epoch #{str(epoch).ljust(2)} loss: {round(loss.item(), 3)}')



'''
Basic test of the neural net
'''
def test(net, test_set):

    diffs = []

    with torch.no_grad():
        for data in test_set:

            sample = data[0]
            label  = data[1]
            output = net(sample.view(-1, DIM_IN))
            diff   = abs((output.item() - label.item())/label.item()) * 100
            diffs.append(diff)

            print(f'\ninput  : {round(sample.item(), 3)}')
            print(f'output : {round(output.item(), 3)}')
            print(f'actual : {round(label.item(), 3)}')
            print(f'% diff : {round(diff, 3)}%')
    
    med_diff = round(np.median(diffs), 3)
    avg_diff = round(np.average(diffs), 3)
    print(f'\nmedian  % diff : {med_diff}%')
    print(f'average % diff : {avg_diff}%')



'''
TODO test again and graph using MatPlotLib
'''
# def fancy_test(net):



'''
Generates a batch_size long tensor with the values
for x and x^2, with the range for x being [-max_x, max_x]
'''
def generate_set(batch_size, max_x):
    RAND_STEP = 0.001 # The granularity of the values in the test set
    set = torch.zeros(batch_size, 2, dtype=torch.float32)

    for i in range (0, batch_size):
        # Get a random value for x in the range [-max_x, max_x]
        x = float(random.randrange(max_x * -1 / RAND_STEP, max_x / RAND_STEP))
        x = x * RAND_STEP

        set[i][0] = x
        set[i][1] = x**2
    return set



'''
Actual code (not functions) begins here
'''
train_set = generate_set(TRAIN_BATCH_SZ, MAX_X_VALUE)
test_set  = generate_set(TEST_BATCH_SZ, MAX_X_VALUE)

net = Net()

print('\nNet: ', net)
print('\nTraining Set: ', train_set)
print('\nTesting Set: ', test_set)
print()

print('\n--- Before training: ---')
test(net, test_set)

print("\n--- Training now: ---\n")
train(net, train_set)

print("\n--- After training: ---")
test(net, test_set)
# fancy_test(net)

