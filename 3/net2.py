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
import torch           # Pytorch dependenices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random          # Python dependency

BATCH_SZ   = 10    # Batch size
DIM_IN     = 1     # Input dimension          (1, for the value of x)
DIM_H      = 3     # Hidden layer dimensions
DIM_OUT    = 1     # Output dimension         (1, for the value of y = x^2)
LEARN_RATE = 0.1   # Learning rate of NN
EPOCHS     = 100   # Maximum allowed number of training iterations for NN


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_H)  # Input layer
        self.fc2 = nn.Linear(DIM_H, DIM_H)   # Hidden layer
        self.fc3 = nn.Linear(DIM_H, DIM_OUT) # Output layer

    def forward(self, x):
        x = x.view(-1, DIM_IN)
        x = torch.sigmoid(self.fc1(x)) # Forward pass through all the layers
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)            # Don't want to ReLU the last layer
        x = torch.sigmoid(x)
        return x


# Train the neural net
def train(net, train_set):
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    loss = torch.tensor([1])
    epoch = 0

    while epoch <= EPOCHS and loss.item() > 0.01:
        epoch = epoch + 1

        for data in train_set:
            sample = data[:1]
            label  = data[1:]

            # print('sample:', sample)
            # print('label :', label)

            output = net(sample.view(-1, 1))      # Run the NN on the value
            loss   = F.mse_loss(output[0], label) # Calculate how incorrect the NN was

            optimizer.zero_grad() # Start gradient at zero
            loss.backward()       # Backward propagate the loss
            optimizer.step()      # Adjust the weights based on the backprop

            # print('data  :', data)
            # print('output:', output.item())

        if epoch is 0 or (epoch + 1) % 10 is 0:
            print(f'Epoch #{str(epoch+1).ljust(2)} loss: {loss.item()}')

    print(f'Epoch #{str(epoch+1).ljust(2)} loss: {loss.item()}')

# Test the neural net
def test(net, test_set):

    print('\nTest output:')
    with torch.no_grad():
        for data in test_set:
            # print('data:', data)
            sample = data[:2]
            label  = data[2:]

            output = torch.argmax(net(sample.view(-1, DIM_IN)))

            print('Input : ', sample)
            print('Output: ', output.item())
            print()

# Generates a batch_size long tensor with the values
# for x and x^2, with the range for x being [-max_x, max_x]
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


# The actual code (not functions) begins here

train_set = generate_set(100, 25)
test_set  = generate_set(10, 25)
print('\nTraining Set: ', set)

net = Net()
print('\nNet: ', net)

print('\nTesting Set: ', set)

train(net, train_set)
test(net, test_set)


# def print_results():