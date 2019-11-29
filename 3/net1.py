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

import numpy as np     # Numpy dependency
import torch           # Pytorch dependenices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SZ   = 4    # Batch size
DIM_IN     = 2    # Input dimension          (2, for two XOR inputs)
DIM_H      = 4    # Hidden layers dimensions (4 neurons per layer)
DIM_OUT    = 1    # Output dimension         (1, for XOR output in (0,1))
LEARN_RATE = 1e-3 # Learning rate of NN
EPOCHS     = 3    # Number of training iterations for NN


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_H)  # Input layer
        self.fc2 = nn.Linear(DIM_H, DIM_H)   # Hidden layer 1
        self.fc3 = nn.Linear(DIM_H, DIM_H)   # Hidden layer 2
        self.fc4 = nn.Linear(DIM_H, DIM_OUT) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Forward pass through all the layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)         # Don't want to ReLU the last layer
        return F.log_softmax(x, dim=1)


# Train the neural net
def train(net, train_set):
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)

    for epoch in range(EPOCHS):

        print(f'\nEpoch #{epoch}')
        for data in train_set:
            sample = data[:2]
            label  = data[2:]

            net.zero_grad()                      # Start gradient at zero
            output = net(sample.view(-1, 2))     # Run the NN on the value
            loss   = F.l1_loss(output[0], label) # Calculate how incorrect the NN was
            print(loss)
                             # TODO this is not working?
            loss.backward()  # Backward propagate the loss
            optimizer.step() # Adjust the weights based on the backprop

            print('data  :', data)
            print('output:', output.item())
        print(f'Epoch #{epoch} loss:', loss.item())

# Test the neural net
def test(net, test_set):

    print('\nTest output:')
    with torch.no_grad():
        for data in test_set:
            print('data:', data)
            sample = data[:2]
            label  = data[2:]

            output = net(sample.view(-1, DIM_IN))

            print('Input : ', sample)
            print('Output: ', output)
            print()


# The actual code (not functions) begins here
test_set  = torch.tensor([(0,0,0), (0,1,1), (1,0,1), (1,1,0)], dtype=torch.float32)
train_set = torch.tensor([(0,0,0), (0,1,1), (1,0,1), (1,1,0)], dtype=torch.float32)
np.random.shuffle(train_set) # TODO it is not shuffling the first entry for some reason

print('\ntest_set:\n', test_set)
print('\ntrain_set:\n', train_set) 


net = Net()
print('\nNet: ', net)

train(net, train_set)
test(net, test_set)


# def print_results():