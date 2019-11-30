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
DIM_H      = 3    # Hidden layer dimensions
DIM_OUT    = 2    # Output dimension         (2, for XOR output of 0 (index 0) or 1 (index 1) - one hot classification)
LEARN_RATE = 0.1  # Learning rate of NN
EPOCHS     = 1000  # Maximum allowed number of training iterations for NN


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
            sample = data[:2]
            label  = data[2:]

            # print('sample:', sample)
            # print('label :', label)

            output = net(sample.view(-1, 2))      # Run the NN on the value
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


# The actual code (not functions) begins here
test_set  = torch.tensor([[0,0,1,0], [0,1,0,1], [1,0,0,1], [1,1,1,0]], dtype=torch.float32)
train_set = torch.tensor([[0,0,1,0], [0,1,0,1], [1,0,0,1], [1,1,1,0]], dtype=torch.float32)
# np.random.shuffle(train_set) # TODO it is not shuffling the values correctly for some reason

print('\ntest_set:\n', test_set)
print('\ntrain_set:\n', train_set) 


net = Net()
print('\nNet: ', net)

train(net, train_set)
test(net, test_set)


# def print_results():