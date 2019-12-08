'''
File:    net1.py
Authors: Tyler Tracy, Carson Molder
Class:   CSCE 4613 Artificial Intelligence
Date:    12/8/19

Neural network that implements the XOR function.

A  |  B  |  Expected
0     0     0
0     1     1
1     0     1
1     1     0
'''

import numpy as np              # Numpy dependency
import torch                    # Pytorch dependenices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt # MatPlotLib dependency
import itertools                # Python dependency



'''
Constants for the net, training, and testing
'''
BATCH_SZ   = 4    # Batch size

DIM_IN     = 2    # Input dimension          (2, for two XOR inputs)
DIM_H      = 16   # Hidden layer dimensions
DIM_OUT    = 2    # Output dimension         (2, for XOR output of 0 (index 0) or 1 (index 1) - one hot classification)

LEARN_RATE  = 0.02  # Learning rate of NN
CUTOFF_LOSS = 0.01  # During training, if loss reaches at or below this value, stop training
EPOCHS      = 1000  # Maximum allowed number of training iterations for NN



'''
Class definition of the XOR neural net
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_H)  # Input layer
        self.fc2 = nn.Linear(DIM_H, DIM_H)   # Hidden layer
        self.fc3 = nn.Linear(DIM_H, DIM_OUT) # Output layer

    def forward(self, x):
        x = x.view(-1, DIM_IN)
        x = F.relu(self.fc1(x)) # Forward pass through all the layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)            # Don't want to ReLU the last layer
        return x



'''
Train the neural net (with backpropagation)
'''
def train(net, train_set):
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    loss = torch.tensor([1])
    epoch = 0

    while epoch < EPOCHS and loss.item() > CUTOFF_LOSS:
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

        print(f'Epoch #{str(epoch).ljust(2)} loss: {round(loss.item(), 3)}')



'''
Basic test of the neural net
'''
def test(net, test_set):

    with torch.no_grad():
        for data in test_set:
            # print('data:', data)
            sample = data[:2]
            label  = data[2:]

            output = torch.argmax(net(sample.view(-1, DIM_IN)))

            print('Input : ', sample)
            print('Output: ', output.item())
            print()



'''
Generates a batch_size long tensor with the values
for A and B between (0,1), and a label based on
rounding A and B to the nearest whole number.
'''
def generate_orderly_set(step = 0.01):
    xvals = np.arange(0, 1 + step, step)
    yvals = np.arange(0, 1 + step, step)
    points = torch.Tensor(list(itertools.product(xvals, yvals)))
    return points



'''
TODO description
'''
def fancy_test(net, title):

    test_set = generate_orderly_set()
    results = []

    with torch.no_grad():
        for data in test_set:
            sample = data[:2]
            label  = data[2:]

            output = torch.argmax(net(sample.view(-1, DIM_IN)))
            results.append([round(sample[0].item(), 5), round(sample[1].item(), 5), output.item()])

    return results



'''
TODO description
'''
def fancy_plot(before_results, after_results):

    plt.figure()

    # Plot the test results before training
    plt.subplot(1, 2, 1)
    plt.scatter(x = [item[0] for item in before_results], y = [item[1] for item in before_results], c = [item[2] for item in before_results], cmap = 'winter')
    plt.title('Before trianing')
    plt.xlabel('A')
    plt.ylabel('B')

    plt.axhline(0.5)
    plt.axvline(0.5)

    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.xlim([0,1])

    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.ylim([0,1])

    # Plot the test results after training
    plt.subplot(1, 2, 2)
    plt.scatter(x = [item[0] for item in after_results], y = [item[1] for item in after_results], c = [item[2] for item in after_results], cmap = 'winter')
    plt.title('After training')
    plt.xlabel('A')
    plt.ylabel('B')

    plt.axhline(0.5)
    plt.axvline(0.5)

    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.xlim([0,1])

    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.ylim([0,1])

    plt.show()



'''
Actual code (not functions) begins here
'''
test_set  = torch.tensor([[0,0,1,0], [0,1,0,1], [1,0,0,1], [1,1,1,0]], dtype=torch.float32)
train_set = torch.tensor([[0,0,1,0], [0,1,0,1], [1,0,0,1], [1,1,1,0]], dtype=torch.float32)

net = Net()

print('\nNet: ', net)
print('\nTraining Set:\n', train_set) 
print('\nTesting Set:\n', test_set)

print("\n--- Before training: ---")
before_results = fancy_test(net, 'Before training')

print("\n--- Training now: ---\n")
train(net, train_set)

print("\n--- After training: ---")
after_results = fancy_test(net, 'After training')

fancy_plot(before_results, after_results)