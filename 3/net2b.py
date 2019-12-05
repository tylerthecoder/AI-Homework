'''
File:    net2.py
Authors: Tyler Tracy, Carson Molder
Class:   CSCE 4613 Artificial Intelligence
Date:    12/12/19

Neural network that implements the function y = x^2.
If y < x^2, the output is negative (1H index 0)
If y = x^2, the output is zero     (1H index 1)
If y > x^2, the output is positive (1H index 2)

Input x | Input y | Output
 etc.     etc.
-3       -9         0
-3       -8         1
-3       -10       -1
-1        1         0
 1        10        1
 1        0        -1
 etc.     etc.
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
TRAIN_BATCH_SZ = 20   # Batch size for train / test sets
TEST_BATCH_SZ  = 30
MAX_X_VALUE    = 50   # Maximum value for x for y = x^2
DELTA_Y_VALUE  = 1000 # Maximum divergence from x for y
MIN_Y_DELTA    = 10   # y is at least this far away from x

DIM_IN    = 2   # Input dimension            (1, for the value of x)
DIM_H1    = 8   # 1st hidden layer dimension 
DIM_H2    = 16  # 2st hidden layer dimension 
DIM_OUT   = 2   # Output dimension           (One-hot classification [y < x^2, y = x^2, y > x^2])

LEARN_RATE  = 0.01  # Learning rate of NN
CUTOFF_LOSS = 0.01  # During training, if loss reaches at or below this value, stop training
EPOCHS      = 100   # Maximum allowed number of training iterations for NN


'''
Class definition of the x^2 neural net
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_H1)   # Input layer
        self.fc2 = nn.Linear(DIM_H1, DIM_H1)    # Hidden layer
        self.fc3 = nn.Linear(DIM_H1, DIM_H2)
        self.fc4 = nn.Linear(DIM_H2, DIM_OUT)  # Output layer

    def forward(self, x):
        x = x.view(-1, DIM_IN)
        x = F.relu(self.fc1(x)) # Forward pass through all the layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)             # Don't want to ReLU the last layer
        return x


'''
Train the neural net (with backpropagation)
'''
def train(net, train_set):
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    loss  = torch.tensor([1])
    epoch = 0

    while epoch < EPOCHS and loss.item() > CUTOFF_LOSS:
        epoch = epoch + 1

        for data in train_set:
            sample = data[:2]
            label  = data[2:]

            output = net(sample.view(-1, 2))      # Run the NN on the value
            loss   = F.mse_loss(output[0], label) # Calculate how incorrect the NN was

            optimizer.zero_grad() # Start gradient at zero
            loss.backward()       # Backward propagate the loss
            optimizer.step()      # Adjust the weights based on the backprop

        if epoch % 1 == 0:
            print(f'Epoch #{str(epoch).ljust(2)} loss: {round(loss.item(), 3)}')



'''
Basic test of the neural net
'''
def test(net, test_set):

    print('x        | x^2      | y        | label    | guess   | confidence')
    print('---------+----------+----------+----------+---------+-----------')

    num_items = 0
    num_correct = 0

    with torch.no_grad():
        for data in test_set:

            sample = data[:2]
            label  = data[2:]
            
            output = net(sample.view(-1, DIM_IN))
            label_val  = torch.argmax(label.view(-1, 2))
            output_val = torch.argmax(output.view(-1, 2))

            if label_val == 0:
                label_str = 'negative'
            else:
                label_str = 'positive'

            if output_val == 0:
                output_str = 'negative'
            else:
                output_str = 'positive'

            if label_val == output_val:
                num_correct = num_correct + 1
            num_items = num_items + 1

            print(f'{str(round(sample[0].item(), 3)).ljust(8)} | {str(round(sample[0].item()**2,3)).ljust(8)} | {str(round(sample[1].item(), 3)).ljust(8)} | {label_str.ljust(8)} | {output_str.ljust(8)} | {output}')

        print(f'Num correct: {num_correct}')
        print(f'Num total  : {num_items}')



'''
TODO test again and graph using MatPlotLib
'''
# def fancy_test(net):



'''
Generates a batch_size long tensor with the values
for x and x^2, with the range for x being [-max_x, max_x]
'''
def generate_set(batch_size):
    RAND_STEP = 0.001 # The granularity of the values in the test set
    set = torch.zeros(batch_size, 4, dtype=torch.float32)

    for i in range (0, batch_size):
        # Get a random value for x in the range [-max_x, max_x]
        x = float(random.randrange(MAX_X_VALUE * -1 / RAND_STEP, MAX_X_VALUE / RAND_STEP))
        x = x * RAND_STEP

        deltaY = DELTA_Y_VALUE - MIN_Y_DELTA

        y = float(random.randrange(deltaY * -1 / RAND_STEP, deltaY / RAND_STEP)) + MIN_Y_DELTA
        y = ((y * RAND_STEP) + x) ** 2

        set[i][0] = x    # x
        set[i][1] = y    # y

        if y < x ** 2:
            set[i][2] = 1
            set[i][3] = 0
        else:
            set[i][2] = 0
            set[i][3] = 1

    return set

def print_set(set, batch_size):
    neg_count = 0
    pos_count = 0

    print('x        | x^2      | y        | label')
    print('---------+----------+----------+----------')
    for i in range(batch_size):
        sample = set[i][:2]
        label  = set[i][2:]
        label_val = torch.argmax(label.view(-1,2))

        if label_val == 0:
            label_str = 'negative'
            neg_count = neg_count + 1
        else:
            label_str = 'positive'
            pos_count = pos_count + 1

        print(f'{str(round(sample[0].item(), 3)).ljust(8)} | {str(round(sample[0].item()**2,3)).ljust(8)} | {str(round(sample[1].item(), 3)).ljust(8)} | {label_str}')
    print(f'\nnegative count: {neg_count}')
    print(f'positive count: {pos_count}')



'''
Actual code (not functions) begins here
'''
train_set = generate_set(TRAIN_BATCH_SZ)
test_set  = generate_set(TEST_BATCH_SZ)

net = Net()

print('\nNet: ', net)
print('\nTraining Set: ')
print_set(train_set, TRAIN_BATCH_SZ)
print('\nTesting Set: ')
print_set(test_set, TEST_BATCH_SZ)

train(net, train_set)
test(net, test_set)
# print()


# print('\n--- Before training: ---')
# test(net, test_set)

# print("\n--- Training now: ---\n")
# train(net, train_set)

# print("\n--- After training: ---")
# test(net, test_set)
# fancy_test(net)

