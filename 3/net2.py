'''
File:    net2.py
Authors: Tyler Tracy, Carson Molder
Class:   CSCE 4613 Artificial Intelligence
Date:    12/8/19

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

import numpy as np              # Numpy dependency
import torch                    # Pytorch dependenices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt # MatPlotLib dependency
import random                   # Python dependencies
import itertools


'''
Constants for the net, training, and testing
'''
TRAIN_BATCH_SZ = 100 # Batch size for training set
TEST_BATCH_SZ  = 50  # Batch size for basic testing set

MAX_X   = 10         # Maximum value for x for y = x^2     (training / basic test)
DELTA_Y = 10         # Maximum that y can diverge from x^2 (training / basic test)

FANCY_TEST_MAX_X = 10   # Maximum value of x for y = x^2            (fancy test)
FANCY_TEST_STEP  = 0.25  # Granularity of the fancy test testing set (fancy test)

DIM_IN     = 1     # Input dimension            (1, for the value of x)
DIM_H      = 4     # 1st hidden layer dimension (2nd hidden layer dimension is this, squared)
DIM_OUT    = 1     # Output dimension           (1, for the NN's estimate of x^2)

LEARN_RATE = 0.02  # Learning rate of NN
EPOCHS     = 30    # Number of training iterations for NN



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
            # data = [ x, y, x^2 ]

            x  = data[:1] # sample [ x ]
            x2 = data[2:] # label  [ x^2 ]

            output = net(x.view(-1, DIM_IN))         # Run the NN on the value
            loss   = F.smooth_l1_loss(output[0], x2) # Calculate how incorrect the NN was

            optimizer.zero_grad() # Start gradient at zero
            loss.backward()       # Backward propagate the loss
            optimizer.step()      # Adjust the weights based on the backprop

        if(epoch % 5 == 0):
            print(f'Epoch #{str(epoch).ljust(2)} loss: {round(loss.item(), 3)}')



'''
Basic test of the neural net
'''
def test(net, test_set):

    print('x         | y         | x2 NN     | x2 label  | y vs x2 NN | y vs x2 label | correct?')
    print('----------+-----------+-----------+-----------+------------+---------------+---------')

    diffs = []
    num_correct = 0

    num_positive_nn = 0 # Number of y's that were greater than their NN's estimate of x^2
    num_negative_nn = 0 # Number of y's that were less than their NN's estimate of x^2

    num_positive_x2 = 0 # Number of y's that were greater than their x^2 label
    num_negative_x2 = 0 # Number of y's that were less than their x^2 label

    with torch.no_grad():
        for data in test_set:
            # data = [ x, y, x^2 ]

            x  = data[0] # Input [ x ]
            y  = data[1] # Value [ y ] for output to be compared to
            x2 = data[2] # Label [ x^2 ]

            output = net(x.view(-1, DIM_IN))
            diff   = abs((output.item() - x2.item())/x2.item()) * 100
            diffs.append(diff)

            # Determine strings for y vs x2 columns
            if y > output.item():
                nn_str = '+ (y > x2)'
                num_positive_nn = num_positive_nn + 1
            else:
                nn_str = '- (y < x2)'
                num_negative_nn = num_negative_nn + 1

            if y > x2.item():
                x2_str = '+ (y > x2)'
                num_positive_x2 = num_positive_x2 + 1
            else:
                x2_str = '- (y < x2)'
                num_negative_x2 = num_negative_x2 + 1

            # Detertime string for last column
            if nn_str == x2_str:
                correct_str = 'yes'
                num_correct = num_correct + 1
            else:
                correct_str = 'no'

            print(f'{str(round(x.item(), 3)).ljust(9)} | {str(round(y.item(),3)).ljust(9)} | {str(round(output.item(), 3)).ljust(9)} | {str(round(x.item()**2, 3)).ljust(9)} | {nn_str.ljust(10)} | {x2_str.ljust(13)} | {correct_str}')
    
    pct_correct = round(num_correct/len(diffs) * 100, 3)
    med_diff = round(np.median(diffs), 3)
    avg_diff = round(np.average(diffs), 3)

    print()
    print(f'# tested  : {len(diffs)}')
    print(f'# correct : {num_correct}')
    print()
    print(f'# of y > x2 label: {num_positive_x2}')
    print(f'# of y < x2 label: {num_negative_x2}')
    print()
    print(f'# of y > x2 NN: {num_positive_nn}')
    print(f'# of y < x2 NN: {num_negative_nn}')
    print()
    print(f'% correct guess : {pct_correct}%')
    print(f'% x2 NN and label diff (median) : {med_diff}%')
    print(f'% x2 NN and label diff (average): {avg_diff}%')



'''
Generates a batch_size long tensor with random values
for x, y, (inputs) and x^2 (label)
- range for x is [-max_x, max_x]
- range for y is [x^2 - delta_y, x^2 + delta_y]
'''
def generate_random_set(batch_size, max_x, delta_y):
    RAND_STEP = 0.001 # The granularity of the values in the test set
    set = torch.zeros(batch_size, 3, dtype=torch.float32)

    for i in range (0, batch_size):

        # Get a random value for x in the range [-max_x, max_x]
        x = float(random.randrange(max_x * -1 / RAND_STEP, max_x / RAND_STEP))
        x = x * RAND_STEP

        # Get a random value for y in the range [ x^2 - delta_y, x^2 + delta_y ]
        y = float(random.randrange(delta_y / RAND_STEP * -1, delta_y / RAND_STEP))

        y = x**2 + (y * RAND_STEP)

        set[i][0] = x     # Input 1
        set[i][1] = y     # Input 2
        set[i][2] = x**2  # Label

    return set


'''
Generates a tensor with the values
representing points on the x-y plane.
- range for x is [-max_x, max_x]
- range for y is [0, max_x ^2]
- step is the width (granularity) between points on the plane for both axes
'''
def generate_orderly_set(max_x, step = 0.1):
    xvals  = np.arange(-max_x, max_x + step, step)
    yvals  = np.arange(0, max_x**2 + step, step)
    points = torch.Tensor(list(itertools.product(xvals, yvals)))
    return points



'''
Generates a set of points using generate_orderly_set, then
runs the neural net through each of them, labelling it with...
- 0, if y <= NN's estimate of x2
- 1, if y > NN's estimate of x2

Returns the set as a list with objects of the form [x, y, label]
'''
def fancy_test(net, max_x, step):
    test_set = generate_orderly_set(max_x, step)
    results = []

    with torch.no_grad():
        for data in test_set:
            x  = data[0] # input [ x ]
            y  = data[1] # value [ y ] 

            output = net(x.view(-1, DIM_IN))

            if y > output.item():
                results.append([x.item(), y.item(), 1]) # Output is positive
            else:
                results.append([x.item(), y.item(), 0])  # Output is negative

    return results




'''
Plots two graphs with an overlay of the actual function y = x^2.

If y > the NN's estimate of x2, it is plotted as a green point
If y < the NN's estimate of x2, it is plotted as a blue point

The graph's domain is [-max_x, max_x]
The graph's range is [0, (max_x)^2]
'''
def fancy_plot(before_results, after_results, max_x):
    plt.figure()

    x = np.linspace(-max_x, max_x, 500)
    y = x**2

    # Plot the test results before training
    plt.subplot(1, 2, 1)
    plt.scatter(x = [item[0] for item in before_results], y = [item[1] for item in before_results], c = [item[2] for item in before_results], cmap = 'winter')
    plt.title('Before training')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim([-max_x, max_x])
    plt.ylim([0, max_x**2])
    plt.plot(x, y, 'r')

    # Plot the test results after training
    plt.subplot(1, 2, 2)
    plt.scatter(x = [item[0] for item in after_results], y = [item[1] for item in after_results], c = [item[2] for item in after_results], cmap = 'winter')
    plt.title('After training')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim([-max_x, max_x])
    plt.ylim([0, max_x**2])
    plt.plot(x, y, 'r')

    plt.show()



'''
Actual code (not functions) begins here
'''
train_set = generate_random_set(TRAIN_BATCH_SZ, MAX_X, DELTA_Y)
test_set  = generate_random_set(TEST_BATCH_SZ, MAX_X, DELTA_Y)

net = Net()

print('\n--- Net: ---\n', net)
print('\n--- Training Set: ---\n', train_set)
print('\n--- Testing Set: ---\n', test_set)
print()

print('\n--- Testing data before training... ---')
test(net, test_set)
before_results = fancy_test(net, FANCY_TEST_MAX_X, FANCY_TEST_STEP)

print("\n--- Training now... ---")
train(net, train_set)

print("\n--- Testing data after training... ---")
test(net, test_set)
after_results = fancy_test(net, FANCY_TEST_MAX_X, FANCY_TEST_STEP)

print('\n--- Plotting test results before and after... ---')
fancy_plot(before_results, after_results, FANCY_TEST_MAX_X)
