#!/usr/bin/env/python

"""
    The following Python file contzins
    all the functions used to generate
    the datasets, trzin and evaluate
    the MLP.

    Author      : Ilyass Taouil
    Module      : Machine Learning
    Instructor  : Dr. Matteo Leonetti
"""

# Import packages
import math
import pandas as pd
import numpy as np
import config as cf
import matplotlib.pyplot as plt

"""
    Helper functions.
"""
# One hot encoding on
# dataset inputs
def encode(dataset):

    # Create nx4 vector to
    # to hold the dataset
    # classes
    targets = np.zeros((np.shape(dataset)[0], 4))

    # Class1 as [1, 0, 0, 0]
    indices = np.where(dataset[:,2] == 1)
    targets[indices,0] = 1

    # Class2 as [0, 1, 0, 0]
    indices = np.where(dataset[:,2] == 2)
    targets[indices,1] = 1

    # Class3 as [0, 0, 1, 0]
    indices = np.where(dataset[:,2] == 3)
    targets[indices,2] = 1

    # Class4 as [0, 0, 0, 1]
    indices = np.where(dataset[:,2] == 4)
    targets[indices,3] = 1

    return targets

# Decode class
def decode(output):
    return np.argmax(output) + 1

# Sigmoid function vectorization
def vsigmoid(x):
    f = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
    return f(x)

# Sum of Squared Difference function
def error_function(targets, outputs):
    f = np.vectorize(lambda x,y: 0.5 * (x - y) ** 2)
    return f(targets, outputs)

# Stacks two arrays together
def stack(A, B):
    return np.column_stack((A, B))

# Generates uniform data
def uniform(l, h, size):
    return np.random.uniform(l, h, size)

# Generates multivariate data
def multivariate(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)

"""
    Generates datasets and
    returns the three sets
    for the MLP algorithm.

    Input   : None
    Output  : Array of arrays
"""
def generate_dataset():

    # Rotation matrix (R)
    R = np.array([ [math.cos(cf.data["angle"]), -math.sin(cf.data["angle"])],
                   [math.sin(cf.data["angle"]), math.cos(cf.data["angle"])] ])

    # Generate points for class1
    c1 = stack(uniform(cf.data["c1_x_low"], cf.data["c1_x_high"], cf.data["size"]),
               uniform(cf.data["c1_y_low"], cf.data["c1_y_high"], cf.data["size"]))

    # Generate points for class2
    c2 = stack(uniform(cf.data["c2_x_low"], cf.data["c2_x_high"], cf.data["size"]),
               uniform(cf.data["c2_y_low"], cf.data["c2_y_high"], cf.data["size"]))

    # Rotate class1 and class2 points
    c1 = np.dot(c1, R)
    c2 = np.dot(c2, R)

    # Generate points for class3
    c3 = multivariate(cf.data["c3_mean"], cf.data["c3_cov"], cf.data["size"])

    # Generate points for class4
    c4 = multivariate(cf.data["c4_mean"], cf.data["c4_cov"], cf.data["size"])

    # Add columns of 1s to class1
    c1 = stack(c1, np.full(cf.data["dim"], 1))

    # Add columns of 2s to class2
    c2 = stack(c2, np.full(cf.data["dim"], 2))

    # Add columns of 3s to class3
    c3 = stack(c3, np.full(cf.data["dim"], 3))

    # Add columns of 4s to class4
    c4 = stack(c4, np.full(cf.data["dim"], 4))

    # Aggregate classes together in one bigger matrix (2000,3)
    matrix = np.concatenate((c1, c2, c3, c4))

    # Randomly order data
    np.random.shuffle(matrix)

    # Normalise dataset
    matrix[:,:2] = (matrix[:,:2] - matrix[:,:2].mean(axis=0)) / matrix[:,:2].var(axis=0)

    # Plot classes' points
    # plt.plot(c1.T[0], c1.T[1], 'ro')
    # plt.plot(c2.T[0], c2.T[1], 'bo')
    # plt.plot(c3.T[0], c3.T[1], 'go')
    # plt.plot(c4.T[0], c4.T[1], 'co')
    # plt.axis([-10,7,-10,6])
    # plt.show()

    # Returns training, evaluation and testing sets
    return matrix[:1000], matrix[1000:1500], matrix[1500:]

"""
    Classifies test set
    using trained MLP.
"""
def classify_mlp(w1,w2,x):

    # Forward phase (hidden layer)
    # Please note that the sigmoid
    # is applied directly to the dot
    # product and the result of it
    # is stacked together with a
    # bias input
    # print(np.shape(vsigmoid(np.dot(x, w1))))
    # print(np.shape(x))
    zj = np.append(vsigmoid(np.dot(x, w1)), -1)

    # Forward phase (output layer)
    # Please note that the sigmoid
    # function is applied directly
    # to the dot product
    zk = vsigmoid(np.dot(zj, w2))

    # Decode output
    return decode(zk)

"""
    Evalutates error
    on the validation
    set.
"""
def evaluate_mlp(w1,w2,D,targets):

    # Get targets of dataset
    # targets = encode(D)

    # Forward phase (hidden layer)
    # Please note that the sigmoid
    # is applied directly to the dot
    # product and the result of it
    # is stacked together with a
    # bias input
    zj = stack(vsigmoid(np.dot(D[:, :3], w1)), np.full((np.shape(D)[0], 1), -1).ravel())

    # Forward phase (output layer)
    # Please note that the sigmoid
    # function is applied directly
    # to the dot product
    zk = vsigmoid(np.dot(zj, w2))

    # Compute errors on the
    # weights matrix given
    errors = np.sum(error_function(targets, zk))

    return errors

"""
    Trains the MLP with the
    forward fit, followed
    by the backpropagation
    phase.

    Input   : w1, w2, # of hidden nodes, step size, training set, evalutaion set
    Output  : Updated weights
"""
def train_mlp(w1,w2,h,eta,D,E):

    # Get targets for training dataset
    t = encode(D)

    # Add bias input to D (in place of target)
    D[:, 2] = np.full((np.shape(D)[0], 1), -1).ravel()

    # Validation error
    val_error = evaluate_mlp(w1,w2,E, encode(E))

    # Errors
    train_errors = []
    valid_errors = []

    # Training + Validation
    for x in range(20000):

        # Shuffle training every iteration
        # to change the order on which the
        # MLP is trained
        # order = range(np.shape(D)[0])
        # order = np.random.shuffle(order)
        # t = np.reshape(t[order, :], (1000, 4))
        # D = np.reshape(D[order, :], (1000, 3))

        # Forward phase (hidden layer)
        # Please note that the sigmoid
        # is applied directly to the dot
        # product and the result of it
        # is stacked together with a
        # bias input
        zj = stack(vsigmoid(np.dot(D[:, :3], w1)), np.full((np.shape(D)[0], 1), -1).ravel())

        # Forward phase (output layer)
        # Please note that the sigmoid
        # function is applied directly
        # to the dot product
        zk = vsigmoid(np.dot(zj, w2))

        # Check overfitting on
        # the validation set and
        # stop training accordingly
        # (the checking is done every
        # 30 iterations on the test set)
        if x % 20 == 0 and val_error < evaluate_mlp(w1,w2,E, encode(E)):
            break
        elif x % 20 == 0 and val_error >= evaluate_mlp(w1,w2,E, encode(E)):
            val_error = evaluate_mlp(w1,w2,E, encode(E))

        # Store errors
        train_errors.append(evaluate_mlp(w1, w2, D, t))
        valid_errors.append(evaluate_mlp(w1, w2, E, encode(E)))

        # Backpropagation
        for r in range(len(D)):

            # Caching deltak
            deltak = []

            # Outer neurons wjk update
            for j in range(h+1):
                tmp = 0
                for k in range(4):
                    delta = (zk[r,k] - t[r,k]) * zk[r,k] * (1 - zk[r,k])
                    w2[j,k] -= eta * delta * zj[r,j]
                    tmp += delta * w2[j,k]

                deltak.append(tmp)

            # Inner neurons wij update
            for i in range(3):
                for j in range(h):
                    delta = zj[r,j] * (1 - zj[r,j]) * deltak[j]
                    w1[i,j] -= eta * delta * D[r,i]

        # Print errors
        print("Validation E:", evaluate_mlp(w1,w2,E, encode(E)))

    print("Stopped training.")
    print("Prev Eval E:", val_error)
    print("Curr Eval E:", evaluate_mlp(w1,w2,E, encode(E)))

    return w1, w2, train_errors, valid_errors

"""
    Main.
"""

def main():

    # Random weights
    w1 = np.random.uniform(-0.5, 0.5, cf.data["w1_dim"])
    w2 = np.random.uniform(-0.5, 0.5, cf.data["w2_dim"])

    # Retrieve datasets
    training, validation, test = generate_dataset()

    # Train the MLP
    w1_update, w2_update, train_errors, valid_errors = train_mlp(w1, w2, 2, 0.01, training, validation)

    # Test MLP
    for x in range(cf.data["size"]):
        y = classify_mlp(w1_update, w2_update, test[x])
        print("{}: {} -> {}".format(test[x, :2], test[x, 2], y))
        test[x, 2] = y

    # Errors on test set
    print("Errors:", evaluate_mlp(w1_update, w2_update, test, encode(test)))

    # Plot errors
    plt.plot(train_errors)
    plt.plot(valid_errors)
    plt.ylabel("Errors")
    plt.xlabel("Time")
    plt.show()

    # Plot test (as evaluated by MLP)
    indices = np.where(test[:,2] == 1)
    c1 = test[indices, :2]

    indices = np.where(test[:,2] == 2)
    c2 = test[indices, :2]

    indices = np.where(test[:,2] == 3)
    c3 = test[indices, :2]

    indices = np.where(test[:,2] == 4)
    c4 = test[indices, :2]

    plt.plot(c1.T[0], c1.T[1], 'ro')
    plt.plot(c2.T[0], c2.T[1], 'bo')
    plt.plot(c3.T[0], c3.T[1], 'go')
    plt.plot(c4.T[0], c4.T[1], 'co')
    plt.axis([-2,2,-2,2])
    plt.show()

# Check if the node is executing in the mzin path
if __name__ == '__main__':
    main()
