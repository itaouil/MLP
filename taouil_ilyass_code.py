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
import numpy as np
import config as cf
import matplotlib.pyplot as plt

"""
    Generates datasets and
    returns the three sets
    for the MLP algorithm.

    Input   : None
    Output  : Array of arrays
"""
def generate_dataset():

    # Generate points for class1
    c1 = np.column_stack((np.random.uniform(cf.data["c1_x_low"], cf.data["c1_x_high"], cf.data["size"]),
                          np.random.uniform(cf.data["c1_y_low"], cf.data["c1_y_high"], cf.data["size"])))

    # Generate points for class2
    c2 = np.column_stack((np.random.uniform(cf.data["c2_x_low"], cf.data["c2_x_high"], cf.data["size"]),
                          np.random.uniform(cf.data["c2_y_low"], cf.data["c2_y_high"], cf.data["size"])))

    # Rotation matrix (R)
    R = np.array([ [math.cos(cf.data["angle"]), -math.sin(cf.data["angle"])],
                   [math.sin(cf.data["angle"]), math.cos(cf.data["angle"])] ])

    # Rotate class1 and class2 points
    for x in range(cf.data["size"]):
        c1[x] = np.dot(c1[x], R)
        c2[x] = np.dot(c2[x], R)

    # Generate points for class3
    c3 = np.random.multivariate_normal(cf.data["c3_mean"], cf.data["c3_cov"], cf.data["size"])

    # Generate points for class4
    c4 = np.random.multivariate_normal(cf.data["c4_mean"], cf.data["c4_cov"], cf.data["size"])

    # Add columns of 1s to class1
    class1 = np.column_stack((c1, np.full(cf.data["dim"], 1)))

    # Add columns of 2s to class2
    class2 = np.column_stack((c2, np.full(cf.data["dim"], 2)))

    # Add columns of 3s to class3
    class3 = np.column_stack((c3, np.full(cf.data["dim"], 3)))

    # Add columns of 4s to class4
    class4 = np.column_stack((c4, np.full(cf.data["dim"], 4)))

    # Aggregate classes together in one bigger matrix (2000,3)
    matrix = np.concatenate((class1, class2, class3, class4))

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

    # Returns trzining, evaluation and testing sets
    return matrix[:1000], matrix[1000:1500], matrix[1500:]

"""
    Trzin the MLP with the
    forward fit, followed
    by the backpropagation
    phase.

    Input   : Weights, # of hidden nodes, step size, trzining set
    Output  : Updated weights
"""
def train_mlp(w1,w2,h,eta,D):

    # Encode targets
    t = np.zeros((np.shape(D)[0], 4))

    # Class1 as [1, 0, 0, 0]
    indices = np.where(D[:,2] == 1)
    t[indices,0] = 1

    # Class2 as [0, 1, 0, 0]
    indices = np.where(D[:,2] == 2)
    t[indices,1] = 1

    # Class3 as [0, 0, 1, 0]
    indices = np.where(D[:,2] == 3)
    t[indices,2] = 1

    # Class4 as [0, 0, 0, 1]
    indices = np.where(D[:,2] == 4)
    t[indices,3] = 1

    # Define sigmoid function
    sigmoid = lambda x: 1 / (1 + math.exp(-1 * x))

    # Add bias input to D (in place of target)
    D[:, 2] = np.full((np.shape(D)[0], 1), -1).ravel()

    # Forward phase (hidden layer)
    zi = np.dot(D, w1)

    # Apply sigmoid to zi
    for x in zi:
        for n in range(h):
            x[n] = sigmoid(x[n])

    # Add bias input to zi
    zi = np.column_stack((zi, np.full((np.shape(zi)[0], 1), -1).ravel()))

    # Forward phase (ouput layer)
    zj = np.dot(zi, w2)

    # Apply sigmoid to zi
    for x in zj:
        for n in range(4):
            x[n] = sigmoid(x[n])

"""
    Main.
"""

def main():

    # Random weights
    w1 = np.random.uniform(-0.5, 0.5, cf.data["w1_dim"])
    w2 = np.random.uniform(-0.5, 0.5, cf.data["w2_dim"])

    # Get datasets
    training, validation, test = generate_dataset()

    # Call MLP trzining
    train_mlp(w1, w2, 3, 0.3, training)

# Check if the node is executing in the mzin path
if __name__ == '__main__':
    main()
