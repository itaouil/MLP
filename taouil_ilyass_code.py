#!/usr/bin/env/python

"""
    The following Python file contains
    all the functions used to generate
    the datasets, as well as training,
    evaluating and testing the MLP.

    Author      : Ilyass Taouil
    Module      : Machine Learning
    Instructor  : Dr. Matteo Leonetti
"""

# Import packages
import math
import numpy as np
import matplotlib.pyplot as plt

# Import configuration file
import config as cf

# Generate a filled vector
def filled_vector(fill_value):
    return np.full(cf.data["dim"], fill_value)

# Generate datasets
def generate_dataset():

    """
        Classifier1 and Classifier2
    """

    # Class1 points: 2D array formed by stacking x and y arrays
    c1 = np.column_stack((np.random.uniform(cf.data["c1_x_low"], cf.data["c1_x_high"], cf.data["size"]),
                          np.random.uniform(cf.data["c1_y_low"], cf.data["c1_y_high"], cf.data["size"])))

    # Class2 points: 2D array formed by stacking x and y arrays
    c2 = np.column_stack((np.random.uniform(cf.data["c2_x_low"], cf.data["c2_x_high"], cf.data["size"]),
                          np.random.uniform(cf.data["c2_y_low"], cf.data["c2_y_high"], cf.data["size"])))

    # Define matrix R
    R = np.array([ [math.cos(cf.data["angle"]), -math.sin(cf.data["angle"])],
                   [math.sin(cf.data["angle"]), math.cos(cf.data["angle"])] ])

    # C1 and C2 points rotation
    for x in range(cf.data["size"]):
        c1[x] = np.dot(c1[x], R)
        c2[x] = np.dot(c2[x], R)

    """
        Classifier3 and Classifier4
    """

    # Class3 points generation
    c3 = np.random.multivariate_normal(cf.data["c3_mean"], cf.data["c3_cov"], cf.data["size"])

    # Class4 points generation
    c4 = np.random.multivariate_normal(cf.data["c4_mean"], cf.data["c4_cov"], cf.data["size"])

    """
        Add relative class to
        each classifier class
    """

    # Add class to C1
    class1 = np.column_stack((c1, filled_vector(1)))

    # Add class to C2
    class2 = np.column_stack((c2, filled_vector(2)))

    # Add class to C3
    class3 = np.column_stack((c3, filled_vector(3)))

    # Add class to C4
    class4 = np.column_stack((c4, filled_vector(4)))

    """
        Concatenate arrays
        into one array
    """

    # Concatenate arrays
    matrix = np.concatenate((class1, class2, class3, class4))

    """
        Shuffle matrix
    """

    # Shuffle matrix randomly
    np.random.shuffle(matrix)

    """
        Plot points
    """

    # # Plot classes' points
    # plt.plot(c1.T[0], c1.T[1], 'ro')
    # plt.plot(c2.T[0], c2.T[1], 'bo')
    # plt.plot(c3.T[0], c3.T[1], 'go')
    # plt.plot(c4.T[0], c4.T[1], 'co')
    # plt.axis([-10,7,-10,6])
    # plt.show()

    return matrix[:1000], matrix[1000:1500], matrix[1500:]

train, validation, test = generate_dataset()

print("Training", train)
print("Training shape", train.shape)
print("Training size", train.size)
print("Training len", len(train))

print("Validation", validation)
print("Validation shape", validation.shape)
print("Validation size", validation.size)
print("Validation len", len(validation))

print("Test", test)
print("Test shape", test.shape)
print("Test size", test.size)
print("Test len", len(test))
