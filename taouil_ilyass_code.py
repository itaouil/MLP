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
        each classifier.
    """

    # Plot classes' points
    plt.plot(c1.T[0], c1.T[1], 'ro')
    plt.plot(c2.T[0], c2.T[1], 'bo')
    plt.plot(c3.T[0], c3.T[1], 'go')
    plt.plot(c4.T[0], c4.T[1], 'co')
    plt.axis([-10,7,-10,6])
    plt.show()

generate_dataset()
