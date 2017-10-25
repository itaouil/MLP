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
import numpy as np
import matplotlib.pyplot as plt

# Generate datasets
def generate_dataset():

    # Class1 points: 2D array formed by stacking x and y arrays
    c1 = np.column_stack((np.random.uniform(2, 5, 500), np.random.uniform(1, 4, 500)))

    # Class2 points: 2D array formed by stacking x and y arrays
    c2 = np.column_stack((np.random.uniform(1, 3, 500), np.random.uniform(-5, -1, 500)))

    # Plot C1 and C2
    plt.plot(c1.T[0], c1.T[1], 'ro')
    plt.plot(c2.T[0], c2.T[1], 'bo')
    plt.axis([0,10,-8,10])
    plt.show()

generate_dataset()
