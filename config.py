#!/usr/bin/env/python

"""
    The following Python script
    contains values used in the
    *_code.py script, such as the
    angle chosen for the R matrix,
    as well as the rectangles size,
    and more.

    Author      : Ilyass Taouil
    Module      : Machine Learning
    Instructor  : Dr. Matteo Leonetti
"""

# Import packages
import math
import numpy as np

data = {

    # Angle for R matrix (radians)
    "angle"     : math.radians(-75),

    # Number of points
    "size"      : 500,

    # Hidden nodes
    "h"         : 1,

    # C1 rectagle size
    "c1_x_low"  : 2,
    "c1_x_high" : 5,
    "c1_y_low"  : 1,
    "c1_y_high" : 4,

    # C2 rectagle size
    "c2_x_low"  : 1,
    "c2_x_high" : 3,
    "c2_y_low"  : -5,
    "c2_y_high" : -1,

    # C3 mean and covariance arrays
    "c3_mean"   : [-2, -3],
    "c3_cov"    : [[0.5, 0], [0, 3]],

    # C4 mean and covariance arrays
    "c4_mean"   : [-4, -1],
    "c4_cov"    : [[3, 0.5], [0.5, 0.5]],

    # Class dimension
    "dim"       : (500, 1),

    # w1 dimesion
    "w1_dim"     : (3, 1),

    # w2 dimesion
    "w2_dim"     : (1+1, 4)

}
