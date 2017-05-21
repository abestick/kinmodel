#!/usr/bin/python

import numpy as np
import numpy.linalg as npla


def multidot(*args):
    if len(args) == 2:
        return np.dot(*args)

    return np.dot(multidot(*args[0:-1]), args[-1])


def colvec(vec):
    return np.array(vec).reshape((-1, 1))


def array_squared(array):
    return np.dot(array.T, array)


def unit_vector(array):
    """Divides an array by its norm"""
    return array / npla.norm(array)