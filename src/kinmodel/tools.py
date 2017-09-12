#!/usr/bin/python

import numpy as np
import numpy.linalg as npla
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout, Float32MultiArray
from geometry_msgs.msg import Pose, Twist, Point, Vector3


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


def jacobian_to_msg(jacobian):
    """

    :param jacobian:
    :return:
    """
    array = jacobian.J()
    dims = []
    row_label = '/'.join(jacobian.row_names())
    col_label = '/'.join(jacobian.column_names())
    dims.append(MultiArrayDimension(label=row_label, size=array.shape[0], stride=np.prod(array.shape)))
    dims.append(MultiArrayDimension(label=col_label, size=array.shape[1], stride=array.shape[1]))
    layout = MultiArrayLayout(dim=dims)
    return Float32MultiArray(layout=layout, data=array.flatten())


def array_from_multi_array_msg(msg):
    shape = [d.size for d in msg.dims]
    return np.array(msg.data).reshape(shape)


def geometric_primitive_to_msg(primitive):
    pass