#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import scipy.optimize
import se3
from math import pi, log10, sqrt
import json
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from phasespace.mocap_definitions import MocapWrist
from tf.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix, euler_from_matrix, unit_vector
from copy import deepcopy
from numbers import Number
import warnings


class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def new_geometric_primitive(input_data, reference_frame=''):
    try:
        return float(input_data)
    except TypeError:
        pass

    # If the input is a GeometricPrimitive
    try:
        homog_array = input_data.homog()
    except AttributeError:
        # Otherwise, if it's an array-like object
        homog_array = np.asarray(input_data, dtype=float)
        homog_array = homog_array.squeeze()

    if homog_array.shape == (4,4):
        return Transform(homog_array, reference_frame=reference_frame)
    elif homog_array.shape == (3,3):
        return Rotation(homog_array, reference_frame=reference_frame)
    elif homog_array.shape == (4,) and homog_array[3] == 1:
        return Point(homog_array, reference_frame=reference_frame)
    elif homog_array.shape == (3,):
        return Point(np.append(homog_array, 1), reference_frame=reference_frame)
    elif homog_array.shape == (4,) and homog_array[3] == 0:
        return Vector(homog_array, reference_frame=reference_frame)
    elif homog_array.shape == (6,):
        return Twist(omega=homog_array[0:3], nu=homog_array[3:6], reference_frame=reference_frame)
    else:
        raise TypeError('input_data must be array-like or a GeometricPrimitive')


def stack(*args, **kwargs):
    """
    Takes multiple geometric objects and stacks their arrays, will throw a numpy error if incompatible shapes are passed
    :param args: the objects tp stack
    :param kwargs: various options categorized below
    homog (bool): When this is false, only the first three rows are taken, discarding homogeneous representation
    :return: numpy array
    :rtype: numpy.ndarray
    """
    homog = kwargs.pop('homog', True)

    if len(kwargs) > 0:
        print("The following options aren't implemented: %s" % kwargs.keys())

    dims = slice(None) if homog else slice(None, 3)

    return np.stack([arg.homog()[dims] for arg in args])


POSITION_NAMES = ('x', 'y', 'z')
QUATERNION_NAMES = ('w', 'i', 'j', 'k')
EULER_NAMES = ('a', 'b', 'c')
QUATERNION_POSE_NAMES = POSITION_NAMES + QUATERNION_NAMES
EULER_POSE_NAMES = POSITION_NAMES + EULER_NAMES
# TWIST_NAMES = ('omega_x', 'omega_y', 'omega_z', 'dx', 'dy', 'dz')


class ReferenceFrameWarning(Warning):

    def __init__(self, frame_a, frame_b, operation=None):
        op_str = '' if operation is None else ' (%s)' % operation
        super(ReferenceFrameWarning, self).__init__('Performing a frame dependent operation%s on two primitives of '
                                                    'different frames (%s and %s)' % (op_str, frame_a, frame_b))


def check_reference_frames(operator_method):
    def method_wrapper(self, other):
        if isinstance(other, GeometricPrimitive):
            if self._reference_frame != other._reference_frame:
                warnings.warn(ReferenceFrameWarning(self._reference_frame, other._reference_frame,
                                                    operator_method.__name__))

        return operator_method(self, other)

    return method_wrapper


class StateSpaceModel(object):
    """
    An abstract class describing the bare bones needed for a state space model to be defined
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def process_model(self, state_vector):
        """
        The process model of the system x(k+1) = p(x(k))
        :param state_vector: a numpy array of x(k)
        :return: x(k+1) 
        """
        pass

    @abstractmethod
    def measurement_model(self, state_vector):
        """
        The measurement model of the system z(k) = h(x(k))
        :param state_vector: a numpy array of x(k)
        :return: z(k)
        """
        pass

    @abstractmethod
    def vectorize_measurement(self, feature_obs):
        """
        Takes a dictionary of measurements and prepares it as a numpy array in the same order as the measurement_model
        produces z(k)
        :param dict feature_obs: a dictionary of the measurements
        :return: a numpy array z(k)
        """
        pass

    def state_length(self):
        """
        :return: the total number of states
        :rtype: int
        """
        return self._state_length


# Geometric primitives
class GeometricPrimitive(object):
    __metaclass__ = ABCMeta

    def __init__(self, reference_frame=''):
        self._reference_frame = reference_frame

    @abstractmethod
    def __array__(self, dtype=float):
        pass

    @abstractclassmethod
    def from_dict(cls, dictionary):
        pass

    @abstractmethod
    def homog(self):
        pass

    def __repr__(self):
        output = self.__class__.__name__ + ": " + str(self.homog())
        return output

    def __mul__(self, other):
        homog1 = self.homog()
        homog2 = other.homog()

        #Check that dimensions are compatible
        if homog1.shape != (4,4):
            raise TypeError("Dimension mismatch - can't compose primitives in this order")

        homog_result = homog1.dot(homog2)
        return new_geometric_primitive(homog_result)

    @abstractmethod
    def __div__(self, other):
        pass

    def __truediv__(self, other):
        return self.__div__(other)

    def _json(self):
        return self.homog().squeeze().tolist()

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def names(self, prefix=''):
        pass

    def reference_frame(self):
        return self._reference_frame

    def copy(self):
        return deepcopy(self)


class Feature(object):
    """ j = Feature(...) a feature in a kinematic tree

        .name - the name of the feature (must be unique in this tree)
        .primitive - the geometric primitive of this feature (wrt the base coordinate frame)
    """
    def __init__(self, name, primitive):
        self.name = name
        if hasattr(primitive, 'homog'):
            self.primitive = primitive
        else:
            self.primitive = new_geometric_primitive(primitive)

    def _json(self):
        OUTPUT_ATTRIBS = ['name', 'primitive']
        json_dict = OrderedDict()
        for attrib in OUTPUT_ATTRIBS:
            try:
                json_dict[attrib] = getattr(self, attrib)
            except AttributeError:
                pass
        return json_dict


class Joint(object):
    """ j = Joint(...) a joint in a kinematic tree

        .json(filename) - saves the kinematic tree to the specified file/home/pedge/catkin_ws/src

        .name - the name of this joint (must be unique in this tree)
        .children - list of other Joint objects
        .twist - the twist coordinates of the joint (only present if not the root)
    """
    def __init__(self, name, children=None, twist=None):
        self.name = name
        if children is None:
            self.children = []
        else:
            self.children = children #Children of the joint (other Joints)

        # TODO: Remove code for old-style twists here
        if twist is None:
            self.twist = None
        elif hasattr(twist, '_xi') or isinstance(twist, ParameterizedJoint):
            self.twist = twist
        else:
            self.twist = new_geometric_primitive(twist)

    def json(self, filename=None, args={}):
        if filename is None:
                return json.dumps(self, default=lambda o: o._json(), **args)
        else:
            with open(filename, 'w+') as output_file:
                json.dump(self, output_file, default=lambda o: o._json(), **args)

    def _json(self):
        OUTPUT_ATTRIBS = ['name', 'twist', 'children']
        json_dict = OrderedDict()
        for attrib in OUTPUT_ATTRIBS:
            try:
                json_dict[attrib] = getattr(self, attrib)
            except AttributeError:
                pass
        return json_dict


class ParameterizedJoint(object):
    """Base class for a joint with one or more degrees of freedom, parameterized by one or more
    scalar parameters.

    Each degree of freedom in a KinematicTree is parameterized by a single Twist. However, sometimes
    it's useful to group joints together into multi-DoF compound joints (e.g. a 3 DoF ball joint)
    or to allow only a subset of a Twist's six degrees of freedom to vary when fitting the tree.

    ParameterizedJoint allows all these situations. Initialize an instance with a list of twists and
    the current parameter values (as a list of scalars). Each 
    """
    __metaclass__ = ABCMeta

    def __init__(self, params, static_param_attribs=[]):
        params = np.atleast_1d(np.asarray(params).squeeze())
        if params.ndim > 1:
            raise ValueError('params must be a 1D array-like')
        self._twists = None
        self.params = params
        self._static_params = static_param_attribs

        # Let _set_params() populate the self._twists list
        self._set_params(params)

    def _set_twists(self, twists):
        self._twists = twists

    def _get_twists(self):
        return self._twists

    def config_shape(self):
        """Returns the shape of the config ndarray expected by exp() and dexp()
        """
        return (len(self._twists),)

    @abstractmethod
    def _set_params(self, params):
        pass

    @abstractmethod
    def normalize(self):
        """Normalizes the twists in a joint, returns the scaling constants to apply to each 
        config variable to yield an unchanged configuration.
        """
        pass

    def exp(self, thetas):
        thetas = np.atleast_1d(np.asarray(thetas).squeeze())
        if thetas.shape != (len(self._twists),):
            raise ValueError('Exptected a thetas vector of shape (' + str(len(self._twists)) + ',)')
        exps = []
        for i, twist in enumerate(self._twists):
            exps.append(twist.exp(thetas[i]))
        return exps

    def dexp(self, thetas):
        thetas = np.atleast_1d(np.asarray(thetas).squeeze())
        if thetas.shape != (len(self._twists),):
            raise ValueError('Exptected a thetas vector of shape (' + str(len(self._twists)) + ',)')
        dexps = []
        exps = self.exp(thetas)
        prod_exp = Transform()
        for i, twist in enumerate(self._twists):
            dexps.append(Twist(xi=(prod_exp.adjoint().dot(twist.xi()))))
            prod_exp = prod_exp * exps[i]
        return dexps

    def vectorize(self):
        return np.asarray(self.params)

    def to_dict(self):
        json_dict = OrderedDict()
        json_dict['joint_type'] = type(self).__name__
        json_dict['params'] = self.params.tolist()
        for attrib in self._static_params:
            json_dict[attrib] = getattr(self, attrib)
            try:
                json_dict[attrib] = json_dict[attrib].tolist()
            except AttributeError:
                # This isn't an ndarray, no need to list-ify it
                pass
        return json_dict

    def _json(self):
        return self.to_dict()

    def __repr__(self):
        output = self.__class__.__name__ + ": " + str(self.vectorize())
        return output

    @classmethod
    def from_list(cls, param_list):
        param_list = np.atleast_1d(np.asarray(param_list).squeeze())
        if param_list.shape == (6,):
            # 1 DoF twist joint
            return OneDofTwistJoint(param_list)
        elif param_list.shape == (3,):
            # 3 DoF ball joint
            return ThreeDofBallJoint(param_list)
        else:
            raise ValueError('param_list is not the correct shape for any joint type')

    @classmethod
    def from_dict(cls, attrib_dict):
        types = {'OneDofTwistJoint': OneDofTwistJoint, 'ThreeDofBallJoint':ThreeDofBallJoint}
        attrib_dict = attrib_dict.copy()
        new_type = attrib_dict['joint_type']
        del attrib_dict['joint_type']
        return types[new_type](**attrib_dict)


class OneDofTwistJoint(ParameterizedJoint):
    def __init__(self, params=None):
        super(OneDofTwistJoint, self).__init__(params, [])

    def _set_params(self, params):
        self._set_twists([Twist(xi=params)])

    def normalize(self):
        twist = self._get_twists()[0]
        norm_constant = la.norm(twist.omega())
        new_twist = Twist(xi=(twist.xi()/norm_constant))
        self._set_twists([new_twist])
        self.params = np.squeeze(twist.xi()/norm_constant)
        return np.array((norm_constant,))


class ThreeDofBallJoint(ParameterizedJoint):
    def __init__(self, params, joint_axes=[[1,0,0], [0,1,0], [0,0,1]]):
        # Columns of joint_axes specify the axes of rotation in the zero config
        self.joint_axes = joint_axes
        super(ThreeDofBallJoint, self).__init__(params, ['joint_axes'])

    def _set_params(self, params):
        new_twists = []
        for joint_axis in self.joint_axes:
            new_twists.append(Twist(omega=np.array(joint_axis), nu=se3.skew(params).dot(np.array(joint_axis))))
        self._set_twists(new_twists)

    def normalize(self):
        return np.ones(3)


class Transform(GeometricPrimitive):
    """
    g = Transform(...)  element of SE(3)

        .rot() - Rotation object
        .trans() - Vector object
        .homog() - (4,4) - homogeneous transformation ndarray
        .p() - (3,) - translation ndarray
        .R() - (3,3) - rotation ndarray
    """
    conventions = {'quaternion': QUATERNION_POSE_NAMES,
                   'euler': EULER_POSE_NAMES}

    @classmethod
    def from_dict(cls, dictionary, convention='euler'):
        element_names = cls.element_names[convention]
        assert set(element_names) == set(dictionary)
        pose = np.array([float(dictionary[pose_name]) for pose_name in element_names])
        return cls.from_pose_array(pose)

    @classmethod
    def from_pose_array(cls, pose_array):
        homog_array = np.identity(4)
        homog_array[:, 3] = pose_array[:3]
        homog_array[:3, :3] = quaternion_matrix(pose_array[3:])
        return cls(homog_array)

    @classmethod
    def from_p_R(cls, translation_array, rotation_matrix):
        homog_array = np.identity(4)
        homog_array[:, 3] = np.array(translation_array).flatten()
        homog_array[:3, :3] = np.array(rotation_matrix).squeeze()
        return cls(homog_array)

    def __init__(self, homog_array=None, reference_frame=''):
        if homog_array is None:
            self._H = np.identity(4)
        else:
            if homog_array.shape != (4,4):
                raise ValueError('Input ndarray must be (4,4)')
            self._H = homog_array

        super(Transform, self).__init__(reference_frame)

    def homog(self):
        return self._H

    def inv(self, child_frame_name=''):
        return Transform(homog_array=la.inv(self.homog()), reference_frame=child_frame_name)

    def trans(self):
        p = np.append(self._H[0:3,3], 0)
        return Vector(p)

    def rot(self):
        return Rotation(self._H.copy())

    def p(self):
        return self._H[0:3,3]

    def R(self):
        return self._H[0:3,0:3]

    def pose(self, convention='euler'):
        pose = np.empty(6 + int(convention == 'quaternion'))
        pose[:3] = self.trans().q()
        if convention == 'quaternion':
            pose[3:] = quaternion_from_matrix(self.rot().homog())

        elif convention == 'euler':
            pose[3:] = euler_from_matrix(self.R())

        else:
            raise NotImplementedError('Convention %s not yet implemented.' % convention)

        return pose

    def to_dict(self, convention='euler'):
        return dict(zip(self.conventions[convention], self.pose(convention)))

    def names(self, prefix='', convention='euler'):
        return [prefix+name for name in self.conventions[convention]]

    def adjoint(self):
        adj = np.zeros((6,6))
        adj[:3,:3] = self.R()
        adj[3:,3:] = self.R()
        adj[3:,:3] = se3.skew(self.p()).dot(self.R())
        return adj

    @check_reference_frames
    def __sub__(self, other):
        if isinstance(other, Transform):
            nu = self.trans() - other.trans() # the difference in position in the shared reference frame
            relative_rotation = self.rot().T() * other.rot() # the relative rotation
            relative_rotation_vector = relative_rotation.axis_angle() # the axis-angle vector in this one's child frame
            reference_rotation_vector = self.rot() * relative_rotation_vector # convert it to the shared reference frame

            # return result as a Twist for unit time delta, expressed in the shared reference frame
            return Twist(omega=reference_rotation_vector, nu=nu.q(), reference_frame=self._reference_frame)

    def __div__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new._H /= other
            new._H[-1, -1] = 1
            return new

        elif isinstance(other, Transform):
            return other.inv() * self

        else:
            raise NotImplementedError()

    def __array__(self, dtype=float):
        return self.pose().astype(dtype=dtype)

    def transform(self, primitive):
        assert isinstance(primitive, GeometricPrimitive), 'Can only transform GeometricPrimitves, you passed %s' \
                                                          %type(primitive)

        if isinstance(primitive, Twist):
            return Twist(xi=self.adjoint().dot(primitive.xi()))

        else:
            return self * primitive


class Jacobian(object):

    @classmethod
    def from_array(cls, array, row_names, column_names):
        """
        Creates a Jacobian from an array and row and column names
        :param numpy.ndarray array: 2D Matrix of the Jacobian
        :param row_names: the names of the rows
        :param column_names: the names of the columns
        :return: 
        """
        assert array.ndim == 2, "array must be 2-dimensional, not of shape %s" % array.shape
        assert array.shape == (len(row_names), len(column_names)), "Array shape must match length of row_names and " \
                                                                   "column_names. %s (shape) does not match (%d, %d) " \
                                                                   "(row and column name lengths)" % \
                                                                   (array.shape, len(row_names), len(column_names))
        jacobian = cls({}, [])
        jacobian._matrix = array
        jacobian._row_names = row_names
        jacobian._column_names = column_names

        return jacobian

    @classmethod
    def hstack(cls, jacobians):
        if len(jacobians) == 1:
            return jacobians[0].copy()
        return jacobians[0].append_horizontally(*jacobians[1:])

    @classmethod
    def vstack(cls, jacobians):
        if len(jacobians) == 1:
            return jacobians[0].copy()
        return jacobians[0].append_vertically(*jacobians[1:])

    @classmethod
    def zeros(cls, row_names, column_names):
        return cls.from_array(np.zeros((len(row_names), len(column_names))), row_names, column_names)

    @classmethod
    def identity(cls, names):
        return cls.from_array(np.identity(len(names)), names, names)

    def __init__(self, columns, row_names=None, column_names=None):
        """
        Constructor
        :param dict columns: 
        :param list row_names: 
        """
        if row_names is None:
            row_names = EULER_POSE_NAMES

        if column_names is None:
            column_names = columns.keys()

        self._matrix = np.empty((len(row_names), len(columns)))

        assert len(set(row_names)) == len(row_names), "Duplicates were found in row_names!"

        self._row_names = row_names
        self._column_names = []

        for j, column_name in enumerate(column_names):
            column = columns[column_name]
            self._column_names.append(column_name)
            column = np.array(column).flatten()
            assert len(self._row_names) == len(column), "len(row_names) [%d] does not match column length [%d]" % \
                                                        (len(row_names), len(column))
            self._matrix[:, j] = column

    def vectorize(self, input_dict, rows):
        """
        Takes a dictionary and produces a vector ordered according to the rows/columns of the Jacobian
        :param input_dict: the dictionary to be converted to a numpy array
        :param rows: a boolean indicating if it is to be orded according to the rows (false for columns)
        :return: a numpy array
        """
        # Get the right axis as specified by the rows flag
        axis = self._row_names if rows else self._column_names

        # If the keys do not match
        if not set(input_dict) == set(axis):
            new_dict={}

            # Loop through each of the dictionary items
            for prefix, primitive in input_dict.items():
                try:
                    new_dict.update(primitive.to_dict(prefix+'_'))
                except AttributeError:
                    raise ValueError('%s was not in %s nor is it a geometric primitive (%s)' %
                                     (prefix, set(axis), type(primitive)))


            assert set(new_dict) == set(axis), \
                'dict of gemetric primitives was converted, but their elements %s do not match %s' % \
                (set(new_dict), set(axis))

            input_dict = new_dict
        return np.array([input_dict[dim] for dim in axis])

    def copy(self):
        return deepcopy(self)

    def reorder(self, row_names=None, column_names=None):
        if row_names is not None and row_names != self._row_names:
            assert set(row_names) == set(self._row_names), "row_names must contain %s" % self._row_names
            new_array = np.empty_like(self._matrix)

            for i, row_name in enumerate(row_names):
                new_array[i, :] = self._matrix[self._row_names.index(row_name), :]

            self._matrix = new_array
            self._row_names = row_names

        if column_names is not None and column_names != self._column_names:
            assert set(column_names) == set(self._column_names), "column_names must contain %s" % self._column_names
            new_array = np.empty_like(self._matrix)

            for j, column_name in enumerate(column_names):
                new_array[:, j] = self._matrix[:, self._column_names.index(column_name)]

            self._matrix = new_array
            self._column_names = column_names

        return self

    def transform_right(self, array, column_names):
        array = array.reshape((array.shape[0], -1))
        assert len(set(column_names)) == len(column_names) == array.shape[1], "Duplicates were found in column_names!"

        new_one = self.copy()
        new_one._matrix = new_one._matrix.dot(array)
        new_one._column_names = column_names
        return new_one

    def transform_left(self, array, row_names):
        assert len(set(row_names)) == len(row_names) == array.shape[0], "Duplicates were found in row_names!"
        new_one = self.copy()
        new_one._matrix = array.dot(new_one._matrix)
        new_one._row_names = row_names
        return new_one

    def dot(self, other):
        if isinstance(other, np.ndarray):
            return self._matrix.dot(other)

        else:
            raise NotImplementedError("dot is not yet implemented for %s." % type(other))

    def __mul__(self, other):
        if isinstance(other, Jacobian):
            assert set(other._row_names) == set(self._column_names), "Right multiplicant must have the same rows as the " \
                                                                   "left multiplicant's columns. \n" \
                                                                   "Right rows: %s \n" \
                                                                   "Left columns: %s" % \
                                                                     (other._row_names, self._column_names)

            other = other.copy()
            other.reorder(row_names=self._column_names)
            return self.transform_right(other._matrix, other._column_names)

        elif isinstance(other, dict):
            other = self.vectorize(other, rows=False)
            return dict(zip(self.row_names(), self.dot(other)))

        elif isinstance(other, np.ndarray):
            return self.dot(other)

        elif isinstance(other, (int, float)):
            result = self.copy()
            result._matrix *= other
            return result

        else:
            raise NotImplementedError("Multiplication is not yet implemented for %s." % type(other))

    def __rmul__(self, other):

        if isinstance(other, dict):
            other = self.vectorize(other, rows=True)
            return dict(zip(self._column_names, other.dot(self._matrix).squeeze()))

        elif isinstance(other, np.ndarray):
            return other.dot(self._matrix)

        elif isinstance(other, (int, float)):
            return self * other

        else:
            raise NotImplementedError("Multiplication is not yet implemented for %s." % type(other))

    def row_names(self):
        return self._row_names

    def column_names(self):
        return self._column_names

    def J(self):
        return self._matrix.copy()

    def T(self):
        return self.from_array(self._matrix.T, self._column_names, self._row_names)

    def subset(self, row_names=None, column_names=None):
        if row_names is None:
            row_names = self._row_names
            row_indices = slice(None)

        else:
            try:
                row_indices = [self._row_names.index(row_name) for row_name in row_names]

            except ValueError as e:
                raise ValueError("row_names argument must contain a subset of this Jacobian's row_names.\n"
                                 "argument row_names: %s\n"
                                 "object row_names: %s\n"
                                 "mismatch: %s" % (row_names, self._row_names, e.message))

        if column_names is None:
            column_names = self._column_names
            column_indices = slice(None)

        try:
            column_indices = [self._column_names.index(row_name) for row_name in column_names]

        except ValueError as e:
            raise ValueError("column_names argument must contain a subset of this Jacobian's column_names.\n"
                             "argument column_names: %s\n"
                             "object column_names: %s\n"
                             "mismatch: %s" % (column_names, self._column_names, e.message))

        new_jac = self.copy()
        new_jac._matrix = new_jac._matrix[row_indices, :]
        new_jac._matrix = new_jac._matrix[:, column_indices]
        new_jac._row_names = list(row_names)
        new_jac._column_names = list(column_names)
        return new_jac

    def pinv(self):
        pinv_array = la.pinv(self._matrix)
        return InverseJacobian.from_array(pinv_array, self._column_names, self._row_names)

    def append_horizontally(self, other, *args):
        """
        Stacks Jacobians horizontally
        :param Jacobian other: 
        :return: 
        """

        if other is None:
            return self.copy()

        assert set(self._row_names) == set(other._row_names), "Jacobians must have the same row names!"
        new_one = other.copy()
        new_one.reorder(row_names=self._row_names)
        new_one._matrix = np.hstack((self._matrix, new_one._matrix))
        new_one._column_names = self._column_names + new_one._column_names

        if len(args) == 0:
            return new_one

        else:
            return new_one.append_horizontally(args[0], *args[1:])

    def append_vertically(self, other, *args):
        """
        Stacks Jacobians horizontally
        :param Jacobian other: 
        :return: 
        """

        if other is None:
            return self.copy()

        assert set(self._column_names) == set(other._column_names), "Jacobians must have the same column names!"
        new_one = other.copy()
        new_one.reorder(column_names=self._column_names)
        new_one._matrix = np.vstack((self._matrix, new_one._matrix))
        new_one._row_names = self._row_names + new_one._row_names

        if len(args) == 0:
            return new_one

        else:
            return new_one.append_vertically(args[0], *args[1:])

    def pad(self, row_names=None, column_names=None):
        if row_names is not None:
            row_names = [row_name for row_name in row_names if row_name not in self._row_names]
            row_padded = self.append_vertically(Jacobian.zeros(row_names, self._column_names))
        else:
            row_padded = self.copy()

        if column_names is not None:
            column_names = [column_name for column_name in column_names if column_name not in self._column_names]
            return row_padded.append_horizontally(Jacobian.zeros(row_padded._row_names, column_names))

        else:
            return row_padded

    def __str__(self):
        return 'columns: ' + str(self._column_names) + '\n' + 'rows: ' + str(self._row_names) + '\n' + str(self._matrix)

    def __repr__(self):
        return self.__str__()


class InverseJacobian(Jacobian):
    pass

    def pinv(self):
        pinv_array = la.pinv(self._matrix)
        return Jacobian.from_array(pinv_array, self._column_names, self._row_names)


class Twist(GeometricPrimitive):
    """ xi = Twist(...)  element of se(3)

        .xi() - 6 x 1 - twist coordinates ndarray (om, v)
        .omega() - 3 x 1 - rotation axis ndarray
        .nu() - 3 x 1 - translation direction ndarray
        .exp(theta) - Transform object

        ._xi - 6 x 1 - twist coordinates (om, v)
    """

    twist_names = EULER_POSE_NAMES

    @classmethod
    def from_dict(cls, dictionary):
        new_dict = {key[-1]: value for key, value in dictionary.items()}
        xi = np.array([new_dict[key] for key in EULER_POSE_NAMES])
        return cls(xi=xi)

    def __init__(self, omega=None, nu=None, copy=None, vectorized=None, xi=None, reference_frame=''):
        if xi is not None:
            self._xi = xi.squeeze()[:, None]
            assert self._xi.shape == (6,1)
        elif copy is not None:
            self._xi = copy.xi().copy()
        elif omega is not None and nu is not None:
            omega = np.asarray(omega)
            nu = np.asarray(nu)
            omega = np.reshape(omega, (3,1))
            nu = np.reshape(nu, (3,1))
            assert omega.shape == (3,1) and nu.shape == (3,1)
            self._xi = np.vstack((omega, nu))
        elif vectorized is not None:
            self._xi = np.asarray(vectorized)
        else:
            raise TypeError('You must provide either the initial twist coordinates or another Twist to copy')

        super(Twist, self).__init__(reference_frame)

    def __repr__(self):
        output = self.__class__.__name__ + ": " + str(self.xi().squeeze())
        return output

    def xi(self):
        return self._xi

    def omega(self):
        return self._xi.squeeze()[:3]

    def nu(self):
        return self._xi.squeeze()[3:]

    def exp(self, theta):
        return Transform(homog_array=se3.expse3(self._xi, theta))

    def vectorize(self):
        return np.array(self._xi).squeeze()

    def normalize(self):
        norm_constant = la.norm(self.omega())
        self._xi = self._xi / norm_constant
        return norm_constant

    def homog(self):
        # Get the skew-symmetric, (4,4) matrix form of the twist
        return se3.hat_(self._xi)

    def _json(self):
        return self._xi.squeeze().tolist()

    def to_dict(self, prefix=''):
        return {prefix+key: value for key, value in zip(self.twist_names, self._xi)}

    def names(self, prefix=''):
        return [prefix + name for name in self.twist_names]

    def __array__(self, dtype=float):
        return self._xi.squeeze().astype(dtype=dtype)

    def __mul__(self, other):
        if isinstance(other, Number):
            result = self.copy()
            result._xi *= other
            return result

    def __div__(self, other):
        if isinstance(other, Number):
            result = self.copy()
            result._xi /= other
            return result

    def __add__(self, other):
        return Twist(xi=self._xi + other._xi)


class Rotation(GeometricPrimitive):
    """ R = Rotation(...)  element of SO(3)

        .R() - (3,3) - rotation matrix ndarray
        .homog() - (4,4) - homogeneous coordinates ndarray (for a pure rotation)
    """

    quaternion_names = QUATERNION_POSE_NAMES[3:]

    @classmethod
    def from_dict(cls, dictionary):
        quaternion = np.array([dictionary[quaternion_name] for quaternion_name in cls.quaternion_names])
        return cls.from_quaternion(quaternion)

    @classmethod
    def from_quaternion(cls, quaternion):
        homog_array = np.identity(4)
        homog_array[:3, :3] = quaternion_matrix(quaternion)
        return cls(homog_array)

    def __init__(self, homog_array=None, reference_frame=''):
        if homog_array is None:
            self._R = np.identity(3)
        else:
            if homog_array.shape != (4,4):
                raise ValueError('Input ndarray must be (4,4)')
            self._R = homog_array[0:3,0:3]

        super(Rotation, self).__init__(reference_frame)

    def R(self):
        return self._R

    def homog(self):
        homog_matrix = np.identity(4)
        homog_matrix[0:3,0:3] = self._R
        return homog_matrix

    def quaternion(self):
        return quaternion_from_matrix(self.R())

    def axis_angle(self):
        u = np.array([self._R[2, 1] - self._R[1, 2],
                      self._R[0, 2] - self._R[2, 0],
                      self._R[1, 0] - self._R[0, 1]])
        return u

    def to_dict(self):
        return dict(zip(self.quaternion_names, self.quaternion()))

    def names(self, prefix=''):
        return [prefix + name for name in self.quaternion_names]

    def __mul__(self, other):
        if isinstance(other, Rotation):
            result = self.copy()
            result._R = self._R.dot(other._R)
            return result

        elif isinstance(other, np.ndarray):
            return self._R.dot(other)

    def __div__(self, other):
        if isinstance(other, float):
            new = self.copy()
            new._H /= other
            return new

        else:
            raise NotImplementedError()

    def __array__(self, dtype=float):
        return self.quaternion().astype(dtype=dtype)

    def T(self, child_frame_name=''):
        result = self.copy()
        result._R = result._R.T
        result._reference_frame = child_frame_name
        return result


class Vector(GeometricPrimitive):
    """ x = Vector(...)  translation in R^3

        .homog() - (4,) - homogeneous coordinates ndarray (x, 0)
        .q() - (3,) - cartesian coordinates ndarray
    """

    cartesian_names = QUATERNION_POSE_NAMES[:3]

    @classmethod
    def from_cartesian(cls, cartesian):
        return cls(np.append(cartesian, 0))

    @classmethod
    def from_point(cls, point):
        return cls.from_cartesian(point.q())

    @classmethod
    def from_dict(cls, dictionary):
        homog_array = np.array([dictionary[cartesian_name] for cartesian_name in cls.cartesian_names])
        return cls(homog_array)

    def __init__(self, homog_array=None, reference_frame=''):
        if homog_array is None:
            self._H = np.zeros(4)
        else:
            if homog_array.shape != (4,):
                raise ValueError('Input ndarray must be (4,)')
            self._H = homog_array

        super(Vector, self).__init__(reference_frame)

    def q(self):
        return self._H[0:3]

    def homog(self):
        return self._H

    def norm(self):
        return la.norm(self.q())

    def to_dict(self):
        return dict(zip(self.cartesian_names, self.q()))

    def names(self, prefix=''):
        return [prefix + name for name in self.cartesian_names]

    @check_reference_frames
    def __add__(self, other):

        if isinstance(other, (Vector, Point)):
            return new_geometric_primitive(self.homog() + other.homog(), reference_frame=other._reference_frame)

        elif isinstance(other, Number):
            return Vector(self._H + other)

        else:
            raise NotImplementedError()

    @check_reference_frames
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.homog() - other.homog(), reference_frame=other._reference_frame)

        elif isinstance(other, Point):
            return Point(np.append(self.q() - other.q(), 1), reference_frame=other._reference_frame)

        elif isinstance(other, Number):
            return Vector(self._H - other)

        else:
            raise NotImplementedError()

    def __div__(self, other):
        if isinstance(other, float):
            new = self.copy()
            new._H /= other
            return new

        else:
            raise NotImplementedError()

    def __array__(self, dtype=float):
        return self.q().astype(dtype=dtype)


class Point(GeometricPrimitive):
    """ x = Point(...)  point in R^3

        .homog() - (4,) - homogeneous coordinates ndarray (x, 0)
        .q() - (3,) - cartesian coordinates ndarray
    """

    cartesian_names = QUATERNION_POSE_NAMES[:3]

    @classmethod
    def from_cartesian(cls, cartesian):
        return cls(np.append(cartesian, 1))

    @classmethod
    def from_vector(cls, vector):
        return cls.from_cartesian(vector.q())

    @classmethod
    def from_dict(cls, dictionary):
        homog_array = np.array([dictionary[cartesian_name] for cartesian_name in cls.cartesian_names])
        return cls.from_cartesian(homog_array)

    def __init__(self, homog_array=None, vectorized=None, reference_frame=''):
        if vectorized is not None:
            self._H = np.concatenate((vectorized, np.ones((1,))))
        elif homog_array is None:
            self._H = np.zeros(4)
            self._H[3] = 1
        else:
            if homog_array.shape != (4,):
                raise ValueError('Input ndarray must be (4,)')
            self._H = homog_array
        super(Point, self).__init__(reference_frame)

    def q(self):
        return self._H[0:3]

    def homog(self):
        return self._H

    def error(self, other):
        return la.norm(self.q() - other.q())

    @check_reference_frames
    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self.homog() - other.homog(), reference_frame=other._reference_frame)

        elif isinstance(other, Vector):
            return Point(np.append(self.q() - other.q(), 1), reference_frame=other._reference_frame)

        elif isinstance(other, Number):
            return Vector(self._H - other)

        else:
            raise NotImplementedError()

    @check_reference_frames
    def __add__(self, other):
        if isinstance(other, (Point, Vector)):
            return Point(np.append(self.q() - other.q(), 1), reference_frame=other._reference_frame)

        elif isinstance(other, Number):
            return Vector(self._H - other)

        else:
            raise NotImplementedError()

    def __div__(self, other):
        if isinstance(other, float):
            new = self.copy()
            new._H /= other
            new.H[-1] = 1
            return new

        else:
            raise NotImplementedError()

    def __array__(self, dtype=float):
        return self.q().astype(dtype=dtype)

    def norm(self):
        return la.norm(self.q())

    def vectorize(self):
        return self._H[:3]

    def to_dict(self):
        return dict(zip(self.cartesian_names, self.q()))

    def names(self, prefix=''):
        return [prefix + name for name in self.cartesian_names]


def obj_to_joint(orig_obj):
    if 'children' in orig_obj:
        return Joint(**orig_obj)
    elif 'primitive' in orig_obj:
        return Feature(**orig_obj)
    elif 'joint_type' in orig_obj:
        return ParameterizedJoint.from_dict(orig_obj)
    else:
        return orig_obj


class IKSolver(object):
    """Contains the kinematic tree, cost functions, and constraints associated
    with a given inverse kinematics problem.

    Methods allow the IK problem to be solved for different initial
    configurations
    """
    def __init__(self, tree, constraints=[], costs=[]):
        self.tree = tree
        self.constraints = constraints
        self.costs = costs
        self.objective, self.jacobian = self._set_cost_functions()

    #Use a function factory to create the objective and Jacobian functions
    def _set_cost_functions(self, const_weight=1.0):
        #Define the objective function
        def objective(config):
            const_sum = const_weight * sum([const.get_cost(config) for const in self.constraints])
            cost_sum = sum([cost.get_cost(config) for cost in self.costs])
            return const_sum + cost_sum

        #Define the Jacobian of the objective function
        def jacobian(config):
            const_sum = const_weight * sum([const.get_jacobian(config) for const in self.constraints])
            cost_sum = sum([cost.get_jacobian(config) for cost in self.costs])
            return const_sum + cost_sum

        return objective, jacobian

    def solve_ik(self, init_config, weight_consts=True):
        MAX_CONST_WT = 1.0e6
        NUM_ITER = 10
        DEBUG = True
        JAC_TOL = 1e-8

        if weight_consts:
            #Generate the constraint weights for each iteration
            weights = np.logspace(0, log10(MAX_CONST_WT), num=NUM_ITER)

            #Run the optimization
            result = None
            for weight in weights:
                self.objective, self.jacobian = self._set_cost_functions(weight)
                if result is None:
                    result = scipy.optimize.minimize(self.objective, init_config, method='BFGS',
                              jac=self.jacobian, options={'gtol':JAC_TOL})
                    # result = opt.fmin_bfgs(self.objective, init_config, method='BFGS', 
                    #           fprime=self.jacobian, disp=DEBUG, gtol=JAC_TOL)
                else:
                    result = scipy.optimize.minimize(self.objective, result.x, method='BFGS', 
                              jac=self.jacobian, options={'gtol':JAC_TOL})
                    # result = opt.fmin_bfgs(self.objective, result.x, method='BFGS', 
                    #           fprime=self.jacobian, disp=DEBUG, gtol=JAC_TOL)

                #Stop iterating if the optimization failed to converge
                if not result.success:
                    break

        #If we're not weighting the constraint costs, just run once
        else:
            self.objective, self.jacobian = self._set_cost_functions()
            result = scipy.optimize.minimize(self.objective, init_config, method='BFGS', 
                              jac=self.jacobian, options={'gtol':JAC_TOL})
            # result = opt.fmin_bfgs(self.objective, init_config, method='BFGS', 
            #                   fprime=self.jacobian, disp=DEBUG, gtol=JAC_TOL)
        return result


class KinematicCost(object):
    def __init__(self, cost_func, jac_func):
        self.cost_func = cost_func 
        self.jac_func = jac_func

    def get_cost(self, config):
        #Takes a (N,) config and returns a scalar cost
        return self.cost_func(config)

    def get_jacobian(self, config):
        #Takes a (N,) config and returns a (N,) gradient
        return self.jac_func(config)


class KinematicConstraint(KinematicCost):
    def __init__(self, tree, constraint_type, frame, value):
        #TODO: add orientation constraints
        KinematicCost.__init__(self, self._constraint_cost, 
                               self._constraint_jacobian)
        self.tree = tree #The KinematicTree referenced by this constraint
        self.type = constraint_type #Constraint type
        self.frame = frame #Name of the constrained end effector frame
        self.value = value #Desired value of the constrained frame (type depends on self.type)
        #self.type=='position' -> self.value==Point, self.type=='orientation' -> self.value==Rotation
        #Example types: 'position', 'orientation'

    def _constraint_cost(self, config):
        #Get the current value of the end effector transform
        cur_trans = self.tree.get_transform(config, self.frame)

        #Conmpute the value of the constraint depending on its type
        if self.type is 'position':
            diff = self.value.diff(cur_trans.position())
            return diff.norm()**2
        elif self.type is 'orientation':
            raise NotImplementedError('Orientation constraints are not implemented')
        else:
            raise TypeError('Not a valid constraint type')

    def _constraint_jacobian(self, config):
        cur_trans = self.tree.get_transform(config, self.frame)
        cur_jac = self.tree.get_jacobian(config, self.frame)

        #Compute the velocity of the origin of the end effector frame,
        #in spatial frame coordinates, for each joint in the manipulator
        jac_hat = se3.hat(cur_jac) #4 x 4 x N ndarray
        end_vel = np.zeros(jac_hat.shape)
        for i in range(jac_hat.shape[2]):
            end_vel[:,:,i] = jac_hat[:,:,i].dot(cur_trans.homog())
        end_vel = se3.unhat(end_vel)

        if self.type is 'position':
            cost_jac = np.array(config)
            cost_jac = 2 * cur_trans.position().x() - 2 * self.value.x()
            return cost_jac.T.squeeze().dot(end_vel[3:,:])


class QuadraticDisplacementCost(KinematicCost):
    """Kinematic cost which penalizes movement away from a neutral pose.

    The quadratic displacement cost is equal to the squared configuration space 
    distance between the current kinematic configuration and a 
    specified neutral configuration.

    Args:
    neutral_pos - (N,) ndarray: The neutral pose of the manipulator in c-space
    """

    def __init__(self, neutral_pos):
        KinematicCost.__init__(self, self._cost, self._jacobian)
        self.neutral_pos = neutral_pos

    def _cost(self, config):
        return la.norm(config - self.neutral_pos)**2

    def _jacobian(self, config):
        return 2 * config - 2 * self.neutral_pos


class KinematicTree(object):
    # Specify only one of the three sources to load the tree from
    def __init__(self, root=None, json_string=None, json_filename=None):
        if root is not None:
            self._root = root
        elif json_string is not None:
            self._root = json.loads(json_string, object_hook=obj_to_joint, encoding='utf-8')
        elif json_filename is not None:
            with open(json_filename) as json_file:
                self._root = json.load(json_file, object_hook=obj_to_joint, encoding='utf-8')
        self._config = None

        self._pox_stale = True
        self._dpox_stale = True

    def json(self, filename=None):
        return self._root.json(filename, args={'separators':(',',': '), 'indent':4})

    def get_chain(self, feature_name, root=None):
        if root is None:
            root = self._root
        if root.name == feature_name:
            return []
        elif hasattr(root, 'children'):
            # This is a joint
            for child in root.children:
                result = self.get_chain(feature_name, root=child)
                if result is not None:
                    result.insert(0, root.name)
                    return result
        # This is a feature - not the one we're looking for OR this is a joint without the desired
        # feature in any of its child subtrees
        return None

    def compute_transform(self, base_frame_name, target_frame_name):
        # Compute the base to manip transform
        self._compute_pox()
        self._compute_dpox()
        feature_obs = self.observe_features()
        return feature_obs[base_frame_name].inv() * feature_obs[target_frame_name]

    def compute_jacobian(self, base_frame_name, manip_frame_name):
        # Compute the base to manip transform
        self._compute_pox()
        self._compute_dpox()
        feature_obs = self.observe_features()
        base_manip_transform = feature_obs[base_frame_name].inv() * feature_obs[manip_frame_name]
        root_base_transform = feature_obs[base_frame_name]

        # Get all the joints that connect the two frames along the shortest path
        incoming_joints = self.get_chain(base_frame_name)
        outgoing_joints = self.get_chain(manip_frame_name)
        while (len(incoming_joints) > 0 and len(outgoing_joints) > 0 and
                incoming_joints[0] == outgoing_joints[0]):
                incoming_joints.pop(0)
                outgoing_joints.pop(0)

        # Collect all the twists for each chain
        all_joints = self.get_joints()
        incoming_twists = {}
        for i, joint_name in enumerate(incoming_joints):
            try:
                incoming_twists[joint_name] = all_joints[joint_name]._dpox.xi()
            except AttributeError:
                pass # Stationary joint - doesn't have a twist
        outgoing_twists = {}
        for i, joint_name in enumerate(outgoing_joints):
            try:
                outgoing_twists[joint_name] = all_joints[joint_name]._dpox.xi()
            except AttributeError:
                pass #Stationary joint - doesn't have a twist

        # Negate the incoming twists
        for joint in incoming_twists:
            incoming_twists[joint] = -1.0 * incoming_twists[joint]

        # Transform all the twists into the base frame
        outgoing_twists.update(incoming_twists)
        all_twists = outgoing_twists
        root_base_inv_adjoint = root_base_transform.inv().adjoint()
        for joint in all_twists:
            all_twists[joint] = Twist(xi=root_base_inv_adjoint.dot(all_twists[joint]))

        # From each twist, get the linear and angular velocity it will generate at the origin of
        # the manip frame
        manip_velocities = {}
        manip_origin = Point(homog_array=base_manip_transform.homog()[:,3])
        for joint in all_twists:
            linear_vel = all_twists[joint].homog().dot(manip_origin.homog()[:, None])[0:3,0]
            rotational_vel = all_twists[joint].omega()
            manip_velocities[joint] = np.zeros((6,1))
            manip_velocities[joint][0:3, 0] = linear_vel
            manip_velocities[joint][3:6, 0] = rotational_vel
        return manip_velocities # Each velocity is [linear, angular]

    def set_config(self, config, root=None, error_on_missing=True):
        self._pox_stale = True
        self._dpox_stale = True
        if root is None:
            root = self._root
        if hasattr(root, 'children'):
            try:
                root._theta = config[root.name]
            except KeyError:
                if root.twist is not None and error_on_missing:
                    raise ValueError('Config dict is missing an entry for joint: ' + root.name)
            for child_joint in root.children:
                self.set_config(config, root=child_joint, error_on_missing=error_on_missing)

    def get_config(self, root=None, config=None):
        if root is None:
            root = self._root
        if config is None:
            config = {}
        try:
            config[root.name] = root._theta
        except AttributeError:
            # This is an immovable joint (twist=None)
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.get_config(root=child, config=config)
        return config

    def rename_configs(self, name_map=None, prefix='', suffix='', root=None):

        if root is None:
            root = self._root

        if name_map is None:
            root.name = prefix + root.name + suffix

        elif root.name in name_map:
            root.name = prefix + name_map[root.name] + suffix

        if hasattr(root, 'children'):
            for child in root.children:
                self.rename_configs(name_map=name_map, prefix=prefix, suffix=suffix, root=child)

    def _compute_pox(self, root=None, parent_pox=None):
        if self._pox_stale or (root is not None):
            self._pox_stale = False
            if root is None:
                root = self._root
            if parent_pox is None:
                parent_pox = Transform()
            try:
                new_pox_list = root.twist.exp(root._theta)
                root._pox = parent_pox
                for pox in new_pox_list:
                    root._pox = root._pox * pox
            except AttributeError:
                # Root doesn't have a twist (joint is stationary), just copy the parent pox
                root._pox = new_geometric_primitive(parent_pox)
            if hasattr(root, 'children'):
                for child_joint in root.children:
                    self._compute_pox(root=child_joint, parent_pox=root._pox)

    def _compute_dpox(self, root=None, parent_pox=None):
        if self._dpox_stale or (root is not None):
            self._dpox_stale = False
            self._compute_pox()
            if root is None:
                root = self._root
            if parent_pox is None:
                parent_pox = Transform()
            try:
                root._dpox = root.dexp(root._theta)
                for i, dpox in enumerate(root._dpox):
                    root._dpox[i] = Twist(xi=parent_pox.adjoint().dot(dpox.xi()))
            except AttributeError:
                # Root doesn't have a twist (joint is stationary)
                pass
            if hasattr(root, 'children'):
                for child_joint in root.children:
                    self._compute_dpox(root=child_joint, parent_pox=root._pox)

    def observe_features(self, root=None, observations=None):
        self._compute_pox()
        if root is None:
            root = self._root
        if observations is None:
            observations = {}
        if hasattr(root, 'children'):
            for child in root.children:
                self.observe_features(root=child, observations=observations)
        else:
            # This is a feature
            observations[root.name] = root._pox * root.primitive
        return observations

    def set_params(self, params_dict, root=None, error_on_missing=True):
        self._pox_stale = True
        self._dpox_stale = True
        if root is None:
            root = self._root
        if hasattr(root, 'twist') and root.twist is not None:
            try:
                root.twist = params_dict[root.name]
            except KeyError:
                if error_on_missing:
                    raise ValueError('Twist dict is missing an entry for joint: ' + root.name)
        if hasattr(root, 'children'):
            for child in root.children:
                self.set_params(params_dict, root=child, error_on_missing=error_on_missing)

    def set_features(self, features, root=None, error_on_missing=False):
        self._pox_stale = True
        if root is None:
            root = self._root
        if hasattr(root, 'primitive'):
            try:
                root.primitive = features[root.name]
            except KeyError:
                if error_on_missing:
                    raise ValueError('Feature dict is missing an entry for feature: ' + root.name)
        if hasattr(root, 'children'):
            for child in root.children:
                self.set_features(features, root=child, error_on_missing=error_on_missing)

    def get_params(self, root=None, params_dict=None):
        if root is None:
            root = self._root
        if params_dict is None:
            params_dict = {}
        try:
            params_dict[root.name] = root.twist
        except AttributeError:
            # This is an immovable joint
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.get_params(root=child, params_dict=params_dict)
        return params_dict

    def get_features(self, root=None, features=None):
        if root is None:
            root = self._root
        if features is None:
            features = {}
        if hasattr(root, 'children'):
            for child in root.children:
                self.get_features(root=child, features=features)
        else:
            # This is a feature
            features[root.name] = root.primitive
        return features

    def get_joints(self, root=None, joints=None):
        if root is None:
            root = self._root
        if joints is None:
            joints = {}
        if hasattr(root, 'children'):
            joints[root.name] = root
            for child in root.children:
                self.get_joints(root=child, joints=joints)
        return joints

    def get_root_joint(self):
        return self._root

    def compute_error(self, config_dict, feature_obs_dict, vis=False):
        # Assumes that all features in feature_obs_dict will be found in the tree
        # Set configuration and compute pox, assuming config_dict maps joint names to float values
        self.set_config(config_dict)
        self._compute_pox()

        # Assuming feature_obs_dict maps feature names to geometric primitive objects, compute the
        # error between each feature and its actual value
        # TODO: Add an .error(other) method to primitives
        feature_obs = self.observe_features()
        sum_squared_errors = 0

        # Ignore any feature not present in feature_obs_dict
        for feature in feature_obs_dict:
            sum_squared_errors += feature_obs[feature].error(feature_obs_dict[feature]) ** 2

        # Visualize the residuals
        if vis:
            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')

            for feature in feature_obs:
                # Predicted position
                predicted = feature_obs[feature].q()
                axes.scatter(predicted[0], predicted[1], predicted[2], c='r', marker='o')

                # Observed position
                try:
                    observed = feature_obs_dict[feature].q()
                    axes.scatter(observed[0], observed[1], observed[2], c='b', marker='o')
                    endpoints = np.concatenate((observed[:,None], predicted[:,None]), axis=1)
                    axes.plot(endpoints[0,:], endpoints[1,:], endpoints[2,:], 'k-')
                except KeyError:
                    pass
            axes.set_xlabel('X Label')
            axes.set_ylabel('Y Label')
            axes.set_zlabel('Z Label')
            axes.auto_scale_xyz([-0.5,0.5], [-0.5,0.5], [-0.5,0.5])
            plt.ion()
            plt.pause(10)
            plt.close()

        # Compute and return the euclidean sum of the error values for each feature
        return sum_squared_errors

    def compute_sequence_error(self, config_dict_list, feature_obs_dict_list):
        # Interface is the same, except that configs and feature obs are lists of dicts
        if len(config_dict_list) != len(feature_obs_dict_list):
            raise ValueError('Lists of configs and feature obs must be the same length')

        # Compute error for each config using compute_error, then add up
        sum_squared_errors = 0
        for config, feature_obs in zip(config_dict_list, feature_obs_dict_list):
            sum_squared_errors += self.compute_error(config, feature_obs)
        return sum_squared_errors

    def get_objective_function(self, feature_obs_dict_list, **args):
        return KinematicTreeObjectiveFunction(self, feature_obs_dict_list, **args)
    
    def fit_params(self, feature_obs, configs=None, 
            optimize={'configs':True, 'params':True, 'features':True}, print_info=True):
        # TODO: only do this for [0,0,0,1] features
        # Set the feature positions to those seen at the zero configuration
        self.set_features(feature_obs[0])

        # Create an objective function to optimize
        opt = self.get_objective_function(feature_obs, optimize=optimize, config_dict_list=configs)
        initial_params = opt.get_current_param_vector()
        initial_error = opt.error(initial_params)

        # Define a callback for the optimization
        if print_info:
            def opt_callback(current_params):
                print('Current error: ' + str(opt.error(current_params)))
        else:
            opt_callback = None

        # Run the optimization
        result = scipy.optimize.minimize(opt.error, initial_params, callback=opt_callback,
                method='L-BFGS-B')

        # Normalize the twists
        final_configs, final_params, final_features = opt.unvectorize(result.x)
        if final_params is not None:
            for twist in final_params:
                norm_constant = final_params[twist].normalize()
                for config in final_configs:
                    config[twist] = config[twist] * norm_constant

        #Set the tree's features to the optimal values
        if final_features is not None:
            self.set_features(final_features)

        return final_configs, final_params, final_features

    def copy(self):
        return deepcopy(self)


class KinematicTreeParamVectorizer(object):
    def __init__(self):
        self._last_vectorized_sequence = None
        self._vectorize = {'configs':False, 'twists':False, 'features':False}

    def vectorize(self, configs=None, twists=None, features=None):
        # configs - list of dicts of floats
        # twists - dict of ParameterizedJoint objects
        # features - dict of GeometricPrimitive objects
        self._last_vectorized_sequence = []
        vector_list = []

        # Save a list of (type ('config'/'feature'/'twist'), name or (num, name) for config, instance type, length) tuples
        # Add configs
        if configs is not None:
            for i, config in enumerate(configs):
                for joint_name in config:
                    vec_value = config[joint_name]
                    description_tuple = ('config', (i, joint_name), 'array', len(vec_value))
                    self._last_vectorized_sequence.append(description_tuple)
                    vector_list.extend(vec_value)
            self._vectorize['configs'] = True
        else:
            self._vectorize['configs'] = False

        # Add twists
        if twists is not None:
            for joint_name in twists:
                if twists[joint_name] is not None:
                    vec_value = twists[joint_name].vectorize()
                    dict_description = twists[joint_name].to_dict()
                    del dict_description['params']
                    description_tuple = ('twist', joint_name, dict_description, len(vec_value))
                    self._last_vectorized_sequence.append(description_tuple)
                    vector_list.extend(vec_value)
            self._vectorize['twists'] = True
        else:
            self._vectorize['twists'] = False

        # Add features
        if features is not None:
            for feature_name in features:
                vec_value = features[feature_name].vectorize()
                description_tuple = ('feature', feature_name, type(features[feature_name]),
                        len(vec_value))
                self._last_vectorized_sequence.append(description_tuple)
                vector_list.extend(vec_value)
            self._vectorize['features'] = True
        else:
            self._vectorize['features'] = False

        return np.array(vector_list)

    def unvectorize(self, vectorized_params, prefix=''):
        # Shouldn't touch the tree, should operate on params and the order saved by vectorize()
        # Return config dict list, twists dict, features dict
        vector_ind = 0
        if self._vectorize['configs']:
            configs = []
        else:
            configs = None
        if self._vectorize['twists']:
            twists = {}
        else:
            twists = None
        if self._vectorize['features']:
            features = {}
        else:
            features = None

        for desc_tuple in self._last_vectorized_sequence:
            # Pull out the vector value and description tuple for this config/twist/feature
            vec_value = vectorized_params[vector_ind:vector_ind+desc_tuple[3]]
            vector_ind += desc_tuple[3]

            # Reconstruct the original data structures
            if desc_tuple[0] == 'config':
                if self._vectorize['configs']:
                    config = np.array(vec_value)
                    config_idx = desc_tuple[1][0]
                    name = desc_tuple[1][1]
                    while len(configs) < config_idx + 1:
                        configs.append({})
                    configs[config_idx][prefix+name] = config
            elif desc_tuple[0] == 'twist':
                if self._vectorize['twists']:
                    joint_type_dict = desc_tuple[2]
                    joint_type_dict['params'] = vec_value
                    twist = ParameterizedJoint.from_dict(joint_type_dict)
                    name = desc_tuple[1]
                    twists[prefix+name] = twist
            elif desc_tuple[0] == 'feature':
                if self._vectorize['features']:
                    feature = desc_tuple[2](vectorized=vec_value)
                    name = desc_tuple[1]
                    features[prefix+name] = feature
            else:
                raise ValueError('Invalid vectorized type: ' + desc_tuple[0])
        return configs, twists, features


class KinematicTreeStateSpaceModel(StateSpaceModel):
    def __init__(self, tree):
        self._tree = tree

        # Initialize the state vectorizer to output only config values
        self._state_vectorizer = KinematicTreeParamVectorizer()
        tree_joints = self._tree.get_joints()
        initial_config = {joint:0.0 for joint in tree_joints if tree_joints[joint].twist is not None}
        self._state_vectorizer.vectorize(configs=[initial_config])

        # Initialize the measurement vectorizer to output only feature values
        self._meas_vectorizer = KinematicTreeParamVectorizer()
        self._meas_vectorizer.vectorize(features=self._tree.get_features())
        self._state_length = len(initial_config)
        self._state_names = initial_config.keys()

    def measurement_model(self, state_vector):
        """Returns a vectorized observation of predicted feature poses given state=state_vector.

        Returned array is (sum(len(feature_i.vectorize())),).
        """
        # Shape array correctly
        state_vector = state_vector.squeeze()
        if state_vector.ndim < 2:
            state_vector = state_vector[:,None]
        output_arr = None

        for j in range(state_vector.shape[1]):
            # Turn config_vector back into a config dict
            config_dict = self._state_vectorizer.unvectorize(state_vector[:,j])[0][0]

            # Set the kinematic tree to the correct config, observe features, vectorize, and return
            self._tree.set_config(config_dict)
            self._tree._compute_pox()
            feature_obs = self._tree.observe_features()
            output_obs = self._meas_vectorizer.vectorize(features=feature_obs)

            # Write to the output array
            if output_arr is None:
                # Initialize here so we know what the length of each measurment vector will be
                output_arr = np.zeros((len(output_obs), state_vector.shape[1]))
            output_arr[:,j] = output_obs
        return output_arr

    def process_model(self, state_vector):
        # Assume trivial dynamics
        return state_vector

    def vectorize_measurement(self, feature_obs):
        return self._meas_vectorizer.vectorize(features=feature_obs)[:,None]

    def unvectorize_estimation(self, state_vector, prefix=''):
        return self._state_vectorizer.unvectorize(state_vector, prefix)[0][0]


class WristStateSpaceModel2(StateSpaceModel, MocapWrist):
    """
    Currently unused. Describes the system whereby mocap points are measurements and the transformation between hand
    frame and arm frame are the states. This is not very elegant since we can compute this transformation directly but
    may be useful later if we eventually model the wrist
    """
    def __init__(self, reference_features):

        assert all(feature in reference_features for feature in self.names), \
            "reference_frames must contain all these keys %s" % self.names

        self.reference_features = reference_features

        # Initialize the state vectorizer to output only config values
        # We extend our state vector to include the translation we don't care about but need for predicting measurements
        translation_configs = ['x', 'y', 'z']
        self._state_vectorizer = KinematicTreeParamVectorizer()
        initial_config = {config: 0.0 for config in self.configs + translation_configs}
        self._state_vectorizer.vectorize([initial_config])

        # Initialize the measurement vectorizer to output only feature values
        self._meas_vectorizer = KinematicTreeParamVectorizer()
        initial_features = {feature: Point() for feature in self.reference_features}
        self._meas_vectorizer.vectorize(features=initial_features)
        self._state_length = len(initial_config)

    def _state_vector_to_transform(self, state_vector):
        """
        'Unflattens' the the state vector back into the homogenous transform it represents
        :param state_vector: (6,) numpy array
        :return: (4, 4) numpy array
        """
        configs = self._state_vectorizer.unvectorize(state_vector)[0]
        homog = np.zeros((4,4))
        homog[4, 4] = 1
        homog[:3, :3] = euler_matrix(configs['roll'], configs['pitch'], configs['yaw'])
        homog[:3, 3] = np.array([configs['x'], configs['y'], configs['z']])
        return Transform(homog)

    def measurement_model(self, state_vector):
        """
        Transforms all the points by the transform represented in the state vector
        """
        transform = self._configs_to_transform(state_vector)
        meas_features = {feature: transform*self.reference_features[feature] for feature in self.names}
        return self._meas_vectorizer.vectorize(meas_features)

    def process_model(self, state_vector):
        """
        Identity process
        """
        return state_vector

    def vectorize_measurement(self, feature_obs):
        self._meas_vectorizer.vectorize(feature_obs)


class WristStateSpaceModel(StateSpaceModel, MocapWrist):
    """
    Uses the euler angles describing the rotation about the wrist as both the states and the measurement with identity
    process and measurement models.
    """
    def __init__(self):

        # Initialize the state vectorizer to output only config values
        # We extend our state vector to include the translation we don't care about but need for predicting measurements
        self._state_vectorizer = KinematicTreeParamVectorizer()
        initial_config = {config: 0.0 for config in self.configs}
        self._state_vectorizer.vectorize(configs=[initial_config])

        # Initialize the measurement vectorizer to output only config values
        self._meas_vectorizer = KinematicTreeParamVectorizer()
        self._meas_vectorizer.vectorize(configs=[initial_config])

        self._state_length = len(initial_config)

    def measurement_model(self, state_vector):
        """Identity"""
        return state_vector

    def process_model(self, state_vector):
        """Identity"""
        return state_vector

    def vectorize_measurement(self, configs_obs):
        return self._meas_vectorizer.vectorize(configs=[configs_obs])

    def unvectorize_estimation(self, state_vector, prefix=''):
        return self._state_vectorizer.unvectorize(state_vector, prefix)[0][0]


class KinematicTreeObjectiveFunction(object):
    def __init__(self, kinematic_tree, feature_obs_dict_list, config_dict_list=None,
            optimize={'configs':True, 'params':True, 'features':True}):
        self._tree = kinematic_tree
        if optimize['features'] and optimize['features'] is not True:
            self._feature_obs = [{name:feature_obs[name] for name in optimize['features']} for feature_obs in feature_obs_dict_list]
        else:
            self._feature_obs = feature_obs_dict_list
        if config_dict_list is None:
            params = self._tree.get_params()

            # Don't optimize the config of immovable joints
            zero_config = {name: np.zeros(params[name].config_shape()) for name in params if params[name] is not None}
            self._config_dict_list = [zero_config.copy() for config in feature_obs_dict_list]
        else:
            if len(config_dict_list) != len(feature_obs_dict_list):
                raise ValueError('Must have same num of feature obs and config initial guesses')
            self._config_dict_list = config_dict_list
        self._vectorizer = KinematicTreeParamVectorizer()
        self._optimize = optimize

    def get_current_param_vector(self):
        # Pull params from KinematicTree and pass to vectorize
        if self._optimize['params']:
            params = self._tree.get_params()
            if self._optimize['params'] is not True:
                # Select only the specified params to optimize if a list of names is given
                params = {name:params[name] for name in self._optimize['params']}
        else:
            params = None
        if self._optimize['features']:
            features = self._tree.get_features()
            if self._optimize['features'] is not True:
                # Select only the specified features to optimize if a list of names is given
                features = {name:features[name] for name in self._optimize['features']}
        else:
            features = None
        if self._optimize['configs']:
            configs = self._config_dict_list[1:]
            if self._optimize['configs'] is not True:
                # Select only the specified configs to optimize if a list of names is given
                configs = [{name:config_dict[name] for name in self._optimize['configs']} for config_dict in configs]
        else:
            configs = None

        # Vectorize and return
        return self._vectorizer.vectorize(configs, params, features)

    def error(self, vectorized_params):
        # Unvectorize params
        # Use self.unvectorize() so the zero config is handled correctly
        configs, params, features = self.unvectorize(vectorized_params)
        if configs is not None:
            self._config_dict_list = configs

        # Set features and params on tree
        if features is not None:
            self._tree.set_features(features)
        if params is not None:
            self._tree.set_params(params)

        # Compute error
        return self._tree.compute_sequence_error(self._config_dict_list, self._feature_obs)

    def unvectorize(self, vectorized_params):
        configs, twists, features = self._vectorizer.unvectorize(vectorized_params)
        if configs is not None:
            configs.insert(0, self._config_dict_list[0])
        return configs, twists, features


def generate_synthetic_observations(tree, num_obs=100):
    # Get the state dimension of each movable joint in the tree
    movable_joint_dims = {name:np.asarray(tree.get_config()[name]).shape for name in tree.get_params() if tree.get_params()[name] is not None}

    # Generate random combinations of joint angles and output to a list of dicts
    configs = []
    configs.append({name:np.zeros(movable_joint_dims[name]) for name in movable_joint_dims})
    for i in range(num_obs - 1):
        configs.append({name:((2*pi*nprand.random(movable_joint_dims[name]))-pi) for name in movable_joint_dims})

    # Observe features for each config and make a list of feature obs dicts
    feature_obs_dict_list = []
    for config in configs:
        tree.set_config(config)
        feature_obs_dict_list.append(tree.observe_features())
    return configs, feature_obs_dict_list


def differentiate(start, end, delta_time):
    delta_time = float(delta_time)
    if isinstance(start, Number) and isinstance(end, Number):
        return (end - start) / delta_time

    assert type(start) == type(end), "Types do not match (%s and %s)" % (type(start), type(end))

    if hasattr(start, '__sub__'):
        diff = end - start


def main():
    # Construct the kinematic chain
    j0 = Joint('joint0')
    j1 = Joint('joint1')
    j2 = Joint('joint2')

    j1.twist = ThreeDofBallJoint(np.ones(3))
    j2.twist = OneDofTwistJoint(np.array([1,0,0,0,0,0]))

    ft1 = Feature('feat1', Point(np.array([1,2,3,1])))
    ft2 = Feature('feat2', Point(np.array([3,2,1,1])))
    ft3 = Feature('feat3', Point(np.array([2,2,2,1])))
    ft4 = Feature('feat4', Point(np.array([5,1,2,1])))

    j0.children.append(j1)
    j1.children.append(ft1)
    j1.children.append(ft2)
    j1.children.append(j2)
    j2.children.append(ft3)
    j2.children.append(ft4)

    # Test JSON saving and loading
    tree = KinematicTree(j0)
    json_string_1 = tree.json()
    tree.json(filename='kinmodel_test_1.json')

    test_decode = KinematicTree(json_filename='kinmodel_test_1.json')
    json_string_2 = test_decode.json()
    assert json_string_1 == json_string_2, 'JSON saving/loading produces a different KinematicTree'

    #Test parameter optimization
    tree.set_config({'joint1':[0, 0, 0], 'joint2':[0]})

    configs, feature_obs = generate_synthetic_observations(tree, 20)
    final_configs, final_twists, final_features = tree.fit_params(feature_obs, configs=None, 
            optimize={'configs':True, 'params':True, 'features':False})
    1/0

if __name__ == '__main__':
    main()
