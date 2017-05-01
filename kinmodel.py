from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
import numpy.linalg as la
import scipy.optimize
import se3
from math import pi, log10, sqrt
import json
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def new_geometric_primitive(input_data):
    homog_array = None
    #If the input is a GeometricPrimitive
    try:
        homog_array = input_data.homog()
    except AttributeError:
        #Otherwise, if it's an array-like object
        homog_array = np.asarray(input_data)
        homog_array = homog_array.squeeze()

    if homog_array.shape == (4,4):
        return Transform(homog_array)
    elif homog_array.shape == (3,3):
        return Rotation(homog_array)
    elif homog_array.shape == (4,) and homog_array[3] == 1:
        return Point(homog_array)
    elif homog_array.shape == (3,):
        return Point()
    elif homog_array.shape == (4,) and homog_array[3] == 0:
        return Vector(homog_array)
    elif homog_array.shape == (6,):
        return Twist(omega=homog_array[0:3], nu=homog_array[3:6])
    else:
        raise TypeError('input_data must be array-like or a GeometricPrimitive')


#Geometric primitives
class GeometricPrimitive(object):
    __metaclass__ = ABCMeta

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

    def _json(self):
        return self.homog().squeeze().tolist()

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

        .json(filename) - saves the kinematic tree to the specified file

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

        if twist is None:
            self.twist = None
        elif hasattr(twist, '_xi'):
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

class Transform(GeometricPrimitive):
    """
    g = Transform(...)  element of SE(3)

        .rot() - Rotation object
        .trans() - Vector object
        .homog() - (4,4) - homogeneous transformation ndarray
        .p() - (3,) - translation ndarray
        .R() - (3,3) - rotation ndarray
    """
    def __init__(self, homog_array=None):
        if homog_array is None:
            self._H = np.identity(4)
        else:
            if homog_array.shape != (4,4):
                raise ValueError('Input ndarray must be (4,4)')
            self._H = homog_array

    def homog(self):
        return self._H

    def inv(self):
        return Transform(homog_array=la.inv(self.homog()))

    def trans(self):
        p = np.append(self._H[0:3,3], 0)
        return Vector(p)

    def rot(self):
        return Rotation(self._H.copy())

    def p(self):
        return self._H[0:3,3]

    def R(self):
        return self._H[0:3,0:3]

    def adjoint(self):
        adj = np.zeros((6,6))
        adj[:3,:3] = self.R()
        adj[3:,3:] = self.R()
        adj[3:,:3] = se3.skew(self.p()).dot(self.R())
        return adj


class Twist(object):
    """ xi = Twist(...)  element of se(3)

        .xi() - 6 x 1 - twist coordinates ndarray (om, v)
        .omega() - 3 x 1 - rotation axis ndarray
        .nu() - 3 x 1 - translation direction ndarray
        .exp(theta) - Transform object

        ._xi - 6 x 1 - twist coordinates (om, v)
    """
    def __init__(self, omega=None, nu=None, copy=None, vectorized=None, xi=None):
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

class Rotation(GeometricPrimitive):
    """ R = Rotation(...)  element of SO(3)

        .R() - (3,3) - rotation matrix ndarray
        .homog() - (4,4) - homogeneous coordinates ndarray (for a pure rotation)
    """
    def __init__(self, homog_array=None):
        if homog_array is None:
            self._R = np.identity(3)
        else:
            if homog_array.shape != (4,4):
                raise ValueError('Input ndarray must be (4,4)')
            self._R = homog_array[0:3,0:3]

    def R(self):
        return self._R

    def homog(self):
        homog_matrix = np.zeros(4)
        homog_matrix[0:3,0:3] = self._R
        return homog_matrix


class Vector(GeometricPrimitive):
    """ x = Vector(...)  translation in R^3

        .homog() - (4,) - homogeneous coordinates ndarray (x, 0)
        .q() - (3,) - cartesian coordinates ndarray
    """
    def __init__(self, homog_array=None):
        if homog_array is None:
            self._H = np.zeros(4)
        else:
            if homog_array.shape != (4,):
                raise ValueError('Input ndarray must be (4,)')
            self._H = homog_array

    def q(self):
        return self._H[0:3]

    def homog(self):
        return self._H

    def norm(self):
        return la.norm(self.q())


class Point(GeometricPrimitive):
    """ x = Point(...)  point in R^3

        .homog() - (4,) - homogeneous coordinates ndarray (x, 0)
        .q() - (3,) - cartesian coordinates ndarray
    """
    def __init__(self, homog_array=None, vectorized=None):
        if vectorized is not None:
            self._H = np.concatenate((vectorized, np.ones((1,))))
        elif homog_array is None:
            self._H = np.zeros(4)
            self._H[3] = 1
        else:
            if homog_array.shape != (4,):
                raise ValueError('Input ndarray must be (4,)')
            self._H = homog_array

    def q(self):
        return self._H[0:3]

    def homog(self):
        return self._H

    def error(self, other):
        return la.norm(self.q() - other.q())

    # def diff(self, x):
    #     return Vector(x=self.q() - x.q())

    def norm(self):
        return la.norm(self.q())

    def vectorize(self):
        return self._H[:3]

def obj_to_joint(orig_obj):
    if 'children' in orig_obj:
        return Joint(**orig_obj)
    elif 'primitive' in orig_obj:
        return Feature(**orig_obj)
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
                    result = opt.minimize(self.objective, init_config, method='BFGS', 
                              jac=self.jacobian, options={'gtol':JAC_TOL})
                    # result = opt.fmin_bfgs(self.objective, init_config, method='BFGS', 
                    #           fprime=self.jacobian, disp=DEBUG, gtol=JAC_TOL)
                else:
                    result = opt.minimize(self.objective, result.x, method='BFGS', 
                              jac=self.jacobian, options={'gtol':JAC_TOL})
                    # result = opt.fmin_bfgs(self.objective, result.x, method='BFGS', 
                    #           fprime=self.jacobian, disp=DEBUG, gtol=JAC_TOL)

                #Stop iterating if the optimization failed to converge
                if not result.success:
                    break

        #If we're not weighting the constraint costs, just run once
        else:
            self.objective, self.jacobian = self._set_cost_functions()
            result = opt.minimize(self.objective, init_config, method='BFGS', 
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
    def __init__(self, tree, type, frame, value):
        #TODO: add orientation constraints
        KinematicCost.__init__(self, self._constraint_cost, 
                               self._constraint_jacobian)
        self.tree = tree #The KinematicTree referenced by this constraint
        self.type = type #Constraint type
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

    def _compute_pox(self, root=None, parent_pox=None):
        if self._pox_stale or (root is not None):
            self._pox_stale = False
            if root is None:
                root = self._root
            if parent_pox is None:
                parent_pox = Transform()
            try:
                root._pox = parent_pox * root.twist.exp(root._theta)
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
                root._dpox = Twist(xi=(parent_pox.adjoint().dot(root.twist.xi())))
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

    def set_twists(self, twists, root=None, error_on_missing=True):
        self._pox_stale = True
        self._dpox_stale = True
        if root is None:
            root = self._root
        if hasattr(root, 'twist') and root.twist is not None:
            try:
                root.twist = twists[root.name]
            except KeyError:
                if error_on_missing:
                    raise ValueError('Twist dict is missing an entry for joint: ' + root.name)
        if hasattr(root, 'children'):
            for child in root.children:
                self.set_twists(twists, root=child, error_on_missing=error_on_missing)

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

    def get_twists(self, root=None, twists=None):
        if root is None:
            root = self._root
        if twists is None:
            twists = {}
        try:
            twists[root.name] = root.twist
        except AttributeError:
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.get_twists(root=child, twists=twists)
        return twists

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
        # Set configuration and compute pox, assuming config_dict maps joint names to float values
        self.set_config(config_dict)
        self._compute_pox()

        # Assuming feature_obs_dict maps feature names to geometric primitive objects, compute the
        # error between each feature and its actual value (add an .error(other) method to primitives)
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
            optimize={'configs':True, 'twists':True, 'features':True}, print_info=True):
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
        final_configs, final_twists, final_features = opt.unvectorize(result.x)
        if final_twists is not None:
            for twist in final_twists:
                norm_constant = final_twists[twist].normalize()
                for config in final_configs:
                    config[twist] = config[twist] * norm_constant
            self.set_twists(final_twists)

        #Set the tree's features to the optimal values
        if final_features is not None:
            self.set_features(final_features)

        return final_configs, final_twists, final_features

class KinematicTreeParamVectorizer(object):
    def __init__(self):
        self._last_vectorized_sequence = None
        self._vectorize = {'configs':False, 'twists':False, 'features':False}

    def vectorize(self, configs=None, twists=None, features=None):
        # configs - list of dicts of floats
        # twists - dict of Twist objects
        # features - dict of GeometricPrimitive objects
        self._last_vectorized_sequence = []
        vector_list = []

        # Save a list of (type ('config'/'feature'/'twist'), name or (num, name) for config, instance type, length) tuples
        # Add configs
        if configs is not None:
            for i, config in enumerate(configs):
                for joint_name in config:
                    description_tuple = ('config', (i, joint_name), 'int', 1)
                    self._last_vectorized_sequence.append(description_tuple)
                    vector_list.append(config[joint_name])
            self._vectorize['configs'] = True
        else:
            self._vectorize['configs'] = False

        # Add twists
        if twists is not None:
            for joint_name in twists:
                if twists[joint_name] is not None:
                    vec_value = twists[joint_name].vectorize()
                    description_tuple = ('twist', joint_name, type(twists[joint_name]), len(vec_value))
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

    def unvectorize(self, vectorized_params):
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
                    theta = vec_value[0]
                    config_idx = desc_tuple[1][0]
                    name = desc_tuple[1][1]
                    while len(configs) < config_idx + 1:
                        configs.append({})
                    configs[config_idx][name] = theta
            elif desc_tuple[0] == 'twist':
                if self._vectorize['twists']:
                    twist = desc_tuple[2](vectorized=vec_value)
                    name = desc_tuple[1]
                    twists[name] = twist
            elif desc_tuple[0] == 'feature':
                if self._vectorize['features']:
                    feature = desc_tuple[2](vectorized=vec_value)
                    name = desc_tuple[1]
                    features[name] = feature
            else:
                raise ValueError('Invalid vectorized type: ' + desc_tuple[0])
        return configs, twists, features


class KinematicTreeStateSpaceModel(object):
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

        
class KinematicTreeObjectiveFunction(object):
    def __init__(self, kinematic_tree, feature_obs_dict_list, config_dict_list=None,
            optimize={'configs':True, 'twists':True, 'features':True}):
        self._tree = kinematic_tree
        if optimize['features'] and optimize['features'] is not True:
            self._feature_obs = [{name:feature_obs[name] for name in optimize['features']} for feature_obs in feature_obs_dict_list]
        else:
            self._feature_obs = feature_obs_dict_list
        if config_dict_list is None:
            twists = self._tree.get_twists()

            # Don't optimize the config of immovable joints
            zero_config = {name: 0.0 for name in twists if twists[name] is not None}
            self._config_dict_list = [zero_config.copy() for config in feature_obs_dict_list]
        else:
            if len(config_dict_list) != len(feature_obs_dict_list):
                raise ValueError('Must have same num of feature obs and config initial guesses')
            self._config_dict_list = config_dict_list
        self._vectorizer = KinematicTreeParamVectorizer()
        self._optimize = optimize

    def get_current_param_vector(self):
        # Pull params from KinematicTree and pass to vectorize
        if self._optimize['twists']:
            twists = self._tree.get_twists()
            if self._optimize['twists'] is not True:
                # Select only the specified twists to optimize if a list of names is given
                twists = {name:twists[name] for name in self._optimize['twists']}
        else:
            twists = None
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
        return self._vectorizer.vectorize(configs, twists, features)

    def error(self, vectorized_params):
        # Unvectorize params
        # Use self.unvectorize() so the zero config is handled correctly
        configs, twists, features = self.unvectorize(vectorized_params)
        if configs is not None:
            self._config_dict_list = configs

        # Set features and twists on tree
        if features is not None:
            self._tree.set_features(features)
        if twists is not None:
            self._tree.set_twists(twists)

        # Compute error
        return self._tree.compute_sequence_error(self._config_dict_list, self._feature_obs)

    def unvectorize(self, vectorized_params):
        configs, twists, features = self._vectorizer.unvectorize(vectorized_params)
        if configs is not None:
            configs.insert(0, self._config_dict_list[0])
        return configs, twists, features

def generate_synthetic_observations(tree, num_obs=100):
    # Get all the movable joints in the tree
    movable_joint_names = [name for name in tree.get_twists() if name is not None]

    # Generate random combinations of joint angles and output to a list of dicts
    configs = []
    configs.append({name: 0.0 for name in movable_joint_names})
    for i in range(num_obs - 1):
        config = {}
        for name in movable_joint_names:
            config[name] = random.uniform(-3.14, 3.14)
        configs.append(config)

    # Observe features for each config and make a list of feature obs dicts
    feature_obs_dict_list = []
    for config in configs:
        tree.set_config(config)
        tree._compute_pox()
        feature_obs_dict_list.append(tree.observe_features())
    return configs, feature_obs_dict_list


def main():
    j0 = Joint('joint0')
    j1 = Joint('joint1')
    j2 = Joint('joint2')

    j1.twist = Twist(omega=[0,0,1], nu=[1,1,0])
    j2.twist = Twist(omega=[0,0,1], nu=[1,-1,0])

    ft1 = Feature('feat1', Point())
    ft2 = Feature('feat2', Point())
    trans1 = np.array([[ 0, 1, 0,-1],
                       [-1, 0, 0, 3],
                       [ 0, 0, 1, 0],
                       [ 0, 0, 0, 1]])
    trans2 = np.array([[ 1, 0, 0, 3],
                       [ 0, 1, 0, 1],
                       [ 0, 0, 1, 0],
                       [ 0, 0, 0, 1]])
    trans1 = Feature('A', Transform(homog_array=trans1))
    trans2 = Feature('B', Transform(homog_array=trans2))

    j0.children.append(j1)
    j0.children.append(j2)
    j1.children.append(ft1)
    j2.children.append(ft2)
    # j1.children.append(trans1)
    # j2.children.append(trans2)

    tree = KinematicTree(j0)

    string = tree.json()
    with open('base_kinmodel.json', 'w') as json_file:
        json_file.write(string)

    test_decode = json.loads(string, object_hook=obj_to_joint, encoding='utf-8')

    tree.set_config({'joint1':0.0, 'joint2':pi/2})
    # tree.compute_jacobian('A', 'B')

    configs, feature_obs = generate_synthetic_observations(tree, 20)
    final_configs, final_twists, final_features = tree.fit_params(feature_obs, configs=None, 
            optimize={'configs':True, 'twists':True, 'features':True})
    1/0

if __name__ == '__main__':
    main()
