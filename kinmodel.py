from abc import ABCMeta, abstractmethod
#from new_skel import Skel
from collections import OrderedDict
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import se3
from math import pi, log10, sqrt
import json

FILENAME = '../data/scara.bvh'
VARS = {'l_0':0.5, 'l_1':2.0, 'l_2':1.0}

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
    else:
        raise TypeError('input_data must be array-like or a GeometricPrimitive')


#Geometric primitives
class GeometricPrimitive():
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

class Feature:
    """ j = Feature(...) a feature in a kinematic tree

        .name - the name of the feature (must be unique in this tree)
        .primitive - the geometric primitive of this feature (wrt the base coordinate frame)
    """
    def __init__(self, name, primitive):
        self.name = name
        if hasattr(primitive, 'homog'):
            self.primitive = primitive
        else:
            self.primitive = list_to_primitive(primitive)

    def _json(self):
        OUTPUT_ATTRIBS = ['name', 'primitive']
        json_dict = OrderedDict()
        for attrib in OUTPUT_ATTRIBS:
            try:
                json_dict[attrib] = getattr(self, attrib)
            except AttributeError:
                pass
        return json_dict


class Joint:
    """ j = Joint(...) a joint in a kinematic tree

        .json(filename) - saves the kinematic tree to the specified file

        .name - the name of this joint (must be unique in this tree)
        .children - list of other Joint objects
        .twist - the twist coordinates of the joint (only present if not the root)
    """
    def __init__(self, name, children=None, twist=None):
        self.name = name
        if children is not None:
            children = children
        else:
            children = []
        self.children = children #Children of the joint (other Joints)
        if hasattr(twist, '_xi'):
            self.twist = twist
        else:
            self.twist = list_to_primitive(twist)

    def json(self, filename=None, args={}):
        if filename is None:
                return json.dumps(self, default=lambda o: o._json(), **args)
        else:
            with open(filename) as output_file:
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
        adj[:3,:3] = self.R
        adj[3:,3:] = self.R
        adj[3:,:3] = se3.skew(self.p).dot(self.R)
        return adj


class Twist:
    """ xi = Twist(...)  element of se(3)

        .xi() - 6 x 1 - twist coordinates ndarray (om, v)
        .omega() - 3 x 1 - rotation axis ndarray
        .nu() - 3 x 1 - translation direction ndarray
        .exp(theta) - Transform object

        ._xi - 6 x 1 - twist coordinates (om, v)
    """
    def __init__(self, omega=None, nu=None, copy=None, vectorized=None):
        if copy is not None:
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

    def xi(self):
        return _xi

    def omega(self):
        return _xi[:3,:]

    def nu(self):
        return _xi[3:,:]

    def exp(self, theta):
        return Transform(homog_array=se3.expse3(self._xi, theta))

    def vectorize(self):
    	return np.array(self._xi)

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


def list_to_primitive(orig_obj):
    try:
        arr = np.array(orig_obj)
        if arr.shape == (4,):
            if arr[3] == 1:
                return Point(x=arr[0:3])
            elif arr[3] == 0:
                return Vector(x=arr[0:3])
        elif arr.shape == (4,4):
            return Transform(homog=arr)
        elif arr.shape == (3,3):
            return Rotation(matrix=arr)
        elif arr.shape == (6,):
            return Twist(omega=arr[0:3], nu=arr[3:6])
    except TypeError:
        pass
    return orig_obj

def obj_to_joint(orig_obj):
    if 'children' in orig_obj:
        return Joint(**orig_obj)
    elif 'primitive' in orig_obj:
        return Feature(**orig_obj)
    else:
        return orig_obj

    # else:
    #     raise TypeError('The list could not be converted to a geometric primitive')

class IKSolver:
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


class KinematicCost:
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

class KinematicTree():
    def __init__(self, root):
        self._root = root
        self._config = None

    def json(self, filename=None):
        return self._root.json(filename, args={'separators':(',',': '), 'indent':4})

    def get_transform(self, config, name):
        #Set the current kinematic configuration
        self._set_config(config)
 
        #Compute the transform
        frame = self._named_frames[name]
        g_st = Transform(homog=frame._parent_node.pox)
        g_st = g_st.dot(frame)
        return g_st

    # def get_jacobian(self, config, name):
    #     #Returns the spatial frame Jacobian for the manipulator
    #     self._set_config(config)
    #     frame = self._named_frames[name]
    #     return np.hstack([j.dpox[1] for j in frame._parent_node.chain()])
    #     return frame._parent_node.dpox[1]

    def set_config(self, config, root=None):
        if root is None:
            root = self._root
        if hasattr(root, 'children'):
            try:
                root._theta = config[root.name]
            except KeyError:
                if root.twist is not None:
                    raise ValueError('Config dict is missing an entry for joint: ' + root.name)
            for child_joint in root.children:
                self.set_config(config, root=child_joint)

    def get_config(self, root=None, config=None):
        if root is None:
            root = self._root
        if config is None:
            config = {}
        try:
            config[root.name] = root._theta
        except AttributeError:
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.get_config(root=child, config=config)
        return config

    def compute_pox(self, root=None, parent_pox=None):
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
                self.compute_pox(root=child_joint, parent_pox=root._pox)

    def observe_features(self, root=None, observations=None):
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

    def set_twists(self, twists, root=None):
        if root is None:
            root = self._root
        try:
            root.twist = twists[root.name]
        except KeyError:
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.set_twists(twists, root=child)

    def set_features(self, features, root=None):
        if root is None:
            root = self._root
        try:
            root.primitive = features[root.name]
        except KeyError:
            pass
        if hasattr(root, 'children'):
            for child in root.children:
                self.set_features(features, root=child)

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

    def compute_error(self, config_dict, feature_obs_dict):
    	# Set configuration and compute pox, assuming config_dict maps joint names to float values
    	self.set_config(config_dict)
    	self.compute_pox()

    	# Assuming feature_obs_dict maps feature names to geometric primitive objects, compute the
    	# error between each feature and its actual value (add an .error(other) method to primitives)
    	feature_obs = self.observe_features()
    	sum_squared_errors = 0
    	for feature in feature_obs:
    		sum_squared_errors += feature_obs[feature].error(feature_obs_dict[feature]) ** 2

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
        

def main():
    j1 = Joint('joint1')
    j2 = Joint('joint2')
    j3 = Joint('joint3')

    j2.twist = Twist(omega=[1,0.0,0], nu=[1,2,0.0])
    j3.twist = Twist(omega=[0,1.0,0], nu=[1,4,0.0])

    ft1 = Feature('feat1', Point())
    ft2 = Feature('feat2', Point())

    j1.children.append(j2)
    j1.children.append(j3)
    j2.children.append(ft1)
    j3.children.append(ft2)

    tree = KinematicTree(j1)

    string = tree.json()

    test_decode = json.loads(string, object_hook=obj_to_joint, encoding='utf-8')

    tree.set_config({'joint2':0.0, 'joint3':0.0})
    tree.compute_pox()

    # #Create a new Skel object with the correct params
    # skel = Skel()

    # #Load a test BVH file into the skeleton
    # skel.read(FILENAME, vars=VARS)
    # kin_tree = KinematicTree(skel)

    # #Make an end effector position constraint
    # constraint1 = KinematicConstraint(kin_tree, 'position', 'endpoint_0', Point(x=np.array([0,3,0.5])[:,None]))
    
    # #Make a pose cost function
    # constraint2 = QuadraticDisplacementCost(np.array([0,0,.5,0.5]))

    # #Test the optimization
    # ik_sol = IKSolver(kin_tree, constraints=[constraint1], 
    #                   costs=[constraint2])
    # result = ik_sol.solve_ik(np.array([.2,.2,0,0.0]))
    # print(result)
    1/0

if __name__ == '__main__':
    main()
