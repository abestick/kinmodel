from sympy import Symbol, Matrix, eye, simplify, sin, cos, diag, MatrixBase
from sympy.printing.theanocode import theano_function
from kinmodel import KinematicTree, Point, Feature, StateSpaceModel
import numpy as np


def simplified(f):
    def wrapper(*args, **kwargs):
        return simplify(f(*args, **kwargs), ratio=1.0)
    return wrapper


def sym_skew(v):
    """

    :param Matrix v:
    :return:
    """
    return Matrix([[0,       -v[2],      v[1]],
                   [v[2],    0,          -v[0]],
                   [-v[1],   v[0],       0]])


def sym_exp_so3(omega, theta):
    v = 1 - cos(theta)
    c = cos(theta)
    s = sin(theta)
    w1, w2, w3 = omega

    return Matrix([[w1**2 * v + c,          w1 * w2 * v - w3 * s,       w1 * w3 * v + w2 * s],
                   [w1 * w2 * v + w3 * s,   w2**2 * v + c,              w2 * w3 * v - w1 * s],
                   [w1 * w3 * v - w2 * s,   w2 * w3 * v + w1 * s,       w3**2 * v + c]])


def adjoint(transform):
    """

    :param Matrix transform:
    :return:
    """
    R = transform[:3, :3]
    p = transform[:3, 3]
    Adj = diag(R, R)
    Adj[:3, 3:] = sym_skew(p) * R
    return Adj


class SymTwist(object):

    def __init__(self, v, omega, theta, observation_frame=None, target_frame=None, reference_frame=None,
                 reference_point=None):
        self.v = Matrix(v)
        self.omega = Matrix(omega)

        if reference_frame is None:
            reference_frame = observation_frame

        if reference_point is None:
            reference_point = target_frame

        self.observation_frame = observation_frame
        self.target_frame = target_frame
        self.reference_frame = reference_frame
        self.reference_point = reference_point
        self.theta = Symbol(theta)

        self.xi = self._xi()
        self.matrix = self._matrix()
        self.exp = self._exp(self.theta)
        self.adjoint = self._adjoint(self.theta)

    @simplified
    def _xi(self):
        return self.v.col_join(self.omega)

    @simplified
    def _matrix(self):
        return sym_skew(self.omega).row_join(self.v).col_join(Matrix([0, 0, 0, 0]).T)

    @simplified
    def _exp(self, theta):
        R = sym_exp_so3(self.omega, theta)
        p = (eye(3) - R) * (self.omega.cross(self.v) + self.omega * self.omega.T * self.v * theta)
        return R.row_join(p).col_join(Matrix([0, 0, 0, 1]).T)

    @simplified
    def _adjoint(self, theta):
        return adjoint(self._exp(theta))

    def variables(self):
        return {self.theta}

    def constants(self):
        return self.omega.free_symbols + self.v.free_symbols


class DirectKinematicStateSpaceModel(StateSpaceModel):
    def __init__(self, kin_tree, endpoints, joint_names, state_order, dt):
        self.dt = dt
        self.pos_meas_model, self.point_indices = get_measurement_model(kin_tree, endpoints, joint_names)
        self.order = state_order
        self.joint_names = joint_names
        self.order_length = len(joint_names)

    def process_model(self, state_vector):
        new_vec = np.array(state_vector).astype(float)
        for i in range(self.order-1):
            start = i*self.order_length
            mid = start + self.order_length
            end = mid + self.order_length
            new_vec[start:mid] += self.dt * new_vec[mid:end]

        return new_vec

    def measurement_model(self, state_vector):
        return self.pos_meas_model(state_vector[:self.order_length])

    def vectorize_measurement(self, frame):
        return frame[self.point_indices, :].flatten()


def get_sym_jacobian(kin_tree, base_frame, manip_frame):
    """

    :param KinematicTree kin_tree:
    :param str base_frame:
    :param str manip_frame:
    :return:
    """
    incoming, outgoing = kin_tree.get_twist_chain(base_frame, manip_frame)
    sym_incoming = [SymTwist(xi[:3], xi[3:], theta) for theta, xi in incoming.items()]
    sym_outgoing = [SymTwist(xi[:3], xi[3:], theta) for theta, xi in outgoing.items()]

    base_pox_inv = simplify(reduce(lambda x, y: x*y, (t.exp for t in sym_incoming)), ratio=1.0) # syms
    manip_pox = simplify(reduce(lambda x, y: x*y, (t.exp for t in sym_outgoing)), ratio=1.0) # syms
    features = kin_tree.get_features()
    base_root_0 = Matrix(features[base_frame].inv().homog()) # numbers
    root_0_manip = Matrix(features[manip_frame].point()) # numbers
    base_root = base_root_0 * base_pox_inv
    base_manip_point = base_root * manip_pox * root_0_manip
    np_skew = sym_skew(-base_manip_point[:3, :])

    base_root_adj = adjoint(base_root)

    columns = [base_root_adj * twist.xi for twist in sym_incoming + sym_outgoing]

    jacobian = Matrix(reduce(MatrixBase.row_join, columns))
    jacobian[:3, :] += np_skew * jacobian[3:, :]
    return jacobian


def get_jacobian_func(kin_tree, base_frame, manip_frame, joint_names):
    jacobian = get_sym_jacobian(kin_tree, base_frame, manip_frame)
    joint_symbols = map(Symbol, joint_names)
    func = theano_function(joint_symbols, [jacobian])
    return func


def get_point_children(joint):
    points = []
    point_names = []
    for child in joint.children:
        if isinstance(child, Feature):
            if isinstance(child.primitive, Point):
                points.append(Matrix(child.primitive.homog()))
                point_names.append(child.name)

    point_indices = [int(p.split('_')[-1]) for p in point_names]

    return MatrixBase.hstack(*points), point_indices


def get_measurement_model(kin_tree, endpoints, joint_names):
    """

    :param KinematicTree kin_tree:
    :param list endpoints:
    :return:
    """
    kin_tree.set_zero_config()
    chains = map(kin_tree.get_chain, endpoints)
    joints = kin_tree.get_joints()
    root = kin_tree.get_root_joint()

    obs_funcs = []
    point_indices = []
    joint_symbols = map(Symbol, joint_names)

    for chain in chains:
        chain.remove(root.name)
        pox = eye(4)
        for joint_name in chain:
            print('Computing %s' % joint_name)
            xi = joints[joint_name].twist.twist().xi()
            exp = SymTwist(xi[:3], xi[3:], joint_name).exp
            pox = simplify(pox * exp, ratio=1.0)
            points, names = get_point_children(joints[joint_name])
            print('calc observed points')
            observed_points = simplify(pox*points, ratio=1.0)
            obs_funcs.append(theano_function(joint_symbols, [observed_points], on_unused_input='ignore'))
            point_indices.extend(names)
            print('Done.')

    def meas_func(angles):
        return np.concatenate([obs_func(*angles).T[:, :3].flatten() for obs_func in obs_funcs])

    return meas_func, point_indices


def get_state_space_model(kin_tree, endpoints, joint_names, state_order=2, dt=0.01875):

    return DirectKinematicStateSpaceModel(kin_tree, endpoints, joint_names, state_order, dt)


class KinematicModel(object):

    def __init__(self, kin_tree, joint_names, base_frame, manip_frame, state_order=2, dt=0.01875):
        self.joint_names = joint_names
        self.jacobian = get_jacobian_func(kin_tree, base_frame, manip_frame, joint_names)
        self.state_space_model = get_state_space_model(kin_tree, [base_frame, manip_frame], joint_names, state_order,
                                                       dt)

    def joint_angles(self, state_vector):
        return state_vector[:self.state_space_model.order_length]

    def filter(self, frame):
        return None


class DisturbanceModel(object):
    def __init__(self, observation_frame, observation_tree, body_frame, body_tree, reference_point,
                 reference_point_tree, reference_frame, reference_tree):
        self.reference_point_transform = None

    def get_observation_frame(self):
        return

    def get_body_frame(self):
        return

    def get_reference_point_frame(self):
        return

    def get_reference_frame(self):
        return

    def get_observation_body_transform(self, observation_frame, body_frame):
        return np.linalg.pinv(observation_frame).dot(body_frame)

    def get_raw_twist(self):
        return


class ManipulationModel(object):

    def __init__(self, input_model, passive_model, disturbance_model):
        """

        :param KinematicModel input_model:
        :param KinematicModel passive_model:
        :param KinematicModel disturbance_model:
        """
        self.input_model = input_model
        self.passive_model = passive_model
        self.state_names = input_model.joint_names + passive_model.joint_names
        self.split_point = self.input_model.state_space_model.order_length

    def split(self, state_vector):
        return state_vector[:self.split_point], state_vector[self.split_point:]

    def input_matrix(self, state_vector):
        input_joints, passive_joints = self.split(state_vector)
        input_jac = self.input_model.jacobian(input_joints)
        passive_jac_pinv = np.linalg.pinv(self.passive_model.jacobian(passive_joints))
        return np.vstack((np.ones((self.split_point, self.split_point)), passive_jac_pinv.dot(input_jac)))

    def disturbance_matrix(self, state_vector):
        _, passive_joints = self.split(state_vector)
        passive_jac_pinv = np.linalg.pinv(self.passive_model.jacobian(passive_joints))
        return np.vstack((np.zeros((self.split_point, self.split_point)), passive_jac_pinv))

    def dynamics(self, state_vector):
        input_joints, passive_joints = self.split(state_vector)
        input_jac = self.input_model.jacobian(input_joints)
        passive_jac_pinv = np.linalg.pinv(self.passive_model.jacobian(passive_joints))
        return np.vstack((np.ones((self.split_point, self.split_point)), passive_jac_pinv.dot(input_jac))), \
               np.vstack((np.zeros((self.split_point, self.split_point)), passive_jac_pinv))