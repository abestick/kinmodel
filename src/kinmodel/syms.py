from sympy import Symbol, Matrix, eye, simplify, sin, cos, diag, MatrixBase
from sympy.printing.theanocode import theano_function
from kinmodel import KinematicTree, Point, Feature, StateSpaceModel
import numpy as np
from track_mocap import MocapFrameTracker
from copy import deepcopy
from ukf import UnscentedKalmanFilter


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
        self._state_length = self.order_length * self.order
        self._meas_length = 3*len(self.point_indices)

    def process_model(self, state_vectors):
        new_vec = np.array(state_vectors).astype(float).T
        for k, state_vector in enumerate(new_vec):
            for i in range(self.order-1):
                start = i*self.order_length
                mid = start + self.order_length
                end = mid + self.order_length
                new_vec[k, start:mid] += self.dt * state_vector[mid:end]

            return new_vec.T

    def measurement_model(self, state_vectors):
        meas_list = []
        for state_vector in state_vectors.T:
            meas_list.append(self.pos_meas_model(state_vector[:self.order_length]))
        return np.array(meas_list).T


    def vectorize_measurement(self, frame):
        return frame[self.point_indices, :].flatten()


def get_sym_jacobian(kin_tree, base_frame, manip_frame, position_only=False):
    """

    :param KinematicTree kin_tree:
    :param str base_frame:
    :param str manip_frame:
    :return:
    """
    incoming, outgoing = kin_tree.get_twist_chains(base_frame, manip_frame)
    sym_incoming = [SymTwist(xi[:3], xi[3:], theta) for theta, xi in incoming.items()]
    sym_outgoing = [SymTwist(xi[:3], xi[3:], theta) for theta, xi in outgoing.items()]

    if len(sym_incoming) == 0:
        base_pox_inv = eye(4)
    else:
        base_pox_inv = simplify(reduce(lambda x, y: x*y, (t.exp for t in sym_incoming)), ratio=1.0) # syms

    if len(sym_outgoing) == 0:
        manip_pox = eye(4)
    else:
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

    if position_only:
        jacobian = jacobian[:3, :]

    return jacobian


def get_sym_body_jacobian(kin_tree, manip_frame, position_only=False):
    """

    :param KinematicTree kin_tree:
    :param str base_frame:
    :param str manip_frame:
    :return:
    """
    kin_tree = kin_tree.to_1d_chain()
    twists = kin_tree.get_twist_chain(manip_frame)
    sym_twists = [SymTwist(xi[:3], xi[3:], theta) for theta, xi in twists.items()]
    features = kin_tree.get_features()

    manip_pox = simplify(reduce(lambda x, y: x*y, (t.exp for t in sym_twists)), ratio=1.0) # syms
    root0_manip = Matrix(features[manip_frame].point()) # numbers
    root_manip = manip_pox * root0_manip

    columns = [manip_root_adj * twist.xi for twist in sym_twists]

    jacobian = Matrix(reduce(MatrixBase.row_join, columns))
    jacobian[:3, :] += np_skew * jacobian[3:, :]

    if position_only:
        jacobian = jacobian[:3, :]

    return jacobian


def get_jacobian_func(kin_tree, base_frame, manip_frame, joint_names, position_only=False):
    jacobian = get_sym_jacobian(kin_tree, base_frame, manip_frame, position_only)
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
            # print('Computing %s' % joint_name)
            xi = joints[joint_name].twist.twist().xi()
            exp = SymTwist(xi[:3], xi[3:], joint_name).exp
            pox = simplify(pox * exp, ratio=1.0)
            if not sum(isinstance(j, Feature) for j in joints[joint_name].children):
                # print('No observed points.')
                continue

            points, names = get_point_children(joints[joint_name])
            # print('Calculating observed points')
            observed_points = simplify(pox*points, ratio=1.0)
            obs_funcs.append(theano_function(joint_symbols, [observed_points], on_unused_input='ignore'))
            point_indices.extend(names)
            # print('Done.')

    def meas_func(angles):
        return np.concatenate([obs_func(*angles).T[:, :3].flatten() for obs_func in obs_funcs])

    return meas_func, point_indices


def get_state_space_model(kin_tree, endpoints, joint_names, state_order=2, dt=0.01875):

    return DirectKinematicStateSpaceModel(kin_tree, endpoints, joint_names, state_order, dt)


class KinematicModel(object):

    def __init__(self, kin_tree, joint_names, base_frame, manip_frame, state_order=1, dt=0.01875, position_only=True):
        self.joint_names = joint_names
        self.state_space_model = get_state_space_model(kin_tree, [base_frame, manip_frame], joint_names, state_order,
                                                       dt)
        self.jacobian = get_jacobian_func(kin_tree, base_frame, manip_frame, joint_names, position_only)
        p0 = 0.25
        q = np.pi / 2 / 80
        idx = self.state_space_model.order_length * (self.state_space_model.order-1)
        P0 = np.identity(self.state_space_model.state_length()) * p0
        P0[:idx, :idx] = 0
        Q = np.identity(self.state_space_model.state_length()) * q
        Q[:idx, :idx] = 0
        x0 = np.zeros(self.state_space_model.state_length())
        self.ukf = UnscentedKalmanFilter(self.state_space_model.process_model, self.state_space_model.measurement_model,
                                         x0=x0, P0=P0, Q=Q, R=1e-2)
        self.last_angles = self.joint_angles(x0)

    def joint_angles(self, state_vector):
        return state_vector.squeeze()[:self.state_space_model.order_length]

    def joint_velocities(self, state_vector):
        if self.state_space_model.order > 1:
            return state_vector.squeeze()[self.state_space_model.order_length:2*self.state_space_model.order_length]

        else:
            return (state_vector -self.last_angles) / self.state_space_model.dt

    def filter(self, points):
        measurement = self.state_space_model.vectorize_measurement(points)
        return self.ukf.filter(measurement)[0].squeeze()


class CartesianTracker(object):

    def __init__(self, dt, init_value=None, init_twist_value=None):
        self.relative_transform_frames = set()
        self.transforms = {'world': {}}
        self.twists = {}
        self.trackers = {}
        self.dt = dt
        self.init_value = init_value
        self.init_twist_value = init_twist_value

    def has_transform(self, reference_frame, target_frame):
        if reference_frame == target_frame:
            return 1
        if reference_frame in self.transforms:
            if target_frame in self.transforms[reference_frame]:
                return 1

            else:
                return 0

        elif target_frame in self.transforms:
            if reference_frame in self.transforms[target_frame]:
                return -1

            else:
                return 0

    def get_transform(self, reference_frame, target_frame):
        if reference_frame in self.transforms:
            if target_frame in self.transforms[reference_frame]:
                return self.transforms[reference_frame][target_frame]

            else:
                raise ValueError('Transform %s->%s not tracked.' % (reference_frame, target_frame))

        elif target_frame in self.transforms:
            if reference_frame in self.transforms[target_frame]:
                return self.transforms[reference_frame][target_frame].inv()

            else:
                raise ValueError('Transform %s->%s not tracked.' % (reference_frame, target_frame))

    def get_twist(self, observation_frame, target_frame, reference_frame=None, reference_point=None):
        return self.twists[observation_frame][target_frame][reference_frame][reference_point]

    def track_frame(self, name, indices, points):
        self.trackers[name] = MocapFrameTracker(name, indices, points)
        self.transforms['world'][name] = deepcopy(self.init_value)

    def track_relative_frame(self, reference_frame, target_frame):
        assert reference_frame in self.trackers, 'Frame %s not tracked.' % reference_frame
        assert target_frame in self.trackers, 'Frame %s not tracked.' % target_frame

        if reference_frame not in self.transforms:
            self.transforms[reference_frame] = {}

        self.transforms[reference_frame][target_frame] = deepcopy(self.init_value)
        self.relative_transform_frames.add(reference_frame)

    def track_twist(self, observation_frame, target_frame, reference_frame=None, reference_point=None):

        assert self.has_transform(observation_frame, target_frame)

        if reference_frame is None:
            reference_frame = observation_frame
        else:
            assert self.has_transform(reference_frame, observation_frame)

        if reference_point is None:
            reference_point = target_frame
        else:
            assert self.has_transform(reference_point, target_frame)
            assert self.has_transform(reference_frame, reference_point)

        if observation_frame not in self.twists:
            self.twists[observation_frame] = {}

        if target_frame not in self.twists[observation_frame]:
            self.twists[observation_frame][target_frame] = {}

        if reference_frame not in self.twists[observation_frame][target_frame]:
            self.twists[observation_frame][target_frame][reference_frame] = {}

        self.twists[observation_frame][target_frame][reference_frame][reference_point] = deepcopy(self.init_twist_value)

    def step(self, points):
        last_transforms = deepcopy(self.transforms)

        for frame_name in self.trackers:
            self.transforms['world'][frame_name] = self.trackers[frame_name].process_frame(points)

        for reference_frame in self.relative_transform_frames:
            ref_world = self.transforms['world'][reference_frame].inv()
            for target_frame in self.transforms[reference_frame]:
                world_target = self.transforms['world'][target_frame]
                self.transforms[reference_frame][target_frame] = ref_world * world_target

        for observation_frame in self.twists:
            for target_frame in self.twists[observation_frame]:
                raw_twist = (self.transforms[observation_frame][target_frame] -
                             last_transforms[observation_frame][target_frame]) / self.dt

                for reference_frame in self.twists[observation_frame][target_frame]:
                    if reference_frame != observation_frame:
                        ref_twist = self.get_transform(observation_frame, target_frame).rot() * raw_twist
                    else:
                        ref_twist = deepcopy(raw_twist)

                    for reference_point in self.twists[observation_frame][target_frame][reference_frame]:
                        if reference_point != target_frame:
                            point_target_vec = self.get_transform(reference_point, target_frame).trans()
                            if reference_point != reference_frame:
                                point_target_vec = self.get_transform(reference_frame, reference_point) * \
                                                   point_target_vec
                            twist = point_target_vec.apply_translation(ref_twist)

                        else:
                            twist = deepcopy(ref_twist)

                        self.twists[observation_frame][target_frame][reference_frame][reference_point] = twist


class ManipulationModel(object):

    def __init__(self, robot_frame_name, human_frame_name, grip_frame_name, cartesian_tracker, active_kinematic_model,
                 passive_kinematic_model, position_only=True):
        """
        :param str robot_frame_name:
        :param str human_frame_name:
        :param str grip_frame_name:
        :param CartesianTracker cartesian_tracker:
        :param KinematicModel active_kinematic_model:
        :param KinematicModel passive_kinematic_model:
        """

        self.cartesian_tracker = cartesian_tracker
        self.active_kinematic_model = active_kinematic_model
        self.passive_kinematic_model = passive_kinematic_model
        self.split_point = self.active_kinematic_model.state_space_model.order_length

        self.cartesian_tracker.track_relative_frame(robot_frame_name, human_frame_name)
        self.cartesian_tracker.track_relative_frame(robot_frame_name, grip_frame_name)
        self.cartesian_tracker.track_relative_frame(grip_frame_name, human_frame_name)
        self.disturbance_frames = (robot_frame_name, human_frame_name, robot_frame_name, grip_frame_name)
        self.rotation_frames = (robot_frame_name, human_frame_name)
        self.cartesian_tracker.track_twist(*self.disturbance_frames)
        self.position_only = position_only
        self.disturbance_length = 3 if position_only else 6

    def split(self, state_vector):
        return state_vector[:self.split_point], state_vector[self.split_point:]

    def get_disturbance(self):
        twist = self.cartesian_tracker.get_twist(*self.disturbance_frames).xi().squeeze()
        return twist[:3] if self.position_only else twist

    def input_matrix(self, state_vector):
        input_joints, passive_joints = self.split(state_vector)
        input_jac = self.active_kinematic_model.jacobian(*input_joints)
        passive_jac_pinv = np.linalg.pinv(self.passive_kinematic_model.jacobian(*passive_joints))
        rotation = self.cartesian_tracker.get_transform(*self.rotation_frames).R()
        return np.vstack((np.eye(self.split_point), passive_jac_pinv.dot(rotation.dot(input_jac))))

    def disturbance_matrix(self, state_vector):
        _,  = self.split(state_vector)
        passive_jac_pinv = np.linalg.pinv(self.passive_kinematic_model.jacobian(*passive_joints))
        return np.vstack((np.zeros((self.split_point, self.split_point)), passive_jac_pinv))

    def dynamics(self, input_joints, passive_joints):
        input_jac = self.active_kinematic_model.jacobian(*input_joints)
        passive_jac_pinv = np.linalg.pinv(self.passive_kinematic_model.jacobian(*passive_joints))
        rotation = self.cartesian_tracker.get_transform(*self.rotation_frames).R()
        return np.vstack((np.eye(self.split_point), passive_jac_pinv.dot(rotation.dot(input_jac)))), \
            np.vstack((np.zeros((self.split_point, self.disturbance_length)), passive_jac_pinv))

    def init_filters(self, points, reps=50):
        for i in range(reps):
            self.active_kinematic_model.filter(points)
            self.passive_kinematic_model.filter(points)
        self.active_kinematic_model.last_angles = self.active_kinematic_model.joint_angles(
            self.active_kinematic_model.filter(points))
        self.passive_kinematic_model.last_angles = self.passive_kinematic_model.joint_angles(
            self.passive_kinematic_model.filter(points))

    def step(self, points):
        self.cartesian_tracker.step(points)
        input_states = self.active_kinematic_model.filter(points)
        passive_states = self.passive_kinematic_model.filter(points)
        input_joints = self.active_kinematic_model.joint_angles(input_states)
        passive_joints = self.passive_kinematic_model.joint_angles(passive_states)
        input_velocities = self.active_kinematic_model.joint_velocities(input_states)
        passive_velocities = self.passive_kinematic_model.joint_velocities(passive_states)

        return (input_joints, passive_joints, input_velocities, passive_velocities,
                self.get_disturbance()) + self.dynamics(input_joints, passive_joints)

    def error(self, input_velocities, passive_velocities, disturbance, input_matrix, disturbance_matrix):
        predicted_velocities = input_matrix.dot(input_velocities) + disturbance_matrix.dot(disturbance)
        predicted_velocities = predicted_velocities[self.split_point:]
        return np.linalg.norm(predicted_velocities-passive_velocities), predicted_velocities, passive_velocities

    def state_vector(self, input_joints, passive_joints):
        return np.concatenate((input_joints, passive_joints))

    def state_velocities(self, input_velocities, passive_velocities):
        return np.concatenate((input_velocities, passive_velocities))

    def state_names(self):
        return self.active_kinematic_model.joint_names + self.passive_kinematic_model.joint_names


class CostModel(object):

    def __init__(self, names, references, indices):
        self.names = names
        self.references = references
        self.indices = indices
        self.total_subset = sorted(set().union(*indices))

        self.ref_mat = np.zeros((len(self.names), max(self.total_subset)+1)) + np.nan
        for i, (reference, indices) in enumerate(zip(self.references, self.indices)):
            self.ref_mat[i, indices] = reference

        self.ref_mat = self.ref_mat[:, self.total_subset]

    def gradients(self, state_vector, padded=True):
        grad = self.ref_mat - state_vector[self.total_subset]
        return np.nan_to_num(grad) if padded else grad

    def costs(self, state_vector):
        return np.linalg.norm(self.gradients(state_vector), axis=1) ** 2


class CostLearningModel(object):

    def __init__(self, manipulation_model, cost_model):
        """

        :param ManipulationModel manipulation_model:
        :param CostModel cost_model:
        """

        self.manipulation_model = manipulation_model
        self.cost_model = cost_model

    def step(self, points):
        input_joints, passive_joints, input_velocities, passive_velocities, disturbance, input_matrix, \
        disturbance_matrix = self.manipulation_model.step(points)
        state_vector = self.manipulation_model.state_vector(input_joints, passive_joints)
        state_velocities = self.manipulation_model.state_velocities(input_velocities, passive_velocities)
        state_names = self.manipulation_model.state_names()
        cost_gradients = self.cost_model.gradients(state_vector)
        error = self.manipulation_model.error(input_velocities, passive_velocities, disturbance, input_matrix,
                                              disturbance_matrix)
        return state_vector, state_velocities, state_names, disturbance, input_matrix, disturbance_matrix, \
               cost_gradients, error

    def init(self, points):
        self.manipulation_model.init_filters(points)
        return self.step(points)


def left_pinv(M):
    return M.H * (M * M.H) ** -1