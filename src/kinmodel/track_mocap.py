#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import threading
import rospy
import tf
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Header
import ukf
import kinmodel
import numpy.linalg as npla
from phasespace.mocap_definitions import MocapWrist


class MocapTracker(object):

    def __init__(self, mocap_source, state_space_model, base_markers, base_frame_points, marker_indices,
                 joint_states_topic=None, object_tf_frame=None, new_frame_callback=None):
        self.mocap_source = mocap_source
        self.state_space_model = state_space_model
        self.base_markers = base_markers
        self.base_frame_points = base_frame_points
        self._marker_indices = marker_indices
        self._marker_names = {index: name for name, index in self._marker_indices.items()}
        self.base_indices = self.get_marker_indices(self.base_markers)
        self._estimation_pub = None
        self._tf_pub = None
        self._callback = new_frame_callback
        self.exit = False

        if joint_states_topic is not None:
            self._estimation_pub = rospy.Publisher(joint_states_topic, sensor_msgs.JointState,
                                                   queue_size=10)
        if object_tf_frame is not None:
            self._tf_pub = tf.TransformBroadcaster()

        self.uk_filter = ukf.UnscentedKalmanFilter(self.state_space_model.process_model,
                                                   self.state_space_model.measurement_model,
                                                   x0=np.zeros(self.state_space_model.state_length),
                                                   P0=np.identity(self.state_space_model.state_length) * 0.25,
                                                   Q=np.pi / 2 / 80, R=5e-3)

        self.mocap_source.set_coordinates(self.base_indices, self.base_frame_points, mode='time_varying')

        self._estimation = None
        self._observation = None
        self._covariance = None
        self._squared_residual = None

    def _initialize_filter(self, initial_observation, reps=50):
        for i in range(reps):
            self.uk_filter.filter(initial_observation)

    def _update_outputs(self, i):
        if self._callback is not None:
            self._callback(i, self._estimation, covariance=self._covariance,
                           squared_residual=self._squared_residual)

        if self._estimation_pub is not None:
            msg = sensor_msgs.JointState(position=self._estimation.squeeze(),
                                         header=Header(stamp=rospy.Time.now()))
            self._estimation_pub.publish(msg)

        # Publish the base frame pose of the flexible object
        if self._tf_pub is not None:
            homog = npla.inv(self.mocap_source.get_last_coordinates())
            mocap_frame_name = self.mocap_source.get_frame_name()
            if mocap_frame_name is not None:
                self._tf_pub.sendTransform(homog[0:3, 3],
                                           tf.transformations.quaternion_from_matrix(homog),
                                           rospy.Time.now(), '/object_base', '/' + mocap_frame_name)

    def _extract_observation(self, frame):
        return {name: kinmodel.new_geometric_primitive((frame[self._marker_indices[name], :, 0]))
                for name in self._marker_indices}

    def get_marker_indices(self, marker_names):
        return [self._marker_indices[name] for name in marker_names]

    def get_marker_names(self, marker_indices):
        return [self._marker_names[index] for index in marker_indices]

    def start(self):
        self.exit = False
        reader_thread = threading.Thread(target=self.run)
        reader_thread.start()
        return reader_thread

    def stop(self):
        self.exit = True

    def run(self):
        for i, (frame, timestamp) in enumerate(self.mocap_source):

            if self.exit:
                break

            self._observation = self._extract_observation(frame)

            observation_array = self.state_space_model.vectorize_measurement(self._observation)

            if i == 0:
                self._initialize_filter(observation_array)
                continue

            self._estimation, self._covariance, self._squared_residual = self.uk_filter.filter(observation_array)

            self._update_outputs(i)


class KinematicTreeTracker(MocapTracker):
    def __init__(self, kin_tree, mocap_source, joint_states_topic=None, object_tf_frame=None,
                 new_frame_callback=None, return_array=False):
        self.kin_tree = kin_tree
        self._return_array = return_array

        # Get the base marker indices
        base_markers = []
        base_joint = self.kin_tree.get_root_joint()
        for child in base_joint.children:
            if not hasattr(child, 'children'):
                # This is a feature
                base_markers.append(child.name)

        # Get all the marker indices of interest and map to their names
        marker_indices = {}
        for feature_name in self.kin_tree.get_features():
            marker_indices[feature_name] = int(feature_name.split('_')[1])

        # Set the base coordinate transform for the mocap stream
        base_frame_points = np.zeros((len(base_markers), 3, 1))
        all_features = self.kin_tree.get_features()
        for i, idx in enumerate(base_markers):
            base_frame_points[i, :, 0] = all_features['mocap_' + str(idx)].q()

        state_space_model = kinmodel.KinematicTreeStateSpaceModel(self.kin_tree)

        super(KinematicTreeTracker, self).__init__(mocap_source, state_space_model, base_markers, base_frame_points,
                                                   marker_indices, joint_states_topic, object_tf_frame,
                                                   new_frame_callback)

    def start(self):
        self.exit = False
        reader_thread = threading.Thread(target=self.run)
        reader_thread.start()
        return reader_thread

    def stop(self):
        self.exit = True


class KinematicTreeExternalFrameTracker(object):
    def __init__(self, kin_tree, base_tf_frame_name):
        self._kin_tree = kin_tree
        self._base_frame_name = base_tf_frame_name
        self._tf_pub_frame_names = []  # Frame names to publish on tf after an _update() call
        self._attached_frame_names = []  # All attached static frames
        self._attached_tf_frame_names = []  # All attached tf frames - update during an _update() call
        self._tf_pub = tf.TransformBroadcaster()
        self._tf_listener = tf.TransformListener()

    def attach_frame(self, joint_name, frame_name, tf_pub=True, pose=None):
        # Attach a static frame to the tree
        joints = self._kin_tree.get_joints()
        if pose is None:
            # No pose specified, set to mean position of all other Point children of this joint
            trans = np.zeros((3,))
            num_points = 0
            for point in joints[joint_name].children:
                try:
                    trans = trans + point.primitive.q().squeeze()
                    num_points += 1
                except AttributeError:
                    pass  # This feature isn't a Point
            trans = trans / num_points
            homog = np.identity(4)
            homog[0:3, 3] = trans
        else:
            # Attach the frame at the specified pose
            homog = pose.squeeze()
        new_feature = kinmodel.Feature(frame_name, kinmodel.Transform(homog_array=homog))
        joints[joint_name].children.append(new_feature)
        self._attached_frame_names.append(frame_name)
        if tf_pub:
            self._tf_pub_frame_names.append(frame_name)

    def attach_tf_frame(self, joint_name, tf_frame_name):
        # Attach reference frame to specified joint
        self.attach_frame(joint_name, '_tf_ref_' + tf_frame_name, tf_pub=True)

        # Attach tf frame
        joints = self._kin_tree.get_joints()
        new_feature = kinmodel.Feature(tf_frame_name, kinmodel.Transform())
        joints[joint_name].children.append(new_feature)
        self._attached_tf_frame_names.append(tf_frame_name)

    def set_config(self, joint_angles_dict):
        self._kin_tree.set_config(joint_angles_dict)

    def observe_frames(self):
        self._update()
        observations = self._kin_tree.observe_features()
        external_frame_dict = {}
        for frame_name in self._attached_frame_names:
            external_frame_dict[frame_name] = observations[frame_name]
        for frame_name in self._attached_tf_frame_names:
            external_frame_dict[frame_name] = observations[frame_name]
        return external_frame_dict

    def compute_jacobian(self, base_frame_name, manip_name_frame):
        self._update()
        return self._kin_tree.compute_jacobian(base_frame_name, manip_name_frame)

    def _update(self):
        # Get the current and zero-config feature observations
        feature_obs = self._kin_tree.observe_features()
        all_features = self._kin_tree.get_features()

        # Update tf frame poses
        updated_features = {}
        for frame_name in self._attached_tf_frame_names:
            try:
                trans, rot = self._tf_listener.lookupTransform(self._base_frame_name, frame_name, rospy.Time(0))
                tf_transform = tf.transformations.quaternion_matrix(rot)
                tf_transform[0:3, 3] = trans
                base_robot_trans = kinmodel.Transform(homog_array=tf_transform)
                base_reference_trans = feature_obs['_tf_ref_' + frame_name]
                base_reference_zero_config = all_features['_tf_ref_' + frame_name]
                updated_features[frame_name] = base_reference_zero_config * (
                base_reference_trans.inv() * base_robot_trans)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print('Lookup failed for TF frame: ' + frame_name)
        self._kin_tree.set_features(updated_features)

        # Observe and publish specified static features
        for frame_name in self._tf_pub_frame_names:
            obs = feature_obs[frame_name].homog()
            self._tf_pub.sendTransform(obs[0:3, 3],
                                       tf.transformations.quaternion_from_matrix(obs),
                                       rospy.Time.now(), frame_name, self._base_frame_name)


class WristTracker(MocapTracker, MocapWrist):

    def __init__(self, mocap_source, marker_indices, reference_frame, joint_states_topic=None, object_tf_frame=None,
                 new_frame_callback=None):
        assert all(name in marker_indices for name in self.names), \
            "marker_names must contain all these keys %s" % self.names

        base_markers = self.get_marker_group('hand')
        arm_markers = self.get_marker_group('arm')

        _, base_frame_points = determine_hand_coordinate_transform(extract_marker_subset(reference_frame, base_markers,
                                                                                         marker_indices),
                                                                   extract_marker_subset(reference_frame, arm_markers,
                                                                                         marker_indices))

        state_space_model = WristStateSpaceModel()

        super(WristTracker, self).__init__(mocap_source, state_space_model, base_markers, base_frame_points, marker_indices,
                                           joint_states_topic, object_tf_frame, new_frame_callback)


def unit_vector(array):
    return array / npla.norm(array)


def extract_marker_subset(frame_data, names, marker_indices):
    indices = [marker_indices[name] for name in names]
    return frame_data[indices, :]


def determine_hand_coordinate_transform(hand_points, arm_points):
    origin, normal = best_fitting_plane(hand_points)
    z_axis = unit_vector(normal)
    arm_vec = np.mean(arm_points, axis=0) - origin
    arm_z = arm_vec.dot(z_axis) * z_axis
    x_axis = unit_vector(arm_vec - arm_z)

    assert x_axis.dot(z_axis) == 0, "Axes are not orthogonal!"

    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.hstack((x_axis, y_axis, z_axis))
    origin_in_hand_frame = rotation_matrix.dot(origin)
    homog_transform = np.vstack((np.hstack((rotation_matrix, origin_in_hand_frame)), np.append(np.zeros(3), 1)))

    desired_hand_points = np.empty(hand_points.shape[0], 4)
    for i, marker in enumerate(hand_points):
        desired_hand_points[i, :] = homog_transform.dot(np.append(hand_points, 1))

    return homog_transform, desired_hand_points[:, :3]


def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data
    
    Parameters
    ----------        
    data: numpy.ndarray
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])
    
    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix. 
    
    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.   
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.
    
    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.
    
    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.
    
    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points
    
    Parameters
    ----------        
    points: numpy.ndarray
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])
    
    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.
    
    or
    
    mean, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])
    
    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a mean from the plane
    mean = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, mean))
        return a, b, c, d

    else:
        return mean, normal