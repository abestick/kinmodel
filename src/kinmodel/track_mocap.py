#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from numpy.linalg import LinAlgError
import threading
import rospy
import tf
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Header
import ukf
import kinmodel
from .tools import unit_vector
import numpy.linalg as npla
from phasespace.mocap_definitions import MocapWrist
from phasespace.load_mocap import transform_frame, find_homog_trans
from copy import deepcopy


class MocapTracker(object):
    """
    A general class which pipes mocap points from a mocap source to a UKF
    """

    def __init__(self, name, mocap_source, state_space_model, base_markers, base_frame_points, marker_indices,
                 joint_states_topic=None, object_tf_frame=None, new_frame_callback=None, is_master=False):
        """
        Constructor
        :param name: This name helps when defining coordinate frames in the mocap source so that each tracker is unique
        :param mocap_source: the MocapSource object to draw point data from
        :param state_space_model: the model which the UKF will be based on
        :param base_markers: list of strings, the markers which are static relative to one another in the base frame
        I chose to pass the names of the markers instead of the indices so tracker definitions are more readable
        :param base_frame_points: the values of the base_markers when transformed into the base frame
        :param dict marker_indices: a dictionary mapping marker names to marker indices in the MocapSource
        :param joint_states_topic: an optional topic upon which to publish tracker output
        :param object_tf_frame: Not 100% sure how this ties in now, relic of KinematicTreeTracker but maybe it should be
        only in the KinematicTreeTrack child??
        :param new_frame_callback: optional callback function called when UKF output is obtained
        """
        # copy across data
        self.name = name
        self.mocap_source = mocap_source
        self.state_space_model = state_space_model
        self.base_markers = base_markers
        self.base_frame_points = base_frame_points
        self._marker_indices = marker_indices
        # reverse the dict so we can access names for  markers too
        self._marker_names = {index: name for name, index in self._marker_indices.items()}

        # get the indices for the bases
        self.base_indices = self.get_marker_indices(self.base_markers)

        self._estimation_pub = None
        self._tf_pub = None
        self._callbacks = [new_frame_callback] if new_frame_callback is not None else []
        self.exit = False

        if joint_states_topic is not None:
            self._estimation_pub = rospy.Publisher(joint_states_topic, sensor_msgs.JointState,
                                                   queue_size=10)
        if object_tf_frame is not None:
            self._tf_pub = tf.TransformBroadcaster()

        # set up the filter
        self.uk_filter = ukf.UnscentedKalmanFilter(self.state_space_model.process_model,
                                                   self.state_space_model.measurement_model,
                                                   x0=np.zeros(self.state_space_model.state_length()),
                                                   P0=np.identity(self.state_space_model.state_length()) * 0.25,
                                                   Q=np.pi / 2 / 80, R=5e-3)

        # setup a coordinate frame for the system in the source (maybe we need to check for uniqueness)
        self.mocap_source.set_coordinates(self.name, self.base_indices, self.base_frame_points, mode='time_varying')

        self._estimation = None
        self._observation = None
        self._covariance = None
        self._squared_residual = None
        self._is_master = is_master
        self._slaves = {}
        self._state_names = self._unvectorize_estimation().keys()

    def _unvectorize_estimation(self, state_vector=None):
        if state_vector is None:
            state_vector = np.zeros(self.state_space_model.state_length())
        return self.state_space_model.unvectorize_estimation(state_vector, self.name+'_')

    def get_state_names(self):
        return self._state_names

    def is_master(self):
        return self._is_master

    def set_master(self, is_master):
        # If we are changing something
        if is_master != self._is_master:
            # update whether we are a master or not
            self._is_master = is_master

            # in both cases this list should be empty (slaves shouldn't have slaves and new masters don't yet either)
            self._slaves = {}

    def add_slaves(self, slaves):
        """
        Adds a list of other trackers as slaves who's step functions will be called whenever this one is.
        This function checks that the other trackers are slaves and thus is safer than enslave
        :param slaves: list of trackers with is_master() == False
        :return: 
        """
        for slave in slaves:
            assert not slave.is_master(), "You cannot add another master as a slave!"
            self._slaves[slave.name] = slave

    def enslave(self, others):
        """
        This function takes a list of other trackers and forces them into being slaves, thus perhaps losing links.
        :param others: list of trackers
        :return: 
        """
        for other in others:
            other.set_master(False)
            self._slaves[other.name] = other

    def free(self, name):
        self._slaves.pop(name)

    def free_all(self):
        self._slaves = {}

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def get_marker_indices(self, marker_names):
        """Returns the marker indices for a set of marker names"""
        return [self._marker_indices[name] for name in marker_names]

    def get_marker_names(self, marker_indices):
        """Returns the marker names for a set of marker indices"""
        return [self._marker_names[index] for index in marker_indices]

    def start(self):
        """Runs the thread to begin the tracker"""
        self.exit = False
        reader_thread = threading.Thread(target=self.run)
        reader_thread.start()
        return reader_thread

    def stop(self):
        """Stops the thread"""
        self.exit = True

    def run(self, record=False):
        """
        Iterates over the mocap source, applies all processing and filtering to the data and calls the relevant updates 
        """

        estimations = []
        covariances = []
        squared_residuals = []

        for i, (frame, timestamp) in enumerate(self.mocap_source.iterate(coordinate_frame=self.name)):

            if self.exit:
                break

            self.step_frame(frame, i)

            if record:
                estimations.append(self._estimation)
                covariances.append(self._covariance)
                squared_residuals.append(self._squared_residual)

        return estimations, covariances, squared_residuals

    def step(self, i=-1):
        frame, timestamp = self.mocap_source.get_latest_frame()
        self.step_frame(frame)
        return self._estimation

    def step_frame(self, frame, i=-1):
        print("%s: %d" % (self.name, i))
        for slave in self._slaves.items():
            slave.step(i)

        # if its our first frame, converge the error in the filter
        if i == 0:
            state_vector, self._covariance, self._squared_residual = self._initialize_filter(frame)

        else:
            # otherwise store filter output
            state_vector, self._covariance, self._squared_residual = self._filter(frame)

        self._estimation = self._unvectorize_estimation(state_vector)

        # and call the callbacks, publisher's etc
        self._update_outputs(i)

    def _process_frame(self, frame):
        # convert frame into observation dict and store
        self._observation = self._extract_observation(frame)

        # convert observation dict into state space measurement
        measurement = self._preprocess_measurement(self._observation)

        # vectorize the measurement
        return self.state_space_model.vectorize_measurement(measurement)

    def _extract_observation(self, frame):
        """
        Converts a mocap frame into a dict of marker names and geometric Points
        :param frame: mocap frame (n,3)
        :return: dict of geometric Points
        """
        return {name: kinmodel.new_geometric_primitive((frame[self._marker_indices[name], :]))
                for name in self._marker_indices}

    def _preprocess_measurement(self, observation):
        """
        Converts a marker observation into a whatever format the state space's measurement is defined in, this is an
        identity function if not overloaded
        :param observation: dict of marker points
        :return: any format of measurement
        """
        return observation

    def _update_outputs(self, i):
        """
        Calls callbacks, oublishes to topics, and to tf 
        :param i: the frame count
        :return: 
        """
        for callback in self._callbacks:
            callback(i, self._estimation, covariance=self._covariance,
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

    def _initialize_filter(self, initial_frame, reps=50):
        """
        Runs the filter to converge initial error
        :param initial_frame: initial frame
        :param reps: how many cycles to run
        :return: 
        """
        initial_measurement = self._process_frame(initial_frame)

        for i in range(reps):
            self.uk_filter.filter(initial_measurement)
        return self.uk_filter.filter(initial_measurement)

    def _filter(self, frame):
        measurement_array = self._process_frame(frame)
        return self.uk_filter.filter(measurement_array)

    def clone(self):
        return deepcopy(self)


class KinematicTreeTracker(MocapTracker):
    def __init__(self, name, kin_tree, mocap_source, joint_states_topic=None, object_tf_frame=None,
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
        for i, marker in enumerate(base_markers):
            base_frame_points[i, :, 0] = all_features[marker].q()

        state_space_model = kinmodel.KinematicTreeStateSpaceModel(self.kin_tree)

        super(KinematicTreeTracker, self).__init__(name, mocap_source, state_space_model, base_markers, base_frame_points,
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
    """
    Child of MocapTracker and MocapWrist specific to wrist tracking. Inheriting MocapWrist means this class inherits all
    the pre-defined marker names and marker groups specific to the wrist which are define in 
    phasespace.mocap_definitions.MocapWrist
    """
    def __init__(self, name, mocap_source, marker_indices, reference_frame, joint_states_topic=None, object_tf_frame=None,
                 new_frame_callback=None):
        # make sure the marker_indices dict contains all the necessary names
        assert all(name in marker_indices for name in self.names), \
            "marker_indices must contain all these keys %s" % self.names

        # get the relevant subsets
        base_markers = self.get_marker_group('hand')
        arm_markers = self.get_marker_group('arm')

        # determine the frame with the origin at the mean of the hand points, the z axis normal to the plane they form
        # and the x axis the vector to the arm at zero conditions projected onto this plane
        transform, base_frame_points = determine_hand_coordinate_transform(
            extract_marker_subset(reference_frame, base_markers, marker_indices),
            extract_marker_subset(reference_frame, arm_markers, marker_indices))

        # pass the acquired information to the super call
        super(WristTracker, self).__init__(name, mocap_source, kinmodel.WristStateSpaceModel(), base_markers, base_frame_points,
                                           marker_indices, joint_states_topic, object_tf_frame, new_frame_callback)

        # transform the points in the reference frame into the base frame
        reference_in_hand_coords = transform_frame(reference_frame, transform)

        # turn this into a dict of points
        reference_dict = self._extract_observation(reference_in_hand_coords)

        # set the zero conditions of the arm as the arm points of the reference frame in the hand coordinate system
        self._arm_zero = {marker: reference_dict[marker] for marker in arm_markers}

    def _preprocess_measurement(self, observation):
        """
        Overloads parent method, works out the transform between the current arm points and the zero condition arm
        points and returns the rotation component of this transform in euler angles
        :param observation: dict of marker points
        :return: dict of roll pitch yaw
        """
        current = []
        desired = []

        # get the arm markers that are visible
        for marker in self._arm_zero:
            if not np.isnan(observation[marker].q()).any():
                current.append(observation[marker])
                desired.append(self._arm_zero[marker])

        # if we don't have enough, return nans
        if len(current) < 3:
            euler_angles = np.full(3, np.nan)

        # otherwise workout the transform and convert to euler
        else:
            current_array = kinmodel.stack(*current, homog=False)
            desired_array = kinmodel.stack(*desired, homog=False)

            homog, _ = find_homog_trans(current_array, desired_array)

            euler_angles = tf.transformations.euler_from_matrix(homog[:3, :3])

        # except LinAlgError:
        #     euler_angles = np.full(3, np.nan)

        return {config: euler_angles[i] for i, config in enumerate(self.configs)}


def extract_marker_subset(frame_data, names, marker_indices):
    """
    Takes a frame of mocap points, marker names, and a dict that maps to indices and returns the subset of those names
    Should this be a method of MocapTracker??
    """
    indices = [marker_indices[name] for name in names]
    return frame_data[indices, :]


def determine_hand_coordinate_transform(hand_points, arm_points, zero_thresh=1e-15):
    """
    Defines the hand frame as the origin at the mean of the hand points, the z axis normal to the plane they form and 
    the x axis the vector to the arm at zero conditions projected onto this plane. It then transforms the hand points
    into this frame and returns the transform and the transformed hand points
    :param hand_points: 
    :param arm_points: 
    :param zero_thresh: 
    :return: 
    """
    origin, normal = best_fitting_plane(hand_points)
    z_axis = unit_vector(normal)
    arm_vec = np.mean(arm_points, axis=0) - origin
    arm_z = arm_vec.dot(z_axis) * z_axis
    x_axis = unit_vector(arm_vec - arm_z)

    assert abs(x_axis.dot(z_axis)) < zero_thresh, "Axes are not orthogonal!"

    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.stack((x_axis, y_axis, z_axis)).T
    origin_in_hand_frame = rotation_matrix.dot(origin).reshape((-1, 1))
    homog_transform = np.vstack((np.hstack((rotation_matrix, origin_in_hand_frame)), np.append(np.zeros(3), 1)))

    desired_hand_points = np.empty((hand_points.shape[0], 4))
    for i, marker in enumerate(hand_points):
        desired_hand_points[i, :] = homog_transform.dot(np.append(marker, 1))

    return homog_transform, desired_hand_points[:, :3]


"""
These are taken from stack overflow and work pretty well
"""
def pca(data, correlation = False, sort = True):
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

    w, v = pca(points)

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