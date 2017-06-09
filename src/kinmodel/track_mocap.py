#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from abc import ABCMeta, abstractmethod
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
from phasespace.load_mocap import transform_frame, find_homog_trans, MocapStream
from copy import deepcopy

## TODO: Put the following coordinate transform code wherever in the system the MocapStream is 
# iterated over
#-----------------------------------------------------
# Method to get the coordinates and indices of the object base frame markers from a kintree
def get_kin_tree_base_markers(kin_tree):
    #Get base marker names
    base_markers = []
    base_joint = self.kin_tree.get_root_joint()
    for child in base_joint.children:
        if not hasattr(child, 'children'):
            # This is a feature
            base_markers.append(child.name)

    # Get mapping of marker names -> marker idxs
    marker_indices = {}
    for feature_name in self.kin_tree.get_features():
        marker_indices[feature_name] = int(feature_name.split('_')[1])

    # Get the desired coordinates of each base marker
    base_frame_points = np.zeros((len(base_markers), 3, 1))
    all_features = self.kin_tree.get_features()
    for i, marker in enumerate(base_markers):
        base_frame_points[i, :, 0] = all_features[marker].q()

    base_idxs = [marker_indices[name] for name in base_markers]
    return base_idxs, base_frame_points

mocap_source = MocapArray(...) #or whatever other source you want
mocap_stream = mocap_source.get_stream()
mocap_stream.set_coordinates(base_idxs, base_frame_points)
self._tf_pub = tf.TransformBroadcaster()

for (frame, timestamp) in mocap_stream:
    # Compute the current mocap->object_base transform and publish it
    homog = npla.inv(mocap_stream.get_last_coordinates())
    mocap_frame_name = self.mocap_source.get_frame_name()
    if mocap_frame_name is not None:
        self._tf_pub.sendTransform(homog[0:3, 3],
                                   tf.transformations.quaternion_from_matrix(homog),
                                   rospy.Time.now(), '/object_base', '/' + mocap_frame_name)

    # Then call each MocapTracker's process_frame() with frame as the argument

#--------------------------------------------------------



class MocapTracker(object):
    """A general tracker which processes frames from a mocap source to produce some output.

    The tracker can be run on individual frames using the step() method, or can be run independently
    on the data from a MocapSource using the start() (to run in a new thread) or run() (to block
    until processing is complete) methods.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name
        self.exit = False
        self._callbacks = []

    def start(self):
        """Runs the thread to begin the tracker"""
        self.exit = False
        reader_thread = threading.Thread(target=self.run)
        reader_thread.start()
        return reader_thread

    def stop(self):
        """Stops the thread"""
        self.exit = True

    def run(self, mocap_source, mocap_transformer=None, record=False):
        """Runs the tracker in its own thread, with its own mocap source.

        This method shouldn't be used when running this tracker in a synchronized fashion
        in parallel with other MocapTrackers. It is useful, however, when you'd like to run
        a standalone tracker that, for instance, tracks a KinematicTree and publishes its state
        to a ROS topic.

        Args:
        mocap_source: MocapSource - the source to get mocap frames from, either online or offline
        mocap_transformer: MocapTransformer - a transformer to apply to each frame before it's
            consumed by this tracker
        """
        if record:
            recorded_results = []
        else:
            recorded_results = None
        stream = mocap_source.get_stream()
        for i, (frame, timestamp) in enumerate(stream):
            if self.exit:
                break
            if mocap_transformer is not None:
                frame = mocap_transformer.transform(frame)
            result = self.process_frame(frame)
            if recorded_results is not None:
                recorded_results.append(result)
        self.mocap_source.unregister_buffer(stream)
        return recorded_results

    def process_frame(self, frame):
        """Processes a frame, updates the appropriate outputs, and returns the result.

        Args:
        frame - (N,3,1) ndarray - the mocap frame to process

        Returns:
        A dict of all the result values produced by this tracker's _process_frame method
        """
        result = self._process_frame(frame)
        self.update_outputs(result)
        return result

    @abstractmethod
    def _process_frame(self, frame):
        """The specific processing this tracker should perform on each frame to produce the result.

        Client code should NOT call this method directly. Instead, use the process_frame() method to
        ensure that outputs are updated appropriately.

        Args:
        frame - (N,3,1) ndarray - the mocap frame to process

        Returns:
        A dict mapping the names of each result to the value of that result
        """
        pass

    def _update_outputs(self, result):
        """Updates all callbacks with a new result every time process_frame() is called. Subclasses
        which override this method must call the superclass implementation to ensure callbacks are
        handled correctly.

        Args:
        result: dict - the return value from process_frame
        """
        for callback in self._callbacks:
            callback(result)

    def register_callback(self, callback):
        self._callbacks.append(callback)

class MocapFrameTracker(MocapTracker):
    def __init__(self, name, tracked_frame_indices, tracked_frame_points=None):
        super(MocapFrameTracker, self).__init__(name)
        self.tracked_frame_indices = tracked_frame_indices
        self.tracked_frame_points = tracked_frame_points
        self._last_transform = np.identity(4)

    def _process_frame(self, frame):
        if self.tracked_frame_points is None:
            tracked_frame_points = frame[self.tracked_frame_indices,:,0]
            if not np.any(np.isnan(tracked_frame_points)):
                tracked_frame_points = tracked_frame_points - np.mean(tracked_frame_points, axis=1)
                self.tracked_frame_points = tracked_frame_points

        if self.tracked_frame_points is not None:
            # Find which of the specified markers are visible in this frame
            visible_inds = ~np.isnan(frame[self.tracked_frame_indices, :, 0]).any(axis=1)

            # Compute the transformation
            orig_points = frame[self.tracked_frame_indices[visible_inds], :, 0]
            desired_points = self.tracked_frame_points[visible_inds]
            try:
                homog = find_homog_trans(orig_points, desired_points)[0]
                self._last_transform = homog
            except ValueError:
                # Not enough points visible for tf.transformations to compute the transform
                homog = self._last_transform
            return {'homog': homog}
        else:
            return {}



class MocapUkfTracker(MocapTracker):
    """
    A general class which pipes mocap points from a mocap source to a UKF
    """

    def __init__(self, name, state_space_model, marker_indices, joint_states_topic=None):
        """
        Constructor
        :param state_space_model: the model which the UKF will be based on
        :param joint_states_topic: an optional topic upon which to publish tracker output
        """
        super(MocapUkfTracker, self).__init__(name)

        # copy across data
        self.state_space_model = state_space_model
        self._marker_indices = marker_indices

        # reverse the dict so we can access names for  markers too
        self._marker_names = {index: name for name, index in self._marker_indices.items()}

        self._estimation_pub = None
        # self._tf_pub = None
        self._callbacks = [new_frame_callback] if new_frame_callback is not None else []

        if joint_states_topic is not None:
            self._estimation_pub = rospy.Publisher(joint_states_topic, sensor_msgs.JointState,
                                                   queue_size=10)

        # set up the filter
        self.uk_filter = ukf.UnscentedKalmanFilter(self.state_space_model.process_model,
                                                   self.state_space_model.measurement_model,
                                                   x0=np.zeros(self.state_space_model.state_length()),
                                                   P0=np.identity(self.state_space_model.state_length()) * 0.25,
                                                   Q=np.pi / 2 / 80, R=5e-3)
        self._initialized = False

    def _unvectorize_estimation(self, state_vector=None):
        if state_vector is None:
            state_vector = np.zeros(self.state_space_model.state_length())

        return self.state_space_model.unvectorize_estimation(state_vector)

    def _process_frame(self, frame):
        # If its our first frame, converge the error in the filter
        if not self._initialized:
            state_vector, covariance, squared_residual = self._initialize_filter(frame)
            self._initialized = True
        else:
            # Otherwise store filter output
            state_vector, covariance, squared_residual = self._filter(frame)

        estimation = self._unvectorize_estimation(state_vector)

        # And call the callbacks, publishers etc
        result = {'mean':estimation, 'covariance':covariance, 'squared_residual':squared_residual}
        self._update_outputs(result)

    def _preprocess_frame(self, frame):
        # convert frame into observation dict and store
        observation = self._extract_observation(frame)

        # convert observation dict into state space measurement
        measurement = self._preprocess_measurement(observation)

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

    def update_outputs(self, result):
        """
        Calls callbacks, oublishes to topics, and to tf 
        
        Args:
        result: dict - the outputs from one step of this tracker
        """
        # Call superclass implementation so callbacks are handled properly
        super(MocapUkfTracker, self).update_outputs(result)

        # Publish joint states
        if self._estimation_pub is not None:
            msg = sensor_msgs.JointState(position=result['mean'].squeeze(),
                                         header=Header(stamp=rospy.Time.now()))
            self._estimation_pub.publish(msg)

    def _initialize_filter(self, initial_frame, reps=50):
        """
        Runs the filter to converge initial error
        :param initial_frame: initial frame
        :param reps: how many cycles to run
        :return: 
        """
        initial_measurement = self._preprocess_frame(initial_frame)

        for i in range(reps):
            self.uk_filter.filter(initial_measurement)
        return self.uk_filter.filter(initial_measurement)

    def _filter(self, frame):
        measurement_array = self._preprocess_frame(frame)
        return self.uk_filter.filter(measurement_array)

    def get_state_names(self, raw=False):
        prefix = '' if raw else None
        return self._unvectorize_estimation().keys()

    def get_marker_indices(self, marker_names):
        """Returns the marker indices for a set of marker names"""
        return [self._marker_indices[name] for name in marker_names]

    def get_marker_names(self, marker_indices):
        """Returns the marker names for a set of marker indices"""
        return [self._marker_names[index] for index in marker_indices]

    def clone(self):
        return deepcopy(self)


class KinematicTreeTracker(MocapUkfTracker):
    def __init__(self, name, kin_tree, joint_states_topic=None):
        self.kin_tree = kin_tree

        # Get all the marker indices of interest and map to their names
        marker_indices = {}
        for feature_name in self.kin_tree.get_features():
            marker_indices[feature_name] = int(feature_name.split('_')[1])

        state_space_model = kinmodel.KinematicTreeStateSpaceModel(self.kin_tree)
        super(KinematicTreeTracker, self).__init__(name, state_space_model, marker_indices, 
            joint_states_topic)

    def get_observation_func(self):
    	"""Get a function which takes a config dict for the KinematicTree being tracked
		and returns a dict of feature observations at that config.

		Note that the returned function is NOT thread safe at the moment. Calling
		obs_func while running this tracker in another thread may cause weird, intermittent bugs.
		"""
		def obs_func(state_dict):
			"""Observation function which returns the feature observations at the specifed state.

			Args:
			state_dict: dict - maps each joint's name in the tracked KinematicTree
				to that joint's state

			Returns:
			Dict mapping each feature's name to its corresponding GeometricPrimitive observation
			"""
			self._kin_tree.set_config(state_dict)
			return self._kin_tree.observe_features()
		return obs_func


class WristTracker(MocapUkfTracker, MocapWrist):
    """
    Child of MocapUkfTracker and MocapWrist specific to wrist tracking. Inheriting MocapWrist means this class inherits all
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


class FrameTracker(object):

    __metaclass__ = ABCMeta

    def __init__(self, base_tf_frame_name=None, convention='quaternion'):
        self._base_frame_name = base_tf_frame_name
        self._tf_pub_frame_names = []  # Frame names to publish on tf after an _update() call
        self._attached_frame_names = []  # All attached static frames
        self._attached_tf_frame_names = []  # All attached tf frames - update during an _update() call

        self.convention = None
        self.set_convention(convention)

        if base_tf_frame_name is not None:
            self.init_tf(base_tf_frame_name)

        else:
            self._tf_pub = None
            self._tf_listener = None

    def set_convention(self, convention):
        assert convention in kinmodel.Transform.conventions, \
            "Convention '%s' not in %s" % (convention, kinmodel.Transform.conventions.keys())
        self.convention = convention

    def get_convention(self):
        return self.convention

    def init_tf(self, base_tf_frame_name):
        self._base_frame_name = base_tf_frame_name
        self._tf_pub = tf.TransformBroadcaster()
        self._tf_listener = tf.TransformListener()

    def get_frame_names(self):
        return self._attached_frame_names + self._attached_tf_frame_names

    def is_tracked(self, frame):
        return frame in self.get_frame_names()

    @abstractmethod
    def attach_frame(self, joint_name, frame_name, tf_pub=True):
        pass

    @abstractmethod
    def set_config(self, joint_angles_dict):
        pass

    @abstractmethod
    def observe_frames(self):
        pass

    @abstractmethod
    def observe_frame(self, frame_name):
        return None

    @abstractmethod
    def compute_jacobian(self, base_frame_name, manip_name_frame, joint_angles_dict=None):
        pass

    @abstractmethod
    def get_observable_names(self):
        pass

    @abstractmethod
    def get_observables(self, configs=None, frames=None):
        pass

    @abstractmethod
    def partial_derivative(self, output_group, input_group, configs=None):
        pass


class KinematicTreeExternalFrameTracker(FrameTracker):

    def __init__(self, kin_tree, base_tf_frame_name=None, convention='quaternion'):
        self._kin_tree = kin_tree
        self._joint_names = self._kin_tree.get_joints().keys()

        super(KinematicTreeExternalFrameTracker, self).__init__(base_tf_frame_name, convention)

        self.attach_frame(self._kin_tree.get_root_joint(), 'root', pose=kinmodel.Transform())

    def attach_frame(self, joint_name, frame_name, tf_pub=True, pose=None):
        # Attach a static frame to the tree
        self._kin_tree._pox_stale = True
        self._kin_tree._dpox_stale = True
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

    def observe_frames(self, frames=None):
        all_frames = frames = set(self._attached_frame_names) | set(self._attached_tf_frame_names)

        if frames is None:
            frames = all_frames

        else:
            assert set(frames) <= all_frames, '%s is not a subset of %s' % (frames, all_frames)

        self._update()
        observations = self._kin_tree.observe_features()
        external_frame_dict = {}
        for frame_name in frames:
            external_frame_dict[frame_name] = observations[frame_name]
        return external_frame_dict

    def observe_frame(self, frame_name):
        self._update()
        observations = self._kin_tree.observe_features()
        return observations[frame_name]

    def compute_jacobian(self, base_frame_name, manip_name_frame, joint_angles_dict=None):
        if joint_angles_dict is not None:
            self.set_config(joint_angles_dict)
        self._update()
        row_names = [manip_name_frame + '_' + element for element in kinmodel.EULER_POSE_NAMES]
        minimial_jacobian = kinmodel.Jacobian(self._kin_tree.compute_jacobian(base_frame_name, manip_name_frame),
                                              row_names)

        return minimial_jacobian.pad(column_names=self._joint_names).reorder(column_names=self._joint_names)

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

        if self._tf_pub is not None:
            # Observe and publish specified static features
            for frame_name in self._tf_pub_frame_names:
                obs = feature_obs[frame_name].homog()
                self._tf_pub.sendTransform(obs[0:3, 3],
                                           tf.transformations.quaternion_from_matrix(obs),
                                           rospy.Time.now(), frame_name, self._base_frame_name)

    def get_observables(self, configs=None, frames=None):
        if configs is not None:
            self.set_config(configs)

        observables = {}
        observed_frames = self.observe_frames(frames)
        for frame in observed_frames:
            for element, value in observed_frames[frame].to_dict(self.convention).items():
                observables[frame + '_' + element] = value

        return observables

    def get_observable_names(self):
        observable_names = []
        for frame in self._attached_frame_names:
            for element, value in kinmodel.Transform.conventions[self.convention]:
                observable_names.append(frame + '_' + element)

        return observable_names

    def partial_derivative(self, output_group, input_group, configs=None):
        """
        Calculates the partial derivative between two groups of states
        :param output_group: the name of the group whose states are the output vector to the function
        :param input_group: the name of the group whose states are the input vector to the function
        :param configs: the joint angles with which to calculate this partial derivative
        :return: 
        """

        # If the groups are the same, return the identity
        if output_group == input_group:
            length = len(self._kin_tree.get_joints()) if output_group == 'configs' else 7
            return np.identity(length)

        # If the input group is the configs group, return the corresponding jacobian
        elif input_group == 'configs':
            return self.compute_jacobian('base', output_group, configs)

        # If the output group is the configs group, return the corresponding inverse jacobian
        elif output_group == 'configs':
            return self.compute_jacobian('base', input_group, configs).pinv()

        # If both groups are poses, use the chain rule to chain the corresponding jacobian and inverse jacobian
        else:
            return self.compute_jacobian('base', output_group, configs) * \
                   self.compute_jacobian('base', input_group, configs).pinv()

    def full_partial_derivative(self, coordinate_frame, configs=None):

        if configs is not None:
            self.set_config(configs)

        row_names = list(self._joint_names)

        jacobians = [np.identity(len(self._joint_names))]
        pinv_jacobians = [np.identity(len(self._joint_names))]

        for frame in self._attached_frame_names:
            if frame == coordinate_frame:
                continue
            jacobian = self.compute_jacobian(coordinate_frame, frame)
            jacobians.append(jacobian.J())
            pinv_jacobians.append(jacobian.pinv().J())
            row_names.extend(jacobian.row_names())

        column_names = list(row_names)
        jac_column_block = np.concatenate(jacobians, axis=0)
        pinv_jac_row_block = np.concatenate(pinv_jacobians, axis=1)

        return kinmodel.Jacobian.from_array(np.dot(jac_column_block, pinv_jac_row_block), row_names, column_names)

    def get_observation_func(self):
		"""Get a function which takes a config dict for the KinematicTree being tracked
		and returns a dict of frame observations at that config.

		Note that the returned function is NOT thread safe at the moment. Calling
		obs_func while running this tracker in another thread may cause weird, intermittent bugs.
		"""
		def obs_func(state_dict):
			"""Observation function which returns the frame observations at the specifed state.

			Args:
			state_dict: dict - maps each joint's name in the tracked KinematicTree
				to that joint's state

			Returns:
			Dict mapping each frame's name to its corresponding GeometricPrimitive observation
			"""
			self.set_config(state_dict)
			return self.observe_frames()
		return obs_func

	def get_jacobian_func(self, base_frame_name, manip_frame_name):
		"""Get a function which takes a config dict for the KinematicTree being tracked
		and returns a dict with a single element mapped to the jacobian of the specified manip
		frame with respect to the specified base frame.

		Note that the returned function is NOT thread safe at the moment. Calling
		obs_func while running this tracker in another thread may cause weird, intermittent bugs.
		"""
		def obs_func(state_dict):
			"""Observation function which returns the requested jacobian at the specifed state.

			Args:
			state_dict: dict - maps each joint's name in the tracked KinematicTree
				to that joint's state

			Returns:
			Dict mapping '<base>_<manip>_jacobian' to the computed Jacobian object
			"""
			jacobian = self.compute_jacobian(base_frame_name, manip_frame_name,
					joint_angles_dict=state_dict)
			jacobian_dict = {base_frame_name + '_' + manip_frame_name + '_jacobian':jacobian}
			return jacobian_dict
		return obs_func


def extract_marker_subset(frame_data, names, marker_indices):
    """
    Takes a frame of mocap points, marker names, and a dict that maps to indices and returns the subset of those names
    Should this be a method of MocapUkfTracker??
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
