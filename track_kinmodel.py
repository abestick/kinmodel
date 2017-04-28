from __future__ import print_function
import numpy as np
import scipy as sp
import numpy.linalg as np_la
import sys
import phasespace.load_mocap as load_mocap
import argparse
import random
import scipy.optimize
import threading
import matplotlib.pyplot as plt
import cProfile
import json
import kinmodel
import ukf
import matplotlib.pyplot as plt


FRAMERATE = 50
GROUP_NAME = 'tree'

class KinematicTreeTracker(object):
    def __init__(self, kin_tree, mocap_source, joint_states_topic=None, object_tf_frame=None,
            new_frame_callback=None, return_array=False):
        self.kin_tree = kin_tree
        self.mocap_source = mocap_source
        self._joint_states_pub = None
        self._tf_pub = None
        self._callback = new_frame_callback
        self._return_array = return_array
        self.exit = False

        # Import ROS dependencies only if needed - can run w/o ROS otherwise
        if joint_states_topic is not None or object_tf_frame is not None:
            import rospy
            rospy_init_node('kin_tree_tracker')
            if joint_states_topic is not None:
                import sensor_msgs.msg as sensor_msgs
                from std_msgs.msg import Header
                self._joint_states_pub = rospy.Publisher(joint_states_topic, sensor_msgs.JointState,
                        queue_size=10)
            if object_tf_frame is not None:
                import tf
                import tf.transformations
                self.tf_pub = tf.TransformBroadcaster()

    def start(self):
        self.exit = False
        reader_thread = Thread(target=self.run)
        reader_thread.start()

    def stop(self):
        self.exit = True

    def run(self):
        # Get the base marker indices
        base_indices = []
        base_joint = self.kin_tree.get_root_joint()
        for child in base_joint.children:
            if not hasattr(child, 'children'):
                # This is a feature
                base_indices.append(int(child.name.split('_')[1]))

        # Get all the marker indices of interest
        all_marker_indices = []
        for feature_name in self.kin_tree.get_features():
            all_marker_indices.append(int(feature_name.split('_')[1]))

        # Set the base coordinate transform for the mocap stream
        desired = np.zeros((len(base_indices), 3, 1))
        all_features = self.kin_tree.get_features()
        for i, idx in enumerate(base_indices):
            desired[i,:,0] = all_features['mocap_' + str(idx)].q()
        self.mocap_source.set_coordinates(base_indices, desired, mode='time_varying')

        # Create the observation and measurement models
        test_ss_model = kinmodel.KinematicTreeStateSpaceModel(self.kin_tree)
        measurement_dim = len(test_ss_model.measurement_model(np.array([0.0, 0.0])))
        state_dim = 2

        # Run the filter
        ukf_output = []
        for i, (frame, timestamp) in enumerate(self.mocap_source):
            if self.exit:
                break
            feature_dict = {}
            for marker_idx in all_marker_indices:
                obs_point = kinmodel.new_geometric_primitive(
                        np.concatenate((frame[marker_idx,:,0], np.ones(1))))
                feature_dict['mocap_' + str(marker_idx)] = obs_point
            if i == 0:
                print('Initializing UKF...', end='')
                sys.stdout.flush()
                initial_obs = test_ss_model.vectorize_measurement(feature_dict)
                uk_filter = ukf.UnscentedKalmanFilter(test_ss_model.process_model,
                        test_ss_model.measurement_model, np.zeros(2), np.identity(2)*0.25)
                for i in range(50):
                    uk_filter.filter(initial_obs)
                print('Done!')
            else:
                # print('UKF Step: ' + str(i) + '/' + str(len(ukf_mocap)), end='\r')
                # sys.stdout.flush()
                obs_array = test_ss_model.vectorize_measurement(feature_dict)
                joint_angles = uk_filter.filter(obs_array)[0]
                if self._return_array:
                    ukf_output.append(joint_angles)

                if self._joint_states_pub is not None:
                    msg = sensor_msgs.JointState(position=joint_angles.squeeze(),
                            header=Header(stamp=rospy.Time.now()))
                    self._joint_states_pub.publish(msg)

                # Publish the base frame pose of the flexible object
                if self._tf_pub is not None:
                    homog = self.mocap_source.get_last_coordinates()
                    mocap_frame_name = self.mocap_source.get_frame_name()
                    if mocap_frame_name is not None:
                        self._tf_pub.sendTransform(homog[0:3,3],
                                tf.transformations.quaternion_from_matrix(homog[0:3,0:3]),
                                rospy.Time.now(), "/object_base", mocap_frame_name)
        if self._return_array:
            ukf_output = np.concatenate(ukf_output, axis=1)
            return ukf_output

def main():
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('mocap_npz')
    args = parser.parse_args()

    #Load the calibration sequence
    calib_data = np.load(args.mocap_npz)
    ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Load the mocap stream
    # ukf_mocap = load_mocap.PointCloudStream('/mocap_point_cloud')
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    tracker = KinematicTreeTracker(kin_tree, ukf_mocap, return_array=True)
    ukf_output = tracker.run()

    plt.plot(ukf_output.T)
    plt.pause(100)

    
if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()
