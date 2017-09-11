#!/usr/bin/env python
"""Script to track a kinematic tree model given a stream of mocap data.

To track an already-fitted model:

rosrun kinmodel track_kinmodel.py <kinmodel_json_optimized>

where the args are:
- <kinmodel_json_optimized>: the final kinematic model JSON file created by fit_kinmodel.py
"""
from __future__ import print_function
import numpy as np
import phasespace.load_mocap as load_mocap
import argparse
import kinmodel
import matplotlib.pyplot as plt
import rospy
from kinmodel.track_mocap import KinematicTreeTracker

FRAMERATE = 50


def track(kinmodel_json_optimized):
    rospy.init_node('kin_tree_tracker')
    plt.ion()

    #Load the calibration sequence
    # calib_data = np.load(args.mocap_npz)
    # ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Load the mocap stream
    ukf_mocap = load_mocap.PointCloudMocapSource('/mocap_point_cloud')
    tracker_kin_tree = kinmodel.KinematicTree(json_filename=kinmodel_json_optimized)
    # kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)


    # Add the external frames to track
    # frame_tracker = KinematicTreeExternalFrameTracker(kin_tree, 'object_base')
    # frame_tracker.attach_frame('joint2', 'trans2')
    # frame_tracker.attach_frame('joint3', 'trans3')
    # frame_tracker.attach_tf_frame('joint3', 'left_hand')
    # frame_tracker.attach_tf_frame('joint2', 'base')


    tracker = KinematicTreeTracker('test', tracker_kin_tree, 
        joint_states_topic='/kinmodel_state')

    #Run the tracker in the main thread for now
    tracker.run(ukf_mocap)
    
    # tracker.start()
    # rospy.spin()
    # tracker.stop()


    
if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    # parser.add_argument('mocap_npz')
    args = parser.parse_args()
    track(args.kinmodel_json_optimized)
