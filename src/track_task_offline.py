#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import phasespace.load_mocap as load_mocap
from phasespace.mocap_definitions import MocapWrist
import argparse
import kinmodel
from kinmodel.track_mocap import KinematicTreeTracker, WristTracker
import matplotlib.pyplot as plt


FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    parser.add_argument('trial', help='The trial number')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.task_npz)['full_sequence_' + args.trial]
    calib_data = data[:,:,:]
    ukf_mocap = load_mocap.MocapArray(calib_data, FRAMERATE)

    tracker_kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    all_frames = []
    all_covariances = []
    all_residuals = []
    all_frames1 = []
    all_covariances1 = []
    all_residuals1 = []
    all_frames2 = []
    all_covariances2 = []
    all_residuals2 = []

    def new_frame_callback(i, joint_angles, covariance, squared_residual):
        # frame_tracker.set_config({'joint2':joint_angles[0], 'joint3':joint_angles[1]})
        # print(frame_tracker.compute_jacobian('base', 'left_hand'))
        print("0: %d" % i)
        all_frames.append(joint_angles)
        all_covariances.append(covariance[:,:,None])
        all_residuals.append(squared_residual)

    object_tracker = KinematicTreeTracker('object', tracker_kin_tree, ukf_mocap, new_frame_callback=new_frame_callback)

    ref = 0
    while np.isnan(data[:,:, ref]).any():
        ref += 1
    reference_frame = data[:, :, ref]
    raw_input("Reference Frame is frame %d" % ref)
    marker_indices_1 = {name: index + 16 for index, name in enumerate(MocapWrist.names[::-1])}
    marker_indices_2 = {name: index + 24 for index, name in enumerate(MocapWrist.names[::-1])}

    def new_frame_callback_1(i, joint_angles, covariance, squared_residual):
        # frame_tracker.set_config({'joint2':joint_angles[0], 'joint3':joint_angles[1]})
        print("1: %d" % i)
        all_frames1.append(joint_angles)
        all_covariances1.append(covariance[:, :, None])
        all_residuals1.append(squared_residual)

    def new_frame_callback_2(i, joint_angles, covariance, squared_residual):
        print("2: %d" % i)
        all_frames2.append(joint_angles)
        all_covariances2.append(covariance[:, :, None])
        all_residuals2.append(squared_residual)

    wrist_tracker_1 = WristTracker('wrist1', ukf_mocap, marker_indices_1, reference_frame,
                                   new_frame_callback=new_frame_callback_1)
    wrist_tracker_2 = WristTracker('wrist2', ukf_mocap, marker_indices_2, reference_frame,
                                   new_frame_callback=new_frame_callback_2)

    object_tracker.run()
    print("Tracker 0 Done!")
    wrist_tracker_1.run()
    print("Tracker 1 Done!")
    wrist_tracker_2.run()
    print("Tracker 2 Done!")

    # object_tracker.start().join()
    ukf_output = np.concatenate(all_frames, axis=1)
    ukf_covar = np.concatenate(all_covariances, axis=2)
    ukf_residual = np.array(all_residuals)

    ukf_output1 = np.concatenate(all_frames1, axis=1)
    ukf_output2 = np.concatenate(all_frames2, axis=1)

    # Figure 1 - Mocap xyz trajectories
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(111)
    ax.plot(ukf_output.T)#, color='r')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\\theta$ (rad)')

    # Figure 2 - Mocap xyz trajectories
    fig1 = plt.figure(figsize=(16,6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(ukf_output1.T)#, color='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('$\\theta$ (rad)')

    # Figure 3 - Mocap xyz trajectories
    fig2 = plt.figure(figsize=(16,6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(ukf_output2.T)#, color='r')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$\\theta$ (rad)')

    plt.show()

    
if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()
