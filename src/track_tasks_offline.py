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

    object_angles = {}
    wrist_1_angles = {}
    wrist_2_angles = {}

    def new_frame_callback(i, joint_angles, covariance, squared_residual):
        # frame_tracker.set_config({'joint2':joint_angles[0], 'joint3':joint_angles[1]})
        # print(frame_tracker.compute_jacobian('base', 'left_hand'))
        print("0: %d" % i)
        for joint_name in joint_angles:
            object_angles[joint_name] = object_angles.get(joint_name, []) + [joint_angles[joint_name]]

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
        for joint_name in joint_angles:
            wrist_1_angles[joint_name] = wrist_1_angles.get(joint_name, []) + [joint_angles[joint_name]]

    def new_frame_callback_2(i, joint_angles, covariance, squared_residual):
        print("2: %d" % i)
        for joint_name in joint_angles:
            wrist_2_angles[joint_name] = wrist_2_angles.get(joint_name, []) + [joint_angles[joint_name]]


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

    # Figure 1 - Mocap xyz trajectories
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(111)
    # ax.plot(ukf_output.T)#, color='r')
    for key, value in object_angles.items():
        ax.plot(value, label=key)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.legend()

    # Figure 2 - Mocap xyz trajectories
    fig1 = plt.figure(figsize=(16,6))
    ax1 = fig1.add_subplot(111)
    for key, value in wrist_1_angles.items():
        ax1.plot(value, label=key)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (rad)')
    ax1.legend()

    # Figure 3 - Mocap xyz trajectories
    fig2 = plt.figure(figsize=(16,6))
    ax2 = fig2.add_subplot(111)
    for key, value in wrist_2_angles.items():
        ax2.plot(value, label=key)
    ax2.set_xlabel('Timestep (k)')
    ax2.set_ylabel('Angle (rad)')
    ax2.legend()

    plt.show()

    
if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()
