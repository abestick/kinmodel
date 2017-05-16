#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import phasespace.load_mocap as load_mocap
import argparse
import kinmodel
from kinmodel.track_mocap import KinematicTreeTracker
import matplotlib.pyplot as plt


FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('mocap_npz')
    args = parser.parse_args()

    #Load the calibration sequence
    calib_data = np.load(args.mocap_npz)['full_sequence'][:,:,:]
    ukf_mocap = load_mocap.MocapArray(calib_data, FRAMERATE)


    # Load the mocap stream
    # ukf_mocap = load_mocap.PointCloudStream('/mocap_point_cloud')

    tracker_kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)


    # Add the external frames to track
    # frame_tracker = KinematicTreeExternalFrameTracker(kin_tree, 'object_base')
    # frame_tracker.attach_frame('joint2', 'trans2')
    # frame_tracker.attach_frame('joint3', 'trans3')
    # frame_tracker.attach_tf_frame('joint3', 'left_hand')
    # frame_tracker.attach_tf_frame('joint2', 'base')

    all_frames = []
    all_covariances = []
    all_residuals = []

    def new_frame_callback(i, joint_angles, covariance, squared_residual):
        # frame_tracker.set_config({'joint2':joint_angles[0], 'joint3':joint_angles[1]})
        # print(frame_tracker.compute_jacobian('base', 'left_hand'))
        print(i)
        all_frames.append(joint_angles)
        all_covariances.append(covariance[:,:,None])
        all_residuals.append(squared_residual)

    # tracker = KinematicTreeTracker(tracker_kin_tree, ukf_mocap, joint_states_topic='/kinmodel_state',
    #         object_tf_frame='/object_base', new_frame_callback=new_frame_callback)
    tracker = KinematicTreeTracker('cardboard', tracker_kin_tree, ukf_mocap, new_frame_callback=new_frame_callback)
    tracker.run()
    # tracker.start().join()
    ukf_output = np.concatenate(all_frames, axis=1)
    ukf_covar = np.concatenate(all_covariances, axis=2)
    ukf_residual = np.array(all_residuals)

    # Figure 2 - Mocap xyz trajectories
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(111)
    ax.plot(ukf_output.T)#, color='r')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\\theta$ (rad)')

    # # # Figure 1 - Box static mocap markers
    # ukf_mocap.plot_frame(0, xyz_rotation=(math.pi/2, 0, math.pi/180*120))

    # # Figure 2 - Mocap xyz trajectories
    # fig = plt.figure(figsize=(16,6))
    # time_axis = (np.array(range(calib_data.shape[2])) * (1.0 / 80))[:,None]
    # ax_x = fig.add_subplot(311)
    # ax_y = fig.add_subplot(312)
    # ax_z = fig.add_subplot(313)
    # ax_x.plot(time_axis, calib_data[:,0,:].T)#, color='r')
    # ax_x.set_ylim((0,1))
    # ax_x.set_xlabel('Time (s)')
    # ax_x.set_ylabel('X (m)')
    # ax_y.plot(time_axis, calib_data[:,1,:].T)#, color='b')
    # ax_y.set_ylim((0.5,1.5))
    # ax_y.set_xlabel('Time (s)')
    # ax_y.set_ylabel('Y (m)')
    # ax_z.plot(time_axis, calib_data[:,2,:].T)#, color='g')
    # ax_z.set_ylim((-1,0))
    # ax_z.set_xlabel('Time (s)')
    # ax_z.set_ylabel('Z (m)')


    # #UKF Output
    # fig = plt.figure(figsize=(16,7))
    # ax1 = fig.add_subplot(311)
    # ax1.plot(time_axis[:-1], ukf_output[0:1,:].T, color='b')
    # ax1.plot(time_axis[:-1], ukf_output[0,:] + ukf_covar[0,0,:], color='b', linestyle=':')
    # ax1.plot(time_axis[:-1], ukf_output[0,:] - ukf_covar[0,0,:], color='b', linestyle=':')
    # ax1.set_ylim((-0.2, 1.2))
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('$\\theta_1$ (rad)')
    # ax1.legend(['$\\mu_1$', '$\\mu_2 \\pm \\sigma^2_1$'])

    # ax2 = fig.add_subplot(312)
    # ax2.plot(time_axis[:-1], ukf_output[1:2,:].T, color='g')
    # ax2.plot(time_axis[:-1], ukf_output[1,:] + ukf_covar[1,1,:], color='g', linestyle=':')
    # ax2.plot(time_axis[:-1], ukf_output[1,:] - ukf_covar[1,1,:], color='g', linestyle=':')
    # ax2.set_ylim((-0.2, 1.2))
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('$\\theta_2$ (rad)')
    # ax2.legend(['$\\mu_2$', '$\\mu_2 \pm \\sigma^2_2$'])

    # ax3 = fig.add_subplot(313)
    # ax3.plot(time_axis[:-1], ukf_residual, color='r')
    # # ax3.plot(time_axis[:-1], ukf_output[1,:] + ukf_covar[1,1,:], color='g', linestyle=':')
    # # ax3.plot(time_axis[:-1], ukf_output[1,:] - ukf_covar[1,1,:], color='g', linestyle=':')
    # # ax3.set_ylim((-0.2, 1.2))
    # ax3.set_xlabel('Time (s)')
    # ax3.set_ylabel('SSE ($m^2$)')
    # # ax3.legend(['$\\mu_2$', '$\\mu_2 \pm \\sigma^2_2$'])
    # print('MSE: ' + str(np.mean(ukf_residual)))
    plt.pause(100)

    
if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()
