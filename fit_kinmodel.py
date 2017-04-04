import numpy as np
import scipy as sp
import numpy.linalg as np_la
import sys
import load_mocap
import argparse
import random
# import uk
import scipy.optimize

import matplotlib.pyplot as plt
import cProfile
import json
import kinmodel

# Need: base frame marker indices, dicts of marker observations (name->primitive) at each sample point

FRAMERATE = 50
GROUP_NAME = 'tree'

def main():
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json', help='The kinematic model JSON file')
    parser.add_argument('mocap_npz', help='The .npz file from mocap_recorder.py')
    args = parser.parse_args()

    #Load the calibration sequence
    calib_data = np.load(args.mocap_npz)
    ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Get the base marker indices
    base_indices = []
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json)
    base_joint = kin_tree.get_root_joint()
    for child in base_joint.children:
        if not hasattr(child, 'children'):
            # This is a feature
            base_indices.append(int(child.name.split('_')[1]))

    desired = ukf_mocap.get_frames()[base_indices,:,0]
    desired = desired - np.mean(desired, axis=0)

    # ukf_mocap.set_sampling(1000)
    # ukf_mocap.set_coordinates(base_indices, desired, mode='time_varying')

    data_array = np.dstack((calib_data['full_sequence'][:,:,0:1],
                                calib_data['full_sequence'][:,:,calib_data[GROUP_NAME]]))
    mocap = load_mocap.MocapArray(data_array, FRAMERATE)    

    #Set the coordinate frame for the mocap sequence
    mocap.set_coordinates(base_indices, desired, mode='time_varying')
    
    #Generate the feature observation dicts
    feature_obs = []
    for frame, timestamp in mocap:
        feature_dict = {}
        for marker_idx in range(mocap.get_num_points()):
            if not np.isnan(frame[marker_idx,0,0]):
                obs_point = kinmodel.new_geometric_primitive(np.concatenate((frame[marker_idx,:,0], np.ones(1))))
                feature_dict['mocap_' + str(marker_idx)] = obs_point
        feature_obs.append(feature_dict)

    # Run the optimization
    final_configs, final_twists, final_features = kin_tree.fit_params(feature_obs, )

    # Test visualization
    kin_tree.compute_error(final_configs[3], feature_obs[3], vis=True)
    1/0


        # #Save the initial parameter estimates
        # xi0 = sk.twists()
        # ft0 = fts_[:,:,0]
        # # ft0 = np.vstack([j.ft0 for j in sk.nodes if j.ft0.size > 0])
        # n = xi0.shape[0]
        # m = ft0.shape[0]
        # N = fts_.shape[2]-1
        # sk.dims = (n,m,N)
        # ths0 = np.zeros((N+1, n))
        # x0 = sk.encode(xi0,ft0,ths0)

        # #Estimate the skeleton parameters and write to BVH
        # opt = dict(maxfev=100)
        # start_time = time()
        # x0,x = sk.fit(fts_,xi0,ft0,ths0=None,dbg=True,opt=opt)
        # run_time = time() - start_time
        # xi0_,ft0_,ths0 = sk.decode(x0)
        # xi,ft,ths = sk.decode(x)
        # bvh_out = sk._root.bvh()
        # # with open(PREFIX + group_name + BVH_OUT, 'w') as bvh_file:
        #     for line in bvh_out:
        #         bvh_file.write(line + '\n')

        # #Initialize and run the UKF
        # plt.figure(1)
        # ukf = Mocap(sk,Q=2.0e-3,R=0.2e-3,viz=False, dt=ukf_mocap.get_timestamps()[1]-ukf_mocap.get_timestamps()[0])
        # ukf.Ninit = 50
        # z = uk.mocap(ukf, ukf_mocap.get_frames()[group_indices,:,:].T)

        # #Plot the results
        # fig = plt.figure(1)
        # axes1 = fig.add_subplot(2,1,1)
        # axes1.plot(mocap.get_timestamps(), ths)
        # axes2 = fig.add_subplot(2,1,2)
        # # axes2.plot(np.tile(ukf_mocap.get_timestamps(), (z.shape[0],1)).T, z.T)
        # axes2.plot(np.tile(np.arange(0,z.shape[1]), (z.shape[0],1)).T, z.T)
        # axes2.legend(('1','2','3','4','5'))
        # axes1.legend(('1','2','3','4','5'))
        # plt.draw()

        # #Compute the predicted observations based on the UKF trajectory
        # ukf_obs = np.zeros(ukf_mocap.get_frames()[group_indices,:,:].shape)
        # ukf_err = np.zeros(ukf_mocap.get_frames()[group_indices,:,:].shape[2])
        # for i in range(ukf_obs.shape[2]):
        #     sk.angles(z[:,i])
        #     sk.pox()
        #     ukf_fts = sk.fts()
        #     # ukf_err[i] = np.sum(np_la.norm(ukf_fts-ukf_mocap.get_frames()[group_indices,:,i], axis=1))
        #     ukf_err[i] = np.sum(ukf_fts-ukf_mocap.get_frames()[group_indices,:,i])
        # fig = plt.figure(2)
        # axes1 = fig.add_subplot(1,1,1)
        # axes1.plot(ukf_mocap.get_timestamps(), ukf_err)
        # plt.draw()

        # #Compute the hand locations
        # hand_loc = np.mean(ukf_mocap.get_frames()[assignments[HAND_MARKERS[group_name]],:,0], axis=0)
        # print(hand_loc)

        # #Extract the joint limits
        # min_frame = int(raw_input('Minimum index for joint limits: '))
        # max_frame = int(raw_input('Maximum index for joint limits: '))
        # min_angles = np.amin(z.T[min_frame:max_frame,:], axis=0)
        # max_angles = np.amax(z.T[min_frame:max_frame,:], axis=0)
        # print(min_angles)
        # print(max_angles)

        # #Save the results to a file
        # np.savez_compressed(PREFIX + group_name + NPZ_OUT, base_indices=base_indices, 
        #                     base_config=desired, xi=xi, ft=ft, ths=ths, 
        #                     timestamps=mocap.get_timestamps(), traj=z.T,
        #                     hand_loc=hand_loc, min_angles=min_angles,
        #                     max_angles=max_angles)
        
        # print 'MSE err(x0) = %8.6f m' % (np.sum(sk.err(x0)**2,axis=0) / (N+1))
        # print 'MSE err(x ) = %8.6f m' % (np.sum(sk.err(x )**2,axis=0) / (N+1))

        # print 'Median reprojection error (x0) = %8.6f m' % np.median(np_la.norm(sk.err_matrix(x0), axis=2).flatten())
        # print 'Median reprojection error (x) = %8.6f m' % np.median(np_la.norm(sk.err_matrix(x), axis=2).flatten())

        # print 'xi0,xi'
        # print np.round(xi0,4),'\n','\n',np.round(xi,4)
        # # print 'ft0,ft_,ft'
        # # print np.round(ft0,4),'\n',np.round(1.*fts_,4),'\n',np.round(ft,4)
        # plt.close('all')
    
if __name__ == '__main__':
    cProfile.run('main()', 'fit_kinmodel.profile')
    # main()
