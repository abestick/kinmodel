from __future__ import print_function
import numpy as np
import scipy as sp
import numpy.linalg as np_la
import sys
import phasespace.load_mocap as load_mocap
import argparse
import random
import scipy.optimize

import matplotlib.pyplot as plt
import cProfile
import json
import kinmodel
import ukf
import matplotlib.pyplot as plt


# Need: base frame marker indices, dicts of marker observations (name->primitive) at each sample point

FRAMERATE = 50
GROUP_NAME = 'tree'

def main():
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('mocap_npz')
    args = parser.parse_args()

    #Load the calibration sequence
    calib_data = np.load(args.mocap_npz)
    ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Get the base marker indices
    base_indices = []
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    base_joint = kin_tree.get_root_joint()
    for child in base_joint.children:
        if not hasattr(child, 'children'):
            # This is a feature
            base_indices.append(int(child.name.split('_')[1]))

    # Get all the marker indices of interest
    all_marker_indices = []
    for feature_name in kin_tree.get_features():
        all_marker_indices.append(int(feature_name.split('_')[1]))

    # Set the base coordinate transform for the mocap stream
    desired = ukf_mocap.read()[0][base_indices,:,0]
    desired = desired - np.mean(desired, axis=0)
    ukf_mocap.set_coordinates(base_indices, desired, mode='time_varying')

    # Create the observation and measurement models
    test_ss_model = kinmodel.KinematicTreeStateSpaceModel(kin_tree)
    measurement_dim = len(test_ss_model.measurement_model(np.array([0.0, 0.0])))
    state_dim = 2

    # Create a list to hold all the output arrays
    ukf_output = []

    # Run the filter
    for i, (frame, timestamp) in enumerate(ukf_mocap):
        feature_dict = {}
        for marker_idx in all_marker_indices:
            obs_point = kinmodel.new_geometric_primitive(
                    np.concatenate((frame[marker_idx,:,0], np.ones(1))))
            feature_dict['mocap_' + str(marker_idx)] = obs_point

        if i == 0:
            print('Initializing UKF...', end='')
            sys.stdout.flush()
            initial_obs = test_ss_model.vectorize_measurement(feature_dict)
            uk_filter = ukf.UnscentedKalmanFilter(test_ss_model.process_model, test_ss_model.measurement_model, np.zeros(2), np.identity(2)*0.25)
            for i in range(50):
                uk_filter.filter(initial_obs)
            print('Done!')
        else:
            print('UKF Step: ' + str(i) + '/' + str(len(ukf_mocap)), end='\r')
            obs_array = test_ss_model.vectorize_measurement(feature_dict)
            ukf_output.append(uk_filter.filter(obs_array)[0])
    ukf_output = np.concatenate(ukf_output, axis=1)
    plt.plot(ukf_output.T)
    plt.pause(10)
    
if __name__ == '__main__':
    cProfile.run('main()', 'fit_kinmodel.profile')
    # main()
