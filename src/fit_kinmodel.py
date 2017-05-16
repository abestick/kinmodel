#!/usr/bin/env python
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
import kinmodel.src.kinmodel.kinmodel.src.kinmodel.kinmodel
import matplotlib.pyplot as plt

FRAMERATE = 50
GROUP_NAME = 'tree'

def main():
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json', help='The kinematic model JSON file')
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('mocap_npz', help='The .npz file from mocap_recorder.py')
    args = parser.parse_args()

    #Load the calibration sequence
    calib_data = np.load(args.mocap_npz)
    ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Get the base marker indices
    base_indices = []
    kin_tree = kinmodel.src.kinmodel.kinmodel.KinematicTree(json_filename=args.kinmodel_json)
    base_joint = kin_tree.get_root_joint()
    for child in base_joint.children:
        if not hasattr(child, 'children'):
            # This is a feature
            base_indices.append(int(child.name.split('_')[1]))

    # Get all the marker indices of interest
    all_marker_indices = []
    for feature_name in kin_tree.get_features():
        all_marker_indices.append(int(feature_name.split('_')[1]))

    # Set the base frame coordinate transformation
    desired = ukf_mocap.read()[0][base_indices,:,0]
    desired = desired - np.mean(desired, axis=0)
    data_array = np.dstack((calib_data['full_sequence'][:,:,0:1],
                                calib_data['full_sequence'][:,:,calib_data[GROUP_NAME]]))
    mocap = load_mocap.MocapArray(data_array, FRAMERATE)    

    #Set the coordinate frame for the mocap sequence
    mocap.set_coordinates(base_indices, desired, mode='time_varying')
    
    #Generate the feature observation dicts
    feature_obs = []
    for frame, timestamp in mocap:
        feature_dict = {}
        for marker_idx in all_marker_indices:
            if not np.isnan(frame[marker_idx,0,0]):
                obs_point = kinmodel.src.kinmodel.kinmodel.new_geometric_primitive(np.concatenate((frame[marker_idx, :, 0], np.ones(1))))
                feature_dict['mocap_' + str(marker_idx)] = obs_point
        feature_obs.append(feature_dict)

    # Run the optimization
    kin_tree.set_features(feature_obs[0])
    final_configs, final_twists, final_features = kin_tree.fit_params(feature_obs,
            optimize={'twists':True, 'features':False, 'configs':True})
    print('Second optimization...')
    final_configs, final_twists, final_features = kin_tree.fit_params(feature_obs, configs=final_configs,
            optimize={'twists':True, 'features':True, 'configs':True})
    kin_tree.json(args.kinmodel_json_optimized)
    
if __name__ == '__main__':
    cProfile.run('main()', 'fit_kinmodel.profile')
    # main()
