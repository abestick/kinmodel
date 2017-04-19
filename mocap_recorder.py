#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy.linalg as la 
import rospy
import time
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import numpy as np
import json
from collections import deque
from math import atan2
import tf.transformations as convert
import uuid
import kinmodel

parser = argparse.ArgumentParser()
parser.add_argument('kinmodel_json', help='The JSON file with the kinematic model data')
parser.add_argument('output_npz', help='The output mocap data file')
args = parser.parse_args()

# Generate chains from the kinematic model file
kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json)
features = kin_tree.get_features()
tree_marker_nums = []
assignments = {}
for feature_name in features:
    assignments[feature_name] = int(feature_name.split('_')[1])

CHAINS = {}
CHAINS['tree'] = assignments.keys()

class MocapRecorder():
    def __init__(self, max_length=57600):
        self._frames = deque()
        self._annotations = []
        self._record = False
        self._current_frame = -1
        self._max_length = max_length
        self._annotate_when_visible = None
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)

    def record(self):
        self._record = True

    def stop(self):
        self._record = False

    def _new_frame_callback(self, message):
        if self._current_frame >= self._max_length - 1:
            raise RuntimeError('MocapRecorder is full')

        if self._record:
            frame = point_cloud_to_array(message)
            self._frames.append(frame)
            self._current_frame += 1
            if self._annotate_when_visible is not None:
                if not np.any(np.isnan(frame[self._annotate_when_visible[0],:,0])):
                    self.annotate(self._annotate_when_visible[1])
                    self._annotate_when_visible = None

    def annotate(self, label=None):
        if not self._record:
            raise RuntimeError('Can\'t annotate when recording is stopped')

        if self._current_frame == -1:
            self._annotations.append((0, label))
        else:
            self._annotations.append((self._current_frame, label))

    def annotate_next_visible(self, marker_nums, label=None):
        self._annotate_when_visible = (np.array(marker_nums), label)
        while self._annotate_when_visible is not None:
            time.sleep(0.05)

    def get_array(self):
        return np.dstack(self._frames)

    def get_annotations(self):
        return self._annotations

    def close(self):
        self.stop()
        self._sub.unregister()


def point_cloud_to_array(message):
    num_points = len(message.points)
    data = np.empty((num_points, 3, 1))
    for i, point in enumerate(message.points):
        data[i,:,0] = [point.x, point.y, point.z]
    return data

def get_closest_visible(data, index):
    print('Requested index: ' + str(index))
    if not np.any(np.isnan(data[:,:,index])):
        print('Actual index: ' + str(index))
        return data[:,:,index], index
    else:
        for i in range(1, data.shape[2]):
            if index+i < data.shape[2] and not np.any(np.isnan(data[:,:,index+i])):
                print('Actual index: ' + str(index+i))
                return data[:,:,index+i], index+i
            elif index-i >= 0 and not np.any(np.isnan(data[:,:,index-i])):
                print('Actual index: ' + str(index-i))
                return data[:,:,index-i], index-i
        raise RuntimeError('Markers are occluded in every frame of data')

def main():
    #Load the marker assignments and make a list of all markers
    all_markers = []
    for chain in CHAINS.keys():
        groups = CHAINS[chain]
        for group in groups:
            all_markers.append(assignments[group])
    all_markers = list(set(all_markers))
    print('Tracking markers:')
    print(all_markers)

    #Start recording frames
    rospy.init_node('mocap_calibrate_human')
    recorder = MocapRecorder()

    #Capture the 0-config
    raw_input('RECORDING: Press <Enter> to capture the 0-configuration: ')
    print('Waiting for all markers to be visible...')
    recorder.record()
    recorder.annotate_next_visible(all_markers, 'zero_config')

    #Capture the calibration sequence
    frame = 1
    for chain in CHAINS.keys():
        while True:
            command = raw_input('RECORDING ' + chain + ': Press <Enter> to capture a pose or n+<Enter> to move to the next chain: ')
            if command is 'n':
                break
            else:
                recorder.annotate(chain)
                print('Captured frame ' + str(frame))
                frame += 1
    recorder.stop()

    #Generate the sets of training samples
    print('Extracting chains:')
    calib_frames = {chain:[] for chain in CHAINS.keys()}
    for chain in CHAINS.keys():
        groups = CHAINS[chain]
        group_markers = []
        for group in groups:
            group_markers.append(assignments[group])
        print(chain + ': ' + str(group_markers))
        group_markers = np.array(group_markers)

        #Select only the marker data for this group
        group_data = recorder.get_array()[group_markers,:,:]

        #Pull out the frame nearest to each annotation where all the group's
        #markers are visible
        for annotation in recorder.get_annotations():
            if annotation[1] == chain:
                calib_frames[chain].append(get_closest_visible(group_data, int(annotation[0]))[1])
        li = []
        li.extend(set(calib_frames[chain]))
        li.sort()
        calib_frames[chain] = np.array(li)

    #Set the zero config as the first frame
    print('calib_frames:')
    print(calib_frames)
    first_frame = int(recorder.get_annotations()[0][0])
    for chain in CHAINS.keys():
        calib_frames[chain] = calib_frames[chain] - first_frame
    calib_frames['full_sequence'] = recorder.get_array()[:,:,first_frame:]

    #Add the full sequence and the id of the marker assignments to the dataset,
    #then write it to a file
    with open(args.output_npz, 'w') as output_file:
        np.savez_compressed(output_file, **calib_frames)
        print('Calibration sequences saved to ' + args.output_npz)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
