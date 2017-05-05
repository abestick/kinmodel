#!/usr/bin/env python
from __future__ import print_function
import rospy
import time
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import phasespace.load_mocap as load_mocap
import sys
import numpy as np
import argparse
import json
import uuid
import kinmodel

class MarkerAssignments():
    def __init__(self, kin_tree, assignments={}):
        #Create attributes to hold the last frame received, the list of markers 
        # to highlight, and the current group assignments
        self._tree = kin_tree
        self._last_frame = None
        self._highlighted_markers = []
        self._assignments = assignments

        #Subscribe to the mocap stream, and republish highlighted points on
        #a new topic
        self._pub = rospy.Publisher('/mocap_highlighted_points', 
                                    sensor_msgs.PointCloud)
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)

    def _new_frame_callback(self, message):
        #Save the frame
        self._last_frame = message

        #Generate a new PointCloud holding only the highlighted markers
        if self._highlighted_markers:
            new_message = sensor_msgs.PointCloud()
            new_message.header = std_msgs.Header()
            new_message.header.frame_id = 'mocap'
            new_message.points = []
            for marker_num in self._highlighted_markers:
                new_message.points.append(message.points[marker_num])

            #Republish the highlighted markers
            self._pub.publish(new_message)

    def highlight_markers(self, marker_nums):
        self._highlighted_markers = marker_nums

    def highlight_group(self, group_name):
        marker_nums = []
        joint = self._tree.get_joints()[group_name]
        for child in joint.children:
            if not hasattr(child, 'children'):
                marker_nums.append(int(child.name.split('_')[1]))
        self._highlighted_markers = marker_nums

    def assign_marker(self, marker_num, joint_name, name_prefix='mocap_'):
        joints = self._tree.get_joints()
        primitive = kinmodel.new_geometric_primitive([0.0,0.0,0.0,1.0])
        feature = kinmodel.Feature(name_prefix + str(marker_num), primitive)
        joints[joint_name].children.append(feature)

    def get_last_frame(self):
        return self._last_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_kinmodel_file', help='The input kinematic model')
    parser.add_argument('output_kinmodel_file', help='The output kinematic model')
    # parser.add_argument('mocap_file', help='The .npz file with the mocap sequence')
    args = parser.parse_args()
                           


    # Load the kinematic tree, list all joints, and delete existing markers
    kin_tree = kinmodel.KinematicTree(json_filename=args.input_kinmodel_file)
    tree_joints = kin_tree.get_joints()
    for joint in tree_joints:
        for child in tree_joints[joint].children:
            if not hasattr(child, 'children'):
                # This is a feature - delete it
                tree_joints[joint].children.remove(child)

    # # Load the mocap file
    # mocap_file = np.load(args.mocap_file)
    # mocap_array = mocap_file['mocap']

    # # List all markers seen at least once
    # markers_seen = np.where(np.any(np.logical_not(np.isnan(mocap_array[:,0,:])), axis=1))

    # # Iterate over each marker, ask for a joint assignment, and add a corresponding feature




    rospy.init_node('marker_assignments')
    assign = MarkerAssignments(kin_tree)

    #Get the first mocap frame to determine how many markers there are
    #Wait 30 seconds for the frame and quit if it isn't received
    num_markers = None
    for i in range(300):
        frame = assign.get_last_frame()
        if frame is not None:
            num_markers = len(frame.points)
            break
        else:
            time.sleep(0.1)
    if num_markers is None:
        print('Error: No mocap stream on /mocap_point_cloud')
        print('Quitting now...')
        return

    #Highlight each mocap point and ask the user for an assignment
    for i in range(num_markers):
        assign.highlight_markers([i])
        group_name = raw_input('Enter a group name for marker ' + str(i) + ' or <Enter> to skip: ')
        if group_name.strip():
            #If the string isn't empty, assign to a group
            assign.assign_marker(i, group_name)

            #Display the current group members
            assign.highlight_group(group_name)
            raw_input('Displaying group ' + group_name + ' - Press <Enter> to continue...')

    assign._tree.json(args.output_kinmodel_file)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
