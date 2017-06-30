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
import threading

class MarkerAssignments(object):
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
        primitive = kinmodel.new_geometric_primitive([0.0, 0.0, 0.0, 1.0])
        feature = kinmodel.Feature(name_prefix + str(marker_num), primitive)
        joints[joint_name].children.append(feature)

    def get_last_frame(self):
        return self._last_frame


class MocapFilePlayer(object):
    def __init__(self, npz_filename, topic='/mocap_point_cloud', framerate=50):
        data_array = np.load(npz_filename)['full_sequence']
        self._mocap_source = load_mocap.ArrayMocapSource(data_array, framerate)
        self._pub = rospy.Publisher(topic, sensor_msgs.PointCloud)
        self._thread = None
        self._run = False
        self._framerate = framerate

    def run(self, loop=True):
        while True:
            mocap_stream = self._mocap_source.get_stream()
            for (frame, timestamp) in mocap_stream:
                if not self._run:
                    return
                time.sleep(1.0 / self._framerate)
                new_message = sensor_msgs.PointCloud()
                new_message.header = std_msgs.Header()
                new_message.header.frame_id = 'mocap'
                new_message.points = []
                for point_idx in range(frame.shape[0]):
                    point = geometry_msgs.Point32()
                    point.x = frame[point_idx,0,0]
                    point.y = frame[point_idx,1,0]
                    point.z = frame[point_idx,2,0]
                    new_message.points.append(point)
                self._pub.publish(new_message)
            if not loop:
                break

    def start(self, loop=True):
        self._run = True
        self._thread = threading.Thread(target=self.run, kwargs={'loop':loop})
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._run = False
        self._thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_kinmodel_file', help='The input kinematic model')
    parser.add_argument('output_kinmodel_file', help='The output kinematic model')
    parser.add_argument('--mocap_file', help='Optional .npz file with the mocap sequence')
    args = parser.parse_args()
    rospy.init_node('marker_assignments')

    # Start playing the mocap npz file, if specified
    if args.mocap_file is not None:
        mocap_player = MocapFilePlayer(args.mocap_file)
        mocap_player.start()

    # Load the kinematic tree, list all joints, and delete existing markers
    kin_tree = kinmodel.KinematicTree(json_filename=args.input_kinmodel_file)
    tree_joints = kin_tree.get_joints()
    for joint in tree_joints:
        for child in tree_joints[joint].children:
            if not hasattr(child, 'children'):
                # This is a feature - delete it
                tree_joints[joint].children.remove(child)

    assign = MarkerAssignments(kin_tree)

    # Get the first mocap frame to determine how many markers there are
    # Wait 30 seconds for the frame and quit if it isn't received
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

    # Highlight each mocap point and ask the user for an assignment
    for i in range(num_markers):
        assign.highlight_markers([i])
        group_name = raw_input('Enter a group name for marker ' + str(i) + ' or <Enter> to skip: ')
        if group_name.strip():
            # If the string isn't empty, assign to a group
            assign.assign_marker(i, group_name)

            # Display the current group members
            assign.highlight_group(group_name)
            raw_input('Displaying group ' + group_name + ' - Press <Enter> to continue...')

    assign._tree.json(args.output_kinmodel_file)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
