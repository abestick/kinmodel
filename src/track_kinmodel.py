#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import phasespace.load_mocap as load_mocap
import argparse
import kinmodel
import matplotlib.pyplot as plt
import rospy
from kinmodel.track_mocap import KinematicTreeTracker
from extra_baxter_tools.conversions import matrix_to_pose_msg, array_to_point_msg, stamp
import tf
from tf.transformations import inverse_matrix
from geometry_msgs.msg import PoseStamped, PointStamped, Point32, Point, Quaternion, Pose
from sensor_msgs.msg import PointCloud
import rospy
import sys

FRAMERATE = 50


def track(kinmodel_json_optimized):

    #Load the calibration sequence
    calib_data = np.load('/home/pedge/experiment/shoulder_peter/shoulder_peter_rec.npz')
    ukf_mocap = load_mocap.ArrayMocapSource(calib_data['full_sequence'][:,:,:], FRAMERATE).get_stream()

    # Set the base frame coordinate transformation
    zc_points = ukf_mocap.read()[0]
    zc_point_cloud = PointCloud(points=[Point32(*xyz) for xyz in zc_points])
    zc_point_cloud.header.frame_id = 'world'
    pc_pub = rospy.Publisher('zero_conf_points', PointCloud, queue_size=100)


    rospy.init_node('kin_tree_tracker')
    plt.ion()

    #Load the calibration sequence
    # calib_data = np.load(args.mocap_npz)
    # ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Load the mocap stream
    ukf_mocap = load_mocap.PointCloudMocapSource('/mocap_point_cloud')
    kin_tree = kinmodel.KinematicTree(json_filename=kinmodel_json_optimized)
    # tracker_kin_tree.json('new_'+kinmodel_json_optimized)
    params = kin_tree.get_params()
    shoulder_point = params['shoulder'].params
    shoulder_point = array_to_point_msg(shoulder_point)
    elbow_xi = params['elbow'].params
    elbow_nu = elbow_xi[:3]
    elbow_w = elbow_xi[3:]
    elbow_q = np.cross(elbow_w, elbow_nu)
    quat_xyz = np.cross((1, 0, 0), elbow_w)
    quat = list(quat_xyz)  + [1+np.dot((1, 0, 0), elbow_w)]
    elbow_pose = Pose(position=Point(*elbow_q), orientation=Quaternion(*quat))
    tracker_kin_tree = kin_tree.to_1d_chain()

    tf_pub = tf.TransformBroadcaster()
    point_pub = rospy.Publisher('shoulder_point', PointStamped, queue_size=100)
    pose_pub = rospy.Publisher('elbow_pose', PoseStamped, queue_size=100)

    # Add the external frames to track
    # frame_tracker = KinematicTreeExternalFrameTracker(kin_tree, 'object_base')
    # frame_tracker.attach_frame('joint2', 'trans2')
    # frame_tracker.attach_frame('joint3', 'trans3')
    # frame_tracker.attach_tf_frame('joint3', 'left_hand')
    # frame_tracker.attach_tf_frame('joint2', 'base')


    tracker = KinematicTreeTracker('test', tracker_kin_tree, 
        joint_states_topic='/kinmodel_state')

    def publish_extras(*args, **kwargs):
        transform = inverse_matrix(tracker.get_base_transform())
        tf_pub.sendTransform(transform[0:3, 3],
            tf.transformations.quaternion_from_matrix(transform),
            rospy.Time.now(), 'base_frame', 'world')
        point_msg = stamp(shoulder_point, frame_id='base_frame')
        pose_msg = stamp(elbow_pose, frame_id='base_frame')
        point_pub.publish(point_msg)
        pose_pub.publish(pose_msg)
        zc_point_cloud.header.stamp = rospy.Time.now()
        pc_pub.publish(zc_point_cloud)

    tracker.register_callback(publish_extras)
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
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    track(args.kinmodel_json_optimized)
