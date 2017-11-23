#!/usr/bin/env python
import rospy
import kinmodel
from sensor_msgs.msg import JointState, PointCloud
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from kinmodel.track_mocap import KinematicTreeOptimalTracker
import numpy as np

class KinmodelStreamer(object):

    def __init__(self, json_filename, topic='kinmodel/point_cloud'):
        self.kin_tree = kinmodel.KinematicTree(json_filename=json_filename).to_1d_chain()
        self.kin_tree.set_zero_config()
        self.sub = rospy.Subscriber('kinmodel/joint_states', JointState, self.publish_point_cloud)
        self.pub = rospy.Publisher(topic, PointCloud, queue_size=100)

    def publish_point_cloud(self, msg):
        joints = dict(zip(msg.name, msg.position))
        self.kin_tree.set_config(joints)
        point_cloud_msg = PointCloud(header=Header(frame_id='base_frame', stamp=rospy.Time.now()),
                                     points=[Point32(*p.q()) for p in self.kin_tree.observe_features().values()])
        self.pub.publish(point_cloud_msg)


class KinmodelTrackerPublisher(object):
    def __init__(self, json_filename, topic='kinmodel/joint_states'):
        self.kin_tree = kinmodel.KinematicTree(json_filename=json_filename).to_1d_chain()
        self.tracker = KinematicTreeOptimalTracker(self.kin_tree)
        self.pub = rospy.Publisher(topic, JointState, queue_size=100)
        self.sub = rospy.Subscriber('mocap_point_cloud', PointCloud, self.publish_joint_states)

    def publish_joint_states(self, msg):
        configs = self.tracker.process_msg(msg, 1.0)
        msg = JointState(header=Header(stamp=rospy.Time.now()), name=self.tracker.config_order,
                         position=np.rad2deg(self.tracker.get_config_vec(configs)))
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('kinmodel_point_cloud_streamer')
    json_filename = rospy.get_param('~kinmodel')
    ks = KinmodelStreamer(json_filename)
    ktp = KinmodelTrackerPublisher(json_filename)
    rospy.spin()
