#!/usr/bin/env python

import rospy
from kinmodel.mocap_recorder import collect_task_data

if __name__ == '__main__':
    try:
        collect_task_data()
    except rospy.ROSInterruptException:
        pass