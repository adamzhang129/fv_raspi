#!/usr/bin/env python

from __future__ import print_function

# import roslib; roslib.load_manifest('send_command_keyboard')
import rospy

from geometry_msgs.msg import Twist
from std_msgs.msg import String

import sys, select, termios, tty

msg = """
Reading from the keyboard  and Publishing to String --> saving dataset!
-------------------------------------------------
t : label this data frame as [translational slip]
r : Label this data frame as [Rotational slip]
l : Lable this data frame as [rolling unstable]
s : Lable this data frame as [stable]

b : Reset tracking and refill displacement dataframe

CTRL-C to quit
====================================================
"""
key_range = ['t', 'r', 'l', 's', 'b']

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('/FV_l/dataset_saving_command', String, queue_size = 1)
    rospy.init_node('send_command_keyboard')
    
    string_msg = String()

    try:
        print(msg)
        while(1):
            key = getKey()
            print('you just pressed: {}'.format(key))
            if key in key_range:
                string_msg.data = key
                pub.publish(string_msg)
                print(msg)
            elif key == '\x03':
                break

    except Exception as e:
        print(e)

    finally:

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)