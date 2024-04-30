#!/usr/bin/env python3

import subprocess
import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


HEADER = (
    "stamp_sec,stamp_nanosec,stamp_frame_id,pos_x,pos_y,pos_z,or_x,or_y,or_z,or_w"
)

def tofile(text, fpath):
    subprocess.check_call(f"echo '{text}' >> '{fpath}'", shell=True)

class MySubscriber(Node):

    def __init__(self, topic, outfile):
        super().__init__("my_subscriber")
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.BEST_EFFORT,
            # history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL,
            depth=10,
        )
        self.subscription = self.create_subscription(
            msg_type=PoseStamped,
            topic=topic,
            callback=self.listener_callback,
            qos_profile=qos_profile,
        )
        print("subscribed")
        self.outfile = outfile
        self.tofile(HEADER)

    def tofile(self, msg):
        with open(self.outfile, "a") as wf:
            wf.write(msg + "\n")
            wf.flush()
        # tofile(msg, self.outfile)


    def listener_callback(self, msg):
        header = msg.header
        stamp = header.stamp
        pos = msg.pose.position
        orientation = msg.pose.orientation
        towrite = f"{stamp.sec},{stamp.nanosec},{header.frame_id},"
        towrite += f"{pos.x},{pos.y},{pos.z},"
        towrite += f"{orientation.x},{orientation.y},{orientation.z},{orientation.w}"
        # print(towrite)
        self.tofile(towrite)


parser = argparse.ArgumentParser()

parser.add_argument("topic", type=str)
parser.add_argument("output_csv", type=str)
args = parser.parse_args()

if not args.output_csv.endswith(".csv"):
    args.output_csv += ".csv"

rclpy.init()
my_subscriber = MySubscriber(args.topic, args.output_csv)
rclpy.spin(my_subscriber)

# Destroy the node explicitly
# (optional - otherwise it will be done automatically
# when the garbage collector destroys the node object)
my_subscriber.destroy_node()
rclpy.shutdown()
