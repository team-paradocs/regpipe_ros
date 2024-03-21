#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d
from . import open3d_conversions
import threading


class PCDSubscriber(Node):
    """A class for subscribing to a PointCloud2 message in ROS2 and displaying it in Open3D.

    This class subscribes to a PointCloud2 message on a specified topic and displays it in Open3D.

    Attributes:
        topic (str): The name of the ROS2 topic to subscribe to for the PointCloud2 message.
        camera_frame (str): The name of the camera frame to use for the PointCloud2 message.
    """

    def __init__(self):
        super().__init__("pcd_subscriber")
        self.declare_parameter("sub_topic", "/camera/depth/color/points")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")

        self.topic = self.get_parameter("sub_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.subscriber_ = self.create_subscription(PointCloud2, self.topic, self.callback, 10)
        self.last_cloud = None

        self.get_logger().info(f"Subscribing to topic {self.topic}. Press Enter to visualize.")

        # Start a separate thread for the input to not block the ROS2 node
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.start()

    

    def callback(self, msg):
        """Callback function for the subscriber.

        Args:
            msg (sensor_msgs.msg.PointCloud2): The incoming PointCloud2 message.
        """
        try:
            cloud = open3d_conversions.from_msg(msg)
            self.last_cloud = cloud
            self.get_logger().info(f"Received point cloud with {len(cloud.points)} points.")
        except Exception as e:
            self.get_logger().error(f"Error converting message to point cloud: {e}")


    def wait_for_input(self):
        """Wait for the user to press Enter and visualize the last received point cloud."""
        input("Press Enter to visualize the last received point cloud...")
        if self.last_cloud is not None:
            open3d.visualization.draw_geometries([self.last_cloud])
        else:
            self.get_logger().info("No point cloud received yet.")

def main(args=None):
    rclpy.init(args=args)
    pcd_subscriber = PCDSubscriber()
    rclpy.spin(pcd_subscriber)
    pcd_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()