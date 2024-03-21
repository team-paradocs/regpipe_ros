#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from . import open3d_conversions
import threading
import multiprocessing

class PCDRegPipe(Node):
    """A class for subscribing to a PointCloud2 message in ROS2 and processing it with Open3D.
    Attributes:
        topic (str): The name of the ROS2 topic to subscribe to for the PointCloud2 message.
        camera_frame (str): The name of the camera frame to use for the PointCloud2 message.
    """

    def __init__(self):
        super().__init__("pcd_regpipe")
        self.declare_parameter("sub_topic", "/camera/depth/color/points")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")

        self.topic = self.get_parameter("sub_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.subscriber_ = self.create_subscription(PointCloud2, self.topic, self.callback, 10)
        self._publisher_ = self.create_publisher(PointCloud2, 'processed_point_cloud', 10)
        self.last_cloud = None

        self.get_logger().info(f"Subscribing to topic {self.topic}. Press Enter to register.")

        # Start a separate thread for the input to not block the ROS2 node
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
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
        """Waits for the user input to register the latest received point cloud."""
        while rclpy.ok():
            input("Press Enter to register the latest point cloud...")
            self.process_point_cloud()

    def visualize_point_cloud(self):
        """Visualizes the last received point cloud in an Open3D window."""
        if self.last_cloud:
            # Start visualization in a separate process to not block ROS2 node
            vis_process = multiprocessing.Process(target=self._vis_process_target)
            vis_process.start()
            vis_process.join()
        else:
            self.get_logger().info("No point cloud has been received yet.")

    def _vis_process_target(self):
        """The target function for the multiprocessing process that handles visualization."""
        if self.last_cloud:
            o3d.visualization.draw_geometries([self.last_cloud])

    def process_point_cloud(self):
        """Processes the last received point cloud with Open3D."""
        if self.last_cloud:
            # Downsample the point cloud
            downsampled = self.last_cloud.voxel_down_sample(voxel_size=0.01)

            # Paint the point cloud
            downsampled.paint_uniform_color([1, 0.706, 0])

            # Publish the processed point cloud
            self.publish_point_cloud(downsampled)

    def publish_point_cloud(self, cloud):
        """Publishes the processed point cloud to a new ROS2 topic.
        Args:
            cloud (open3d.geometry.PointCloud): The processed point cloud to publish.
        """
        try:
            msg = open3d_conversions.to_msg(cloud, frame_id=self.camera_frame)
            self._publisher_.publish(msg)
            self.get_logger().info(f"Processed point cloud with {len(cloud.points)} points published.")
        except Exception as e:
            self.get_logger().error(f"Error converting point cloud to message: {e}")

def main(args=None):
    rclpy.init(args=args)
    pcd_regpipe = PCDRegPipe()
    rclpy.spin(pcd_regpipe)
    pcd_regpipe.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
