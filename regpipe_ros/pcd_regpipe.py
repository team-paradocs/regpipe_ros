#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from . import open3d_conversions
import threading
import multiprocessing
from . import proc_pipeline


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

        x_thresh = 0.3
        y_thresh = 0.15
        z_thresh = 0.3
        self.proc_pipe = proc_pipeline.PointCloudProcessingPipeline(x_thresh, y_thresh, z_thresh)

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

    def _vis_process_target(self, cloud=None):
        """The target function for the multiprocessing process that handles visualization."""
        if cloud is None:
            cloud = self.last_cloud
        if cloud:
            o3d.visualization.draw_geometries([cloud], window_name="Point Cloud",
                                            zoom=0.57000000000000017,
                                            front=[0.086126891594714566, 0.27099417737370496, -0.9587201439282379],
                                            lookat=[-0.11521162012853672, -0.39411284983247913, 1.8123871758649917],
                                            up=[-0.0069065635673039305, -0.96211034045195976, -0.27257291166787834])
            

    def process_point_cloud(self):
        """Processes the last received point cloud with Open3D."""
        if self.last_cloud:
            cloud = self.proc_pipe.run(self.last_cloud)
            self.publish_point_cloud(cloud)

            
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
