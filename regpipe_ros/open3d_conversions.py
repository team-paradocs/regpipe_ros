#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d
import ros2_numpy as ros_numpy
from sensor_msgs.msg import PointCloud2 
from numpy.lib import recfunctions
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
                [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]


def to_msg(open3d_cloud, frame_id=None, stamp=None):
    header = Header()
    if stamp is not None:
        header.stamp = stamp
    if frame_id is not None:
        header.frame_id = frame_id

    o3d_points = np.asarray(open3d_cloud.points, dtype=np.float32)

    if not open3d_cloud.colors:  # XYZ only
        cloud_msg = ros_numpy.point_cloud2.array_to_xyz_pointcloud2(o3d_points, header)
    else:  # XYZRGB
        o3d_colors = np.asarray(open3d_cloud.colors, dtype=np.float32)
        # Pack RGB values
        rgb_ints = np.floor(o3d_colors * 255).astype(np.uint32)
        rgb_packed = np.left_shift(rgb_ints[:, 0], 16) | np.left_shift(rgb_ints[:, 1], 8) | rgb_ints[:, 2]

        # Combine XYZ and RGB into one structured array
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)]
        structured_array = np.empty(len(o3d_points), dtype=dtype)
        structured_array['x'] = o3d_points[:, 0]
        structured_array['y'] = o3d_points[:, 1]
        structured_array['z'] = o3d_points[:, 2]
        structured_array['rgb'] = rgb_packed

        cloud_msg = ros_numpy.msgify(PointCloud2, structured_array)
        cloud_msg.header = header

    return cloud_msg


def from_msg(ros_cloud):
    xyzrgb_array = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)

    mask = np.isfinite(xyzrgb_array['x']) & np.isfinite(xyzrgb_array['y']) & np.isfinite(xyzrgb_array['z'])
    cloud_array = xyzrgb_array[mask]

    open3d_cloud = open3d.geometry.PointCloud()

    points = np.zeros(cloud_array.shape + (3,), dtype=np.float32)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    open3d_cloud.points = open3d.utility.Vector3dVector(points)

    if 'rgb' in xyzrgb_array.dtype.names:
        rgb_array = ros_numpy.point_cloud2.split_rgb_field(xyzrgb_array)
        cloud_array = rgb_array[mask]

        colors = np.zeros(cloud_array.shape + (3,), dtype=np.float32)
        colors[..., 0] = cloud_array['r']
        colors[..., 1] = cloud_array['g']
        colors[..., 2] = cloud_array['b']

        open3d_cloud.colors = open3d.utility.Vector3dVector(colors / 255.0)

    return open3d_cloud