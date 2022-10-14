

import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import os
from RAFT_Stereo.visualization import VisOpen3D

def transform_depth(w, h, K_depth, K_col, ext, depth, color):
    # project into 3D
    o3d_rgb = o3d.geometry.Image(color)
    o3d_depth = o3d.geometry.Image(depth)
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, convert_rgb_to_intensity=False
    )
    o3d_camera_k = o3d.camera.PinholeCameraIntrinsic(
        w,
        h,
        K_depth[0,0],
        K_depth[1,1],
        K_depth[0,2],
        K_depth[1,2],
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, o3d_camera_k)

    pcl = o3d_pcd.transform(ext)

    window_visible = False
    vis = VisOpen3D(width=w, height=h, visible=window_visible)

    # point cloud
    vis.add_geometry(pcl)

    # update view
    vis.update_view_point(K_col, np.eye(4))

    # capture images
    depth_full = vis.capture_depth_float_buffer()
    vis.destroy_window()
    del vis

    depth_vis = (depth / 1000.0)
    depth_full_vis = np.asarray(depth_full)
    depth_vis[depth_vis > 1.0] = 0
    depth_full_vis[depth_full_vis > 1.0] = 0
    depth_vis[depth_vis < 0.0] = 0
    depth_full_vis[depth_full_vis < 0.0] = 0

    depth_full_vis *= 255

    depth_vis *= 255
    return np.asarray(depth_full)
