
import open3d as o3d
import numpy as np
import cv2
import json

for i in range(0,10):
    # load camera matrix
    camera_k = json.load(open("real_intrinsics"))

    # load disparity
    disp = np.load(f"output_midd/{i}.npy")
    disp = disp[8:-8]

    # convert to depth
    focal_length = float(camera_k["rectified.1.fx"])
    baseline = float(camera_k["baseline"])
    disparity = disp
    cx1 = float(camera_k["rectified.1.ppx"])
    cx0 = float(camera_k["rectified.1.ppx"])

    print(np.shape(disparity))
    print(focal_length)
    print(baseline)
    print(cx1)
    print(cx0)

    depth = (focal_length * (-1 * baseline)) / np.abs(disparity + (cx1 - cx0))

    print(np.unique(depth))

    # load rgb
    rgb = cv2.imread(f"datasets/realsense_infrared_test_data/{i}/laser_off_camera_1.png")
    print(np.shape(rgb))

    # create point cloud
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1000, convert_rgb_to_intensity=False
    )  # depth from realsense comes in mm
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image( 
        o3d_rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width = depth.shape[1],
            height = depth.shape[0],
            fx = float(camera_k["rectified.1.fx"]),
            fy = float(camera_k["rectified.1.fy"]),
            cx = float(camera_k["rectified.1.ppx"]),
            cy = float(camera_k["rectified.1.ppy"]),
        )
    )
    o3d.visualization.draw([o3d_pcd])