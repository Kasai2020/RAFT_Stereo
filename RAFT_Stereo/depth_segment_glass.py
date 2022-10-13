import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import cv2
import json
from torchvision import transforms
import pyrealsense2 as rs
from transform_to_color import transform_depth
from html_vis import visualize_helper
from shutil import rmtree

# Bring your packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Segment_Transparent_Objects')))

# Now do your import
from tools.inference import *

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        # load camera matrix
        camera_k = json.load(open("real_intrinsics"))

        # load glass segmenter
        class Namespace():
            config_file='../Segment_Transparent_Objects/configs/trans10K/translab.yaml', 
            input_img='tools/demo_vis.png', 
            local_rank=0, 
            log_iter=10, 
            no_cuda=False, 
            opts=['TEST.TEST_MODEL_PATH', './../Segment_Transparent_Objects/demo/16.pth', 'DEMO_DIR', './demo/imgs'], 
            resume=None, 
            skip_val=False, 
            val_epoch=1

        args2 = Namespace()

        cfg.update_from_file(args2.config_file[0])
        cfg.update_from_list(args2.opts[0])
        cfg.PHASE = 'test'
        cfg.ROOT_PATH = root_path
        cfg.DATASET.NAME = 'trans10k_extra'
        cfg.check_and_freeze()

        default_setup(args2)
        evaluator = Evaluator(args2)


        dump_dir = Path("vis_dir/")
        if dump_dir.exists():
            rmtree(dump_dir)
        dump_dir.mkdir()
        dump_paths = list()

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            file_stem = imfile1.split('/')[-2]

            disp = flow_up.cpu().numpy().squeeze()

            # crop disp
            disp = disp[8:-8]

            # convert to depth
            focal_length = float(camera_k["rectified.1.fx"])
            baseline = float(camera_k["baseline"])
            disparity = disp
            cx1 = float(camera_k["rectified.1.ppx"])
            cx0 = float(camera_k["rectified.1.ppx"])

            depth = (focal_length * (-1 * baseline)) / np.abs(disparity + (cx1 - cx0))

            # load rgb
            rgb = cv2.imread(imfile1[:-17] + "_on_color.png")

            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        
            rgb = input_transform(rgb)
            rgb = torch.unsqueeze(rgb, 0)

            seg = evaluator.eval(rgb)
            seg[seg == 254] = 0
            print(np.unique(seg))

            rgb = cv2.imread(imfile1[:-17] + "_on_color.png")
            rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

            # f, axarr = plt.subplots(2,1) 
            # axarr[0].imshow(rgb)
            # axarr[1].imshow(seg)
            # plt.show()

            ### Alexis' Realsense
            # K_depth = np.asarray([[640.409362792969, 0, 650.067993164062], [0, 640.409362792969, 350.878326416016], [0, 0, 1]])
            # K_col = np.asarray([[911.6954956054688, 0, 643.8764038085938], [0, 911.9517822265625, 369.8354797363281], [0, 0, 1]])
            # ext = [[0.999891,         0.014733,        -0.00102833, 0.0145771522074938],
            #     [-0.0147294,        0.999886,         0.00345947, -2.24726027227007e-05],  
            #     [0.00107918,      -0.00344394,       0.999994, 0.000409498723456636],
            #     [0, 0, 0, 1]]

            ### Gen3's Realsense
            K_depth = np.asarray([[644.642578125, 0, 635.8193359375], [0, 644.642578125, 373.865753173828], [0, 0, 1]])
            K_col = np.asarray([[924.177612304688, 0, 653.009582519531], [0, 924.155700683594, 364.164855957031], [0, 0, 1]])
            ext = [[0.999952,        -0.00975193,      -4.51748e-05, 0.0148850595578551], 
                [0.00975197,       0.999952,         0.000933617, 9.47166117839515e-05],
                [3.60681e-05,     -0.000934013,      1, 0.000182549862074666],
                [0, 0, 0, 1]]

            depth = transform_depth(K_depth, K_col, ext, depth, rgb)

            depth_deep_vis = np.copy(depth * 3.85)

            depth[seg != 127] = 0

            depth = depth * 3.85

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
            # o3d.visualization.draw([o3d_pcd])
            o3d.io.write_point_cloud(f"pcds/fragment_{(Path(imfile1).parts[-2])}.ply", o3d_pcd)

            # create point cloud
            o3d_rgb = o3d.geometry.Image(rgb)
            o3d_depth = o3d.geometry.Image(depth_deep_vis.astype(np.float32))
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
            # o3d.visualization.draw([o3d_pcd])
            o3d.io.write_point_cloud(f"pcds/full_deep_{(Path(imfile1).parts[-2])}.ply", o3d_pcd)



            # load rgb
            # rgb = cv2.imread(imfile1[:-17] + "_on_color.png")
            # load depth
            depth = np.load(imfile1[:-17] + "_on_depth.png.npy")
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
            o3d.io.write_point_cloud(f"pcds/full_real_{(Path(imfile1).parts[-2])}.ply", o3d_pcd)


            
            f, axarr = plt.subplots(2,2) 
            axarr[0,0].imshow(rgb)
            axarr[1,0].imshow(seg)
            axarr[0,1].imshow(depth)
            axarr[1,1].imshow(depth_deep_vis)
            # plt.show()
            plt.savefig(f"pcds/img_{(Path(imfile1).parts[-2])}.png")


            datapoint_vis_path = Path(dump_dir) / Path(Path(imfile1).parts[-2])
            datapoint_vis_path.mkdir()
            dump_path = dict()

            plt.imsave(datapoint_vis_path / "rgb.png", rgb)
            dump_path["rgb"] = datapoint_vis_path / "rgb.png"

            seg = (((seg / 127))).astype('uint8')
            arr3D = np.repeat(seg[...,None],3,axis=2)

            rgb = rgb * arr3D

            plt.imsave(datapoint_vis_path / "seg.png", rgb)
            dump_path["seg"] = datapoint_vis_path / "seg.png"

            plt.imsave(datapoint_vis_path / "depth.png", depth)
            dump_path["depth"] = datapoint_vis_path / "depth.png"

            plt.imsave(datapoint_vis_path / "deep_depth.png", depth_deep_vis)
            dump_path["deep_depth"] = datapoint_vis_path / "deep_depth.png"

            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            dump_paths.append(dump_path)
        visualize_helper(dump_paths, dump_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
