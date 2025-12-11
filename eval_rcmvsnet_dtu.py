import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.casmvsnet import *
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
# from gipuma import gipuma_filter
from torchvision import transforms
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import signal
import logging

from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_test', help='select dataset')
parser.add_argument('--testpath', default="/cluster/51/dichang/datasets/mvsnet/dtu_test", help='test datapath')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default='./lists/dtu/test.txt', help='test list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--loadckpt', default='./pretrain/model_000014_cas.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./dtu_exp', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=1200, help='testing max h')
parser.add_argument('--max_w', type=int, default=1600, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

# filter
parser.add_argument('--prob_thres', type=float, default=0.8, help='prob confidence')
parser.add_argument('--num_consistency', type=int, default=3, help='threshold of num view')
parser.add_argument('--img_dist_thres', type=float, default=0.5, help='hyper1')
parser.add_argument('--depth_thres', type=float, default=0.01, help='hyper2')

parser.add_argument('--no_test', action='store_true', help='fusion without testing')
parser.add_argument('--no_filter', action='store_true', help='testing without fusion')
# os environment
parser.add_argument('--true_gpu', default="1", help='using true gpu')
parser.add_argument('--stage', default=False, help='train stage by stage')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')

# parse arguments and check
args = parser.parse_args()
# print("argv:", sys.argv[1:])
# print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale


# print("***********Interval_Scale**********\n", Interval_Scale)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def save_depth(testlist):
    for scene in testlist:
        save_scene_depth([scene])

    # save_scene_depth(testlist)


# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist: List[str]) -> None:
    """
    Estimate depth from multi-view images and save results.

    This function performs:
        1. Load model and dataset
        2. Run depth estimation inference
        3. Save depth maps, confidence maps, camera parameters, and point clouds

    Args:
        testlist: List of test scenes to process
    """
    log.info("=" * 60)
    log.info("Starting MVS depth estimation pipeline")
    log.info("=" * 60)

    # ==================== Configuration ====================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    stage_downsample = {1: 0.25, 2: 0.5, 3: 1.0}

    output_dir = Path(args.outdir)

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(norm_mean, norm_std)],
        std=[1 / s for s in norm_std]
    )

    # ==================== Load Dataset ====================
    log.info("Dataset path: %s", args.testpath)

    MVSDataset = find_dataset_def(args.dataset)
    dataset = MVSDataset(
        args.testpath, testlist, "test", args.num_view, args.numdepth, Interval_Scale,
        max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res
    )
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    log.info("Dataset loaded: %d samples, %d batches", len(dataset), len(dataloader))

    # ==================== Load Model ====================
    log.info("Loading model from: %s", args.loadckpt)

    model = CascadeMVSNet_eval(
        refine=False,
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d) for d in args.depth_inter_r.split(",") if d],
        share_cr=args.share_cr,
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
        grad_method=args.grad_method
    )

    state_dict = torch.load(args.loadckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    log.info("Model loaded - ndepths: %s, depth_inter_r: %s", args.ndepths, args.depth_inter_r)

    # ==================== Run Inference ====================
    log.info("-" * 60)
    log.info("Starting inference...")
    log.info("-" * 60)

    total_batches = len(dataloader)
    total_width = len(str(total_batches))
    total_time = 0.0

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            batch_start = time.time()

            # Inference
            sample_cuda = tocuda(sample)
            outputs = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"]
            )
            inference_time = time.time() - batch_start
            total_time += inference_time

            # Convert to numpy and free GPU memory
            outputs = tensor2numpy(outputs)
            del sample_cuda

            # Get metadata
            depth_values = sample["depth_values"]
            depth_range = (depth_values[0][0].item(), depth_values[0][-1].item())
            stage_key = f"stage{num_stage}"
            cams = sample["proj_matrices"][stage_key].numpy()

            # Process each sample in batch
            for idx, filename in enumerate(sample["filename"]):
                # Generate output paths
                paths = {
                    "depth": output_dir / filename.format("depth_est", ".pfm"),
                    "depth_img": output_dir / filename.format("depth_map", ".jpg"),
                    "confidence": output_dir / filename.format("confidence", ".pfm"),
                    "confidence_img": output_dir / filename.format("confidence_map", ".jpg"),
                    "cam": output_dir / filename.format("cams", "_cam.txt"),
                    "image": output_dir / filename.format("images", ".jpg"),
                    # "ply": output_dir / filename.format("ply_local", ".ply"),
                }
                if idx == 0:
                    for path in paths.values():
                        path.parent.mkdir(parents=True, exist_ok=True)

                # Get sample data
                img_tensor = sample["imgs"][idx][0]  # Reference view
                cam = cams[idx][0]  # Reference camera
                depth = outputs["depth"][idx]
                confidence = outputs["photometric_confidence"][idx]

                # Restore image
                img = inv_normalize(img_tensor).numpy()
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

                # Save depth map
                save_pfm(str(paths["depth"]), depth)
                plt.imsave(str(paths["depth_img"]), depth, cmap="rainbow",
                           vmin=depth_range[0], vmax=depth_range[1])

                # Save confidence map
                save_pfm(str(paths["confidence"]), confidence)
                plt.imsave(str(paths["confidence_img"]), confidence, cmap="rainbow",
                           vmin=confidence.min(), vmax=confidence.max())

                # Save camera and image
                write_cam(str(paths["cam"]), cam)
                cv2.imwrite(str(paths["image"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # # Generate point cloud (last batch only)
                # if batch_idx == total_batches - 1:
                #     ratio = stage_downsample.get(num_stage, 1.0)
                #     if ratio < 1.0:
                #         downsample_img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
                #     else:
                #         downsample_img = img
                #
                #     generate_pointcloud(downsample_img, depth, str(paths["ply"]), cam[1, :3, :3])
                #     log.info("Generated point cloud: %s", paths["ply"])

            # Log progress
            avg_time = total_time / (batch_idx + 1)
            remaining = avg_time * (total_batches - batch_idx - 1)
            log.info(
                "Batch [%s/%d] | Time: %6.2fs | Avg: %6.2fs | ETA: %6.1fs",
                f"{batch_idx + 1:0{total_width}d}", total_batches,
                inference_time, avg_time, remaining
            )

    # ==================== Cleanup ====================
    torch.cuda.empty_cache()
    gc.collect()

    log.info("Depth estimation completed")
    log.info("Total time: %.2fs | Average: %.2fs/batch", total_time, total_time / total_batches)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                img_dist_thresh, depth_thresh):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < img_dist_thresh, relative_depth_diff < depth_thresh)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def _display_results(
        ref_img: np.ndarray,
        ref_depth: np.ndarray,
        photo_mask: np.ndarray,
        geo_mask: np.ndarray,
        final_mask: np.ndarray
) -> None:
    """Display intermediate results using OpenCV."""
    import cv2

    depth_scale = ref_depth.max() if ref_depth.max() > 0 else 1

    cv2.imshow('ref_img', ref_img[:, :, ::-1])
    cv2.imshow('ref_depth', ref_depth / depth_scale)
    cv2.imshow('ref_depth * photo_mask', ref_depth * photo_mask.astype(np.float32) / depth_scale)
    cv2.imshow('ref_depth * geo_mask', ref_depth * geo_mask.astype(np.float32) / depth_scale)
    cv2.imshow('ref_depth * final_mask', ref_depth * final_mask.astype(np.float32) / depth_scale)
    cv2.waitKey(0)


def _get_colors(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Get colors for valid points based on stage downsampling."""
    if num_stage == 1:
        return img[1::4, 1::4, :][mask]
    elif num_stage == 2:
        return img[1::2, 1::2, :][mask]
    else:
        return img[mask]


def _save_ply(filename: str, vertices: np.ndarray, colors: np.ndarray) -> None:
    """Save point cloud to PLY file."""
    log.info("Saving PLY file: %s", filename)

    # Create structured arrays
    vertices_structured: np.ndarray = np.array(
        [tuple(v) for v in vertices],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    colors_structured: np.ndarray = np.array(
        [tuple(c) for c in colors],
        dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )

    # Merge into single structured array
    vertex_all: np.ndarray = np.empty(
        len(vertices),
        dtype=vertices_structured.dtype.descr + colors_structured.dtype.descr
    )

    for prop in vertices_structured.dtype.names:
        vertex_all[prop] = vertices_structured[prop]
    for prop in colors_structured.dtype.names:
        vertex_all[prop] = colors_structured[prop]

    # Write PLY
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(filename)

    log.info("Saved %d points to PLY file", len(vertices))


def filter_depth(
        pair_folder: str,
        scan_folder: str,
        out_folder: str,
        ply_filename: str,
        prob_threshold: float = 0.8,
        num_consistent: int = 3,
        img_dist_thresh: float = 1.0,
        depth_thresh: float = 0.01,
        display: bool = False
) -> str:
    """
    Filter and fuse multi-view depth maps into a single point cloud.

    This function performs:
        1. Photometric consistency filtering (confidence threshold)
        2. Geometric consistency filtering (multi-view depth agreement)
        3. Depth fusion and reprojection to 3D world coordinates

    Args:
        pair_folder: Folder containing pair.txt
        scan_folder: Folder containing camera parameters and images
        out_folder: Folder containing depth estimation results
        ply_filename: Output PLY file path
        prob_threshold: Confidence threshold for photometric filtering
        num_consistent: Minimum number of consistent source views
        img_dist_thresh: Pixel distance threshold for geometric consistency
        depth_thresh: Depth difference threshold for geometric consistency
        display: Whether to display intermediate results

    Returns:
        Path to the saved PLY file
    """
    log.info("-" * 60)
    log.info("Starting depth filtering and fusion")
    log.info("-" * 60)

    # Setup paths
    scan_folder = Path(scan_folder)
    out_folder = Path(out_folder)
    pair_file = Path(pair_folder) / "pair.txt"

    # Create output directories
    mask_folder = out_folder / "mask"
    filtered_depth_folder = out_folder / "filtered_depth"
    mask_folder.mkdir(parents=True, exist_ok=True)
    filtered_depth_folder.mkdir(parents=True, exist_ok=True)

    # Read pair file
    pair_data = read_pair_file(str(pair_file))
    num_views = len(pair_data)
    log.info("Loaded %d reference views from pair file", num_views)

    # Accumulators for final point cloud
    all_vertices = []
    all_colors = []

    # Statistics
    total_points = 0
    start_time = time.time()

    for idx, (ref_view, src_views) in enumerate(pair_data):
        view_start = time.time()

        cam_file = scan_folder / f"cams/{ref_view:08d}_cam.txt"
        img_file = scan_folder / f"images/{ref_view:08d}.jpg"
        depth_file = out_folder / f"depth_est/{ref_view:08d}.pfm"
        conf_file = out_folder / f"confidence/{ref_view:08d}.pfm"

        # Load reference view data
        ref_intrinsics, ref_extrinsics = read_camera_parameters(cam_file)
        ref_img = read_img(img_file)
        ref_depth = read_pfm(depth_file)[0]
        ref_confidence = read_pfm(conf_file)[0]

        # Step 1: Photometric consistency mask
        photo_mask = ref_confidence > prob_threshold

        # Step 2: Geometric consistency mask
        geo_mask_sum = np.zeros_like(ref_depth, dtype=np.int32)
        depth_reprojected_sum = np.zeros_like(ref_depth, dtype=np.float32)

        for src_view in src_views:
            src_cam_file = scan_folder / f"cams/{src_view:08d}_cam.txt"
            src_depth_file = out_folder / f"depth_est/{src_view:08d}.pfm"

            src_intrinsics, src_extrinsics = read_camera_parameters(src_cam_file)
            src_depth = read_pfm(src_depth_file)[0]

            geo_mask, depth_reprojected, _, _ = check_geometric_consistency(
                ref_depth, ref_intrinsics, ref_extrinsics,
                src_depth, src_intrinsics, src_extrinsics,
                img_dist_thresh=img_dist_thresh,
                depth_thresh=depth_thresh
            )

            geo_mask_sum += geo_mask.astype(np.int32)
            depth_reprojected_sum += depth_reprojected

        # Fused depth (average of consistent views)
        depth_fused = (depth_reprojected_sum + ref_depth) / (geo_mask_sum + 1)

        # At least num_consistent source views must agree
        geo_mask: np.ndarray = geo_mask_sum >= num_consistent

        # Step 3: Combined mask
        final_mask = photo_mask & geo_mask

        # Save masks and filtered depth
        save_mask(str(mask_folder / f"{ref_view:08d}_photo.png"), photo_mask)
        save_mask(str(mask_folder / f"{ref_view:08d}_geo.png"), geo_mask)
        save_mask(str(mask_folder / f"{ref_view:08d}_final.png"), final_mask)

        filtered_depth = ref_depth * final_mask.astype(np.float32)
        plt.imsave(str(filtered_depth_folder / f"{ref_view:08d}.jpg"), filtered_depth, cmap="rainbow")

        # Display if requested
        if display:
            _display_results(ref_img, ref_depth, photo_mask, geo_mask, final_mask)

        # Step 4: Reproject to 3D world coordinates
        height, width = depth_fused.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        valid_points = final_mask
        x_valid = x[valid_points]
        y_valid = y[valid_points]
        depth_valid = depth_fused[valid_points]

        # Get colors based on stage
        color_valid = _get_colors(ref_img, valid_points)

        # 2D + depth -> 3D camera coordinates
        xyz_cam = np.linalg.inv(ref_intrinsics) @ np.vstack([
            x_valid, y_valid, np.ones_like(x_valid)
        ]) * depth_valid

        # 3D camera -> 3D world coordinates
        xyz_world = np.linalg.inv(ref_extrinsics) @ np.vstack([
            xyz_cam, np.ones_like(x_valid)
        ])
        xyz_world = xyz_world[:3].T

        all_vertices.append(xyz_world)
        all_colors.append((color_valid * 255).astype(np.uint8))

        # Statistics
        num_valid = valid_points.sum()
        total_points += num_valid

        # Progress log
        view_time = time.time() - view_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        eta = avg_time * (num_views - idx - 1)

        log.info(
            "View [%02d/%02d] | Photo: %6.2f%% | Geo: %6.2f%% | Final: %6.2f%% | "
            "Points: %6d | Time: %5.2fs",
            idx + 1, num_views,
            photo_mask.mean() * 100,
            geo_mask.mean() * 100,
            final_mask.mean() * 100,
            num_valid,
            view_time
        )

    # Merge all vertices and colors
    log.info("Merging point cloud...")

    vertices = np.concatenate(all_vertices, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    log.info("Total points: %d", len(vertices))

    # Save PLY file
    _save_ply(ply_filename, vertices, colors)

    # Final statistics
    total_time = time.time() - start_time
    log.info("Depth filtering completed")
    log.info("Total time: %.2fs (%.2fs/view)", total_time, total_time / num_views)

    return ply_filename


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    scan_id = 1
    if args.testlist != "all":
        scan_id = int(scan[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    img_dist_thres = {
        '1': 0.5,
        '4': 0.5,
        '9': 0.5,
        '10': 0.25,
        '11': 0.75,
        '12': 0.25,
        '13': 0.75,
        '15': 0.5,
        '23': 0.5,
        '24': 0.5,
        '29': 0.5,
        '32': 0.5,
        '33': 0.5,
        '34': 0.25,
        '48': 0.75,
        '49': 0.5,
        '62': 0.5,
        '75': 0.25,
        '77': 0.25,
        '110': 0.25,
        '114': 0.5,
        '118': 0.75,
    }
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name),
                 prob_threshold=args.prob_thres, num_consistent=args.num_consistency,
                 img_dist_thresh=img_dist_thres[str(scan_id)], depth_thresh=args.depth_thres)


def pcd_filter(testlist, number_worker):
    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.true_gpu
    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        # for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    if not args.no_test:
        save_depth(testlist)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    if not args.no_filter:
        pcd_filter(testlist, args.num_worker)
