#!/usr/bin/env python
"""
Copyright 2019, Jingyang Zhang and Yao Yao, HKUST. Model reading is provided by COLMAP.
Preprocess script.
"""

from __future__ import print_function

import collections
import struct
import numpy as np
import multiprocessing as mp
import os
import shutil
import cv2
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================ read_model.py ============================#
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def calc_score(inputs, index, images, extrinsic, points3d, theta0, sigma1, sigma2):
    """
    Calculate score for image pair (i, j) based on common 3D points and viewing angle
    Args:
        inputs (tuple): Tuple of (i, j)
        index (int): Index of current pair
        images (dict): Dictionary of images
        extrinsic (dict): Dictionary of extrinsic parameters
        points3d (dict): Dictionary of 3D points
        theta0 (float): Threshold angle for view selection
        sigma1 (float): Sigma for angle < theta0
        sigma2 (float): Sigma for angle >= theta0

    """
    i_id, j_id = inputs
    id_i = images[i_id].point3D_ids
    id_j = images[j_id].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]

    logger.debug("Calculating score for image pair (%d, %d) with %d common points",
                 i_id, j_id, len(id_intersect))

    cam_center_i = -np.matmul(extrinsic[i_id][:3, :3].transpose(), extrinsic[i_id][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j_id][:3, :3].transpose(), extrinsic[j_id][:3, 3:4])[:, 0]

    score = 0
    processed_points = 0

    for pid in id_intersect:
        if pid == -1:
            continue

        p = points3d[pid].xyz
        # Calculate angle between viewing rays
        vec_i = cam_center_i - p
        vec_j = cam_center_j - p

        # Normalize vectors
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)

        if norm_i == 0 or norm_j == 0:
            continue

        cos_angle = np.dot(vec_i, vec_j) / (norm_i * norm_j)
        # Clamp to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        theta = (180 / np.pi) * np.arccos(cos_angle)

        # Calculate score contribution based on angular difference
        sigma = sigma1 if theta <= theta0 else sigma2
        score_contribution = np.exp(-(theta - theta0) * (theta - theta0) / (2 * sigma ** 2))
        score += score_contribution
        processed_points += 1

    return i_id, j_id, score


def convert(
        dense_folder,
        output_folder=None,
        max_d=0,
        interval_scale=1,
        theta0=5,
        sigma1=1,
        sigma2=10,
        convert_format=False,
        num_processes=None
):
    """
    Convert COLMAP sparse reconstruction to MVSNet format

    Args:
        dense_folder (str): Path to dense folder containing images and sparse reconstruction
        output_folder (str, optional): Output folder path. If None, uses dense_folder
        max_d (int): Maximum depth levels
        interval_scale (float): Depth interval scale
        theta0 (float): Threshold angle for view selection
        sigma1 (float): Sigma for angle < theta0
        sigma2 (float): Sigma for angle >= theta0
        convert_format (bool): Whether to convert image format
        num_processes (int, optional): Number of processes for parallel computation.
                                     If None, uses 25% of CPU cores
    """
    # Use dense_folder as output_folder if not specified
    if output_folder is None:
        output_folder = dense_folder

    logger.info("Starting COLMAP to MVSNet conversion")
    logger.info("Input folder: %s", dense_folder)
    logger.info("Output folder: %s", output_folder)
    logger.info("Max depth levels: %d", max_d)
    logger.info("Interval scale: %f", interval_scale)
    logger.info("Theta0: %f", theta0)
    logger.info("Sigma1: %f, Sigma2: %f", sigma1, sigma2)
    logger.info("Convert format: %s", convert_format)
    logger.info("Number of processes: %s", num_processes)
    # Initialize paths
    image_dir = os.path.join(dense_folder, 'images')
    model_dir = os.path.join(dense_folder, 'sparse')
    cam_dir = os.path.join(output_folder, 'cams')
    renamed_dir = os.path.join(output_folder, 'images')

    logger.info("Image directory: %s", image_dir)
    logger.info("Model directory: %s", model_dir)
    logger.info("Camera directory: %s", cam_dir)
    logger.info("Renamed directory: %s", renamed_dir)

    # Read model
    logger.info("Reading COLMAP model...")
    try:
        cameras, images, points3d = read_model(model_dir, '.txt')
        logger.info("Successfully read model with %d cameras, %d images, %d points",
                    len(cameras), len(images), len(points3d))
    except Exception as e:
        logger.error("Failed to read model: %s", str(e))
        raise

    sorted_image_ids = sorted(images.keys())
    num_images = len(sorted_image_ids)
    logger.info("Processing %d images", num_images)

    # Create image_id to continuous index mapping
    id_to_idx = {image_id: idx for idx, image_id in enumerate(sorted_image_ids)}

    # Camera parameter types
    param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
    }

    intrinsic = {}
    for camera_id, cam in cameras.items():
        try:
            params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
            if 'f' in param_type[cam.model]:
                params_dict['fx'] = params_dict['f']
                params_dict['fy'] = params_dict['f']
            i = np.array([
                [params_dict['fx'], 0, params_dict['cx']],
                [0, params_dict['fy'], params_dict['cy']],
                [0, 0, 1]
            ])
            intrinsic[camera_id] = i
            logger.debug("Camera %d (%s): fx=%f, fy=%f, cx=%f, cy=%f",
                         camera_id, cam.model, params_dict['fx'], params_dict['fy'],
                         params_dict['cx'], params_dict['cy'])
        except Exception as e:
            logger.warning("Failed to process camera %d (%s): %s", camera_id, cam.model, str(e))

    logger.info("Calculated %d intrinsic matrices", len(intrinsic))

    extrinsic = {}
    for image_id, image in images.items():
        try:
            e = np.zeros((4, 4))
            e[:3, :3] = qvec2rotmat(image.qvec)
            e[:3, 3] = image.tvec
            e[3, 3] = 1
            extrinsic[image_id] = e
            logger.debug("Image %d: translation=[%f, %f, %f]",
                         image_id, image.tvec[0], image.tvec[1], image.tvec[2])
        except Exception as e:
            logger.warning("Failed to process image %d: %s", image_id, str(e))

    logger.info("Calculated %d extrinsic matrices", len(extrinsic))

    depth_ranges = {}
    for idx, image_id in enumerate(sorted_image_ids):
        try:
            image = images[image_id]
            zs = []
            for p3d_id in image.point3D_ids:
                if p3d_id == -1:
                    continue
                transformed = np.matmul(extrinsic[image_id],
                                        [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1],
                                         points3d[p3d_id].xyz[2], 1])
                zs.append(transformed[2].item())

            if len(zs) == 0:
                depth_min = 0.1
                depth_max = 100.0
                logger.debug("Image %d: No 3D points, using default depth range", image_id)
            else:
                zs_sorted = sorted(zs)
                depth_min = zs_sorted[int(len(zs) * 0.01)]
                depth_max = zs_sorted[int(len(zs) * 0.99)]
                logger.debug("Image %d: depth range [%f, %f] from %d points",
                             image_id, depth_min, depth_max, len(zs))

            if max_d == 0:
                image_int = intrinsic[image.camera_id]
                image_ext = extrinsic[image_id]
                image_r = image_ext[0:3, 0:3]
                image_t = image_ext[0:3, 3]
                p1 = [image_int[0, 2], image_int[1, 2], 1]
                p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
                P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
                P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
                P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
                P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
                depth_num = (1 / depth_min - 1 / depth_max) / (
                        1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
            else:
                depth_num = max_d
            depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale
            depth_ranges[image_id] = (depth_min, depth_interval, depth_num, depth_max)

            if idx % 10 == 0 or idx == num_images - 1:
                logger.info("Processed depth ranges for %d/%d images", idx + 1, num_images)

        except Exception as e:
            logger.warning("Failed to calculate depth range for image %d: %s", image_id, str(e))
            # Use default values
            depth_ranges[image_id] = (0.1, 0.5, 100, 50.0)

    logger.info("Completed depth range calculation for all images")

    # View selection
    logger.info("Performing view selection...")
    score = np.zeros((num_images, num_images))
    queue = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            queue.append((sorted_image_ids[i], sorted_image_ids[j]))

    queue_count = len(queue)
    logger.info("Computing scores for %d image pairs", queue_count)

    if num_processes is None:
        num_processes = max(int(mp.cpu_count() * 0.25), 1)

    # Parallel score computation
    logger.info("Starting parallel score computation with %d processes", num_processes)

    p = mp.Pool(processes=num_processes)
    result = p.starmap(
        calc_score,
        [(queue[index], index, images, extrinsic, points3d, theta0, sigma1, sigma2) for index in range(queue_count)]
    )
    p.close()
    p.join()

    logger.info("Completed parallel score computation")

    # Store scores
    logger.info("Storing computed scores...")
    for i_id, j_id, s in result:
        try:
            i_idx = id_to_idx[i_id]
            j_idx = id_to_idx[j_id]
            score[i_idx][j_idx] = s
            score[j_idx][i_idx] = s
        except Exception as e:
            logger.warning("Failed to store score for images %d-%d: %s", i_id, j_id, str(e))

    logger.info("Stored all scores")

    # Generate view selection list
    logger.info("Generating view selection list...")
    view_sel = []
    for i_idx in range(num_images):
        sorted_score = np.argsort(score[i_idx])[::-1]
        view_sel.append([(j_idx, score[i_idx][j_idx]) for j_idx in sorted_score[:10]])

        if i_idx % 10 == 0 or i_idx == num_images - 1:
            logger.info("Generated view selection for %d/%d images", i_idx + 1, num_images)

    logger.info("Completed view selection list generation")

    # Write camera parameter files
    logger.info("Writing camera parameter files to: %s", cam_dir)
    try:
        os.makedirs(cam_dir, exist_ok=True)
        logger.info("Created camera directory")
    except Exception as e:
        logger.error("Failed to create camera directory: %s", str(e))
        raise

    for idx, image_id in enumerate(sorted_image_ids):
        try:
            cam_file = os.path.join(cam_dir, f'{idx:08d}_cam.txt')
            with open(cam_file, 'w') as f:
                f.write('extrinsic\n')
                for j in range(4):
                    for k in range(4):
                        f.write(str(extrinsic[image_id][j, k]) + ' ')
                    f.write('\n')
                f.write('\nintrinsic\n')
                for j in range(3):
                    for k in range(3):
                        f.write(str(intrinsic[images[image_id].camera_id][j, k]) + ' ')
                    f.write('\n')
                f.write('\n%f %f %f %f\n' % (
                    depth_ranges[image_id][0], depth_ranges[image_id][1], depth_ranges[image_id][2],
                    depth_ranges[image_id][3]))

            if idx % 10 == 0 or idx == num_images - 1:
                logger.info("Written camera files for %d/%d images", idx + 1, num_images)

        except Exception as e:
            logger.warning("Failed to write camera file for image %d: %s", idx, str(e))

    logger.info("Completed writing camera parameter files")

    # Write pair.txt
    logger.info("Writing pair.txt...")
    try:
        pair_file = os.path.join(output_folder, 'pair.txt')
        with open(pair_file, 'w') as f:
            f.write(f'{num_images}\n')
            for i_idx in range(num_images):
                f.write(f'{i_idx}\n{len(view_sel[i_idx])} ')
                for j_idx, s in view_sel[i_idx]:
                    f.write(f'{j_idx} {s} ')
                f.write('\n')
        logger.info("Successfully wrote pair.txt with %d images", num_images)
    except Exception as e:
        logger.error("Failed to write pair.txt: %s", str(e))
        raise

    # Image renaming
    logger.info("Renaming and copying images...")
    try:
        os.makedirs(renamed_dir, exist_ok=True)
        logger.info("Created renamed images directory: %s", renamed_dir)
    except Exception as e:
        logger.error("Failed to create renamed images directory: %s", str(e))
        raise

    for idx, image_id in enumerate(sorted_image_ids):
        try:
            src = os.path.join(image_dir, images[image_id].name)
            dst = os.path.join(renamed_dir, f'{idx:08d}.jpg')

            if os.path.exists(src):
                if convert_format:
                    img = cv2.imread(src)
                    if img is not None:
                        cv2.imwrite(dst, img)
                        logger.debug("Converted and saved image %d: %s -> %s", idx, src, dst)
                    else:
                        logger.warning("Failed to read image: %s", src)
                else:
                    shutil.copyfile(src, dst)
                    logger.debug("Copied image %d: %s -> %s", idx, src, dst)
                os.remove(src)
            else:
                logger.warning("Source image not found: %s", src)

            if idx % 10 == 0 or idx == num_images - 1:
                logger.info("Processed images: %d/%d", idx + 1, num_images)

        except Exception as e:
            logger.warning("Failed to process image %d (%s): %s", idx, images[image_id].name, str(e))

    logger.info("Completed image processing")


def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP sparse reconstruction to MVSNet format')
    parser.add_argument('dense_folder', help='Path to dense folder containing images and sparse reconstruction')
    parser.add_argument('--output_folder', help='Output folder path (default: same as dense_folder)')
    parser.add_argument('--max_d', type=int, default=0, help='Maximum depth levels (default: 0)')
    parser.add_argument('--interval_scale', type=float, default=1.0, help='Depth interval scale (default: 1.0)')
    parser.add_argument('--theta0', type=float, default=5.0, help='Threshold angle for view selection (default: 5.0)')
    parser.add_argument('--sigma1', type=float, default=1.0, help='Sigma for angle < theta0 (default: 1.0)')
    parser.add_argument('--sigma2', type=float, default=10.0, help='Sigma for angle >= theta0 (default: 10.0)')
    parser.add_argument('--convert_format', action='store_true', help='Convert image format')
    parser.add_argument('--num_processes', type=int, help='Number of processes for parallel computation (default: 25%% of CPU cores)')

    args = parser.parse_args()

    convert(
        dense_folder=args.dense_folder,
        output_folder=args.output_folder,
        max_d=args.max_d,
        interval_scale=args.interval_scale,
        theta0=args.theta0,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        convert_format=args.convert_format,
        num_processes=args.num_processes
    )


if __name__ == '__main__':
    main()
