# Code is adapted from Monodepth2:
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset
import torch


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.load_calibrations()

    def load_calibrations(self):
        sequences = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
        self.calibrations = {}
        for seq in sequences:
            data = load_calib_cam_to_cam(os.path.join(self.data_path, seq))
            intrinsic_scaled = data[f"K_cam2"].copy()
            intrinsic_scaled[0] *= self.width / data["im_size_2"][0]
            intrinsic_scaled[1] *= self.height / data["im_size_2"][1]
            self.calibrations[seq] = {
                "intrinsic": data[f"K_cam2"],
                "intrinsic_scaled": intrinsic_scaled,
                "cam_to_world": np.linalg.inv(data[f"T_velo_to_cam2"]),
                "HW": data["im_size_2"][::-1],
            }

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        return self.loader(self.get_image_path(folder, frame_index, side))

    def get_intrinsic(self, folder):
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = self.calibrations[folder]["intrinsic"]
        return intrinsic

    def get_ground(self, folder, flip, transform, out_intrinsic):
        c = self.calibrations[folder]
        cam_to_world = c["cam_to_world"]

        if flip:
            # recover unflipped intrinsics
            out_intrinsic = out_intrinsic.copy()
            out_intrinsic[0, 2] = self.width - out_intrinsic[0, 2]

        if transform is None:
            transform = np.eye(4)
            transform[:3, :3] = c["intrinsic"] @ np.linalg.inv(c["intrinsic_scaled"])

        else:
            intrinsic = np.eye(4)
            intrinsic[:3, :3] = c["intrinsic"]
            transform = intrinsic @ transform.numpy() @ np.linalg.inv(out_intrinsic)

        w, h = self.width, self.height

        cam_to_world = cam_to_world @ transform
        cam_to_world[2, 3] += 1.65
        u, v = np.meshgrid(range(w), range(h), indexing="xy")
        ground = -cam_to_world[2, 3] / (cam_to_world[2, 0] * u + cam_to_world[2, 1] * v + cam_to_world[2, 2])

        if flip:
            ground = np.fliplr(ground)

        ground[ground < 0] = 0
        ground = ground[None]
        ground = torch.from_numpy(ground.astype(np.float32))

        return ground, cam_to_world[2, 3], torch.tensor(cam_to_world).float()


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt,
            self.full_res_shape[::-1],
            order=0,
            preserve_range=True,
            mode="constant",
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps"""

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str,
        )

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


### utils functions ###


def load_calib_cam_to_cam(path):
    # We'll return the camera calibration as a dictionary
    data = {}
    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(os.path.join(path, "calib_cam_to_cam.txt"))

    names = ["P_rect_00", "P_rect_01", "P_rect_02", "P_rect_03"]
    if "P0" in filedata:
        names = ["P0", "P1", "P2", "P3"]

    # Create 3x4 projection matrices
    p_rect = [np.reshape(filedata[p], (3, 4)) for p in names]

    for i, p in enumerate(p_rect):
        data[f"P_rect_{i}0"] = p

    # Get image sizes

    for i in range(4):
        data[f"im_size_{i}"] = filedata[f"S_rect_0{i}"]

    # Compute the rectified extrinsics from cam0 to camN
    rectified_extrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        rectified_extrinsics[i][0, 3] = p_rect[i][0, 3] / p_rect[i][0, 0]
        data[f"T_cam{i}_rect"] = rectified_extrinsics[i]

        # Compute the camera intrinsics
        data[f"K_cam{i}"] = p_rect[i][0:3, 0:3]

    # Create 4x4 matrices from the rectifying rotation matrices
    r_rect = None
    if "R_rect_00" in filedata:
        r_rect = [np.eye(4) for _ in range(4)]
        for i in range(4):
            r_rect[i][0:3, 0:3] = np.reshape(filedata["R_rect_0" + str(i)], (3, 3))
            data[f"R_rect_{i}0"] = r_rect[i]

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates

    t_cam0unrect_velo = load_calib_rigid(os.path.join(path, "calib_velo_to_cam.txt"))

    intrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        intrinsics[i][:3] = p_rect[i]

    velo_to_cam = [intrinsics[i].dot(r_rect[0].dot(t_cam0unrect_velo)) for i in range(4)]

    for i in range(4):
        data[f"T_velo_to_cam{i}"] = velo_to_cam[i]

    return data


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_calib_rigid(filepath):
    """Read a rigid transform calibration file as a numpy.array."""
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data["R"], data["T"])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(" "))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices"""
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data"""
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    velo2cam = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
    velo2cam = np.hstack((velo2cam["R"].reshape(3, 3), velo2cam["T"][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam["P_rect_0" + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
