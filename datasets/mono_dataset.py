# Code is adapted from Monodepth2:
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils import transform_from_angles


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
        ground
    """

    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        frame_idxs,
        num_scales,
        is_train=False,
        img_ext=".jpg",
        ground=False,
        angles_aug=[0, 0, 0],
        load_depth=False,
        tau=0.25,
    ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.ground = ground
        self.load_depth = load_depth

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s))

        self.do_angle_augs = angles_aug != [0, 0, 0]
        self.load_intrinsic = self.do_angle_augs
        self.angles = angles_aug
        self.tau = tau

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        if self.do_angle_augs:
            if self.is_train:
                angles = self.angles.copy()
                angles[1] = 0
                rand = random.random()
                if rand < (1 / 3):
                    angles[0] *= random.random() * 2 - 1
                    angles[1] = 0
                    angles[2] = 0
                elif rand < (2 / 3):
                    angles[0] = 0
                    angles[1] *= random.random() * 2 - 1
                    angles[2] = 0
                else:
                    angles[0] = 0
                    angles[1] = 0
                    angles[2] *= random.random() * 2 - 1

            else:
                angles = self.angles

            inputs["angles"] = torch.tensor(angles, dtype=torch.float32)
        else:
            inputs["angles"] = torch.zeros(3, dtype=torch.float32)
            angles = None

        inputs["tau"] = self.tau

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        if self.load_intrinsic:
            K0 = self.get_intrinsic(folder.split("/")[0])
        else:
            K0 = self.K.copy()
            K0[0, :] *= self.width
            K0[1, :] *= self.height

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x: x

        transform = None
        crop = np.zeros((len(self.frame_idxs), 2), dtype=np.int64)
        if angles is not None:
            transform = transform_from_angles(angles)
            KTK_inv = K0 @ torch.linalg.inv(transform).numpy() @ np.linalg.inv(K0)
            inputs["KTK_inv"] = KTK_inv
            for i in self.frame_idxs:
                inputs[("color", i, -1)], crop[i] = self.warp_image(inputs[("color", i, -1)], KTK_inv)
                # resize K to self.width, self.height using params
                inputs[("crop", i)] = crop[i]
                if do_flip:
                    inputs[("color", i, -1)] = TF.hflip(inputs[("color", i, -1)])

            # update intrinsics
            K0[:2, 2] -= crop[0]
            size = inputs[("color", 0, -1)].size
            K0[0, :] *= self.width / size[0]
            K0[1, :] *= self.height / size[1]
            if do_flip:
                K0[0, 2] = self.width - K0[0, 2]

        elif do_flip:
            for i in self.frame_idxs:
                inputs[("color", i, -1)] = inputs[("color", i, -1)].transpose(Image.FLIP_LEFT_RIGHT)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = K0.copy()
            K[0, :] /= 2**scale
            K[1, :] /= 2**scale
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if angles is not None:
            inputs["angles"] = torch.from_numpy(np.array(angles).astype(np.float32))

        if self.ground:
            inputs["ground"], inputs["height"], inputs["A"] = self.get_ground(
                folder.split("/")[0], do_flip, transform, K0
            )
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def warp_image(self, image, KTK_inv):
        W, H = image.size
        # image corners
        p = torch.tensor([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], dtype=torch.float32)
        # synthetic rotation of camera
        points = p @ KTK_inv[:3, :3].T
        points = points[:, :2] / points[:, 2:]

        warped = TF.perspective(image, p[:, :2].cpu().tolist(), points.cpu().tolist())
        # crop the image to remove black areas and keep principal point in the center
        crop_x = torch.maximum(points[::2, 0].max(), W - points[1::2, 0].min()).clip(0)
        crop_x = crop_x.round().long()
        crop_y = torch.maximum(points[:2, 1].max(), H - points[2:, 1].min()).clip(0)
        crop_y = crop_y.round().long()

        # crop so that the image respect the output image ratio (symetrically)
        size = torch.tensor([W - 2 * crop_x, H - 2 * crop_y])  # W, H
        out_ratio = self.height / self.width
        in_ratio = size[1] / size[0]
        if in_ratio > out_ratio:
            # crop H
            hh = (size[1] - size[0] * out_ratio) / 2
            crop_y += hh.long()
        elif in_ratio < out_ratio:
            # crop W
            wh = (size[0] - size[1] / out_ratio) / 2
            crop_x += wh.long()

        crop_x, crop_y = int(crop_x), int(crop_y)
        warped = TF.crop(warped, crop_y, crop_x, H - 2 * crop_y, W - 2 * crop_x)
        return warped, np.array([crop_x, crop_y], dtype=np.int64)
