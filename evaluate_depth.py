from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import argparse
import datasets
import networks
from tqdm import trange, tqdm
import cv2


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
np.set_printoptions(precision=2, suppress=True)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def readlines(filename):
    """Read all the lines in a text file and return as a list"""
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return lines


def warp_depth(depth, KTK_inv, crop):
    depth = torch.tensor(depth)
    KTK_inv = torch.tensor(KTK_inv)
    H, W = depth.shape
    grid_in = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))).flip((0,))
    grid = torch.cat((grid_in, torch.ones((1, H, W)))).to(depth.device) * depth
    grid = torch.cat((grid, torch.ones((1, H, W), device=depth.device)))

    grid = KTK_inv @ grid.reshape(4, -1)
    grid[:2] = grid[:2] / grid[2]
    grid[grid.isnan()] = -1
    grid = grid[:3].reshape(3, H, W)

    warped_depth = torch.zeros(H, W, device=depth.device)
    valid = (grid[0] >= 0) & (grid[1] >= 0) & (grid[2] > 0) & (grid[0] < (W - 1)) & (grid[1] < (H - 1))
    warped_depth[torch.round(grid[1, valid]).long(), torch.round(grid[0, valid]).long()] = grid[2, valid]
    warped_depth = warped_depth[crop[1] : H - crop[1], crop[0] : W - crop[0]]
    return warped_depth


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1"""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def adaptive_ground(flat_ground, slope, h, softmax=False):
    if softmax:
        slope = torch.sum(
            torch.softmax(slope, dim=1) * torch.deg2rad(torch.arange(11, device="cuda") - 5).float().view(1, -1, 1, 1),
            dim=1,
            keepdim=True,
        )
    slope = torch.tan(slope)
    h = h.view(-1, 1, 1, 1).float().to(flat_ground.device)
    ground_inv = 1 / (flat_ground + 1e-8) + slope / h
    return ground_inv


def dynamic_depth(out, ground, opt):
    b, _, h, w = out.shape

    if opt.absolute:
        out = out * 2 - 1
    else:
        out = 2 * out / h
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, w, device="cuda", dtype=torch.float32),
            torch.linspace(-1, 1, h, device="cuda", dtype=torch.float32),
            indexing="xy",
        ),
        dim=2,
    )[None].repeat(b, 1, 1, 1)
    if opt.absolute:
        grid[:, :, :, 1] = out[:, 0]
    else:
        grid[:, :, :, 1] += out[:, 0]
    gd = ground.clone()
    gd[gd == 0] = opt.max_depth
    depth = torch.nn.functional.grid_sample(gd, grid.clip(-1, 1), padding_mode="reflection")
    return depth.clip(opt.min_depth, opt.max_depth)


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    use_rotation = opt.rotations != [0, 0, 0]
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(
        opt.data_path,
        filenames,
        opt.height,
        opt.width,
        [0],
        4,
        is_train=False,
        ground=True,
        img_ext=".jpg" if opt.jpg else ".png",
        angles_aug=opt.rotations,
    )
    dataloader = DataLoader(
        dataset,
        16,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    encoder = networks.ResnetEncoder(opt.num_layers, False, ground=True)

    depth_decoder = networks.DepthDecoder(
        encoder.num_ch_enc,
    )

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    preds = []
    ktk_inv = []
    crop = []

    print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            input_color = data[("color", 0, 0)].cuda()
            input_ground = data["ground"].cuda()
            input_ground[input_ground > opt.max_depth] = 0
            input_color = torch.cat((input_color, input_ground / opt.max_depth), 1)

            features = encoder(input_color)
            output = depth_decoder(features)

            depth = output["depth"]
            attention = output["ground_attn"]
            attention = attention * (input_ground > 0).float()
            depth = (1 - attention) * depth + attention * input_ground
            pred = depth.clip(opt.min_depth, opt.max_depth)

            preds.append(pred.cpu())
            if use_rotation:
                ktk_inv.append(data["KTK_inv"])
                crop.append(data[("crop", 0)])

    preds = np.concatenate(preds)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation")

    if use_rotation:
        ktk_inv = np.concatenate(ktk_inv)
        d_crop = np.concatenate(crop)

    errors = []

    for i in trange(preds.shape[0]):
        if opt.rotations != [0, 0, 0]:
            gt_depth = warp_depth(gt_depths[i], ktk_inv[i], d_crop[i]).numpy()
        else:
            gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred = preds[i, 0]
        pred = cv2.resize(pred, (gt_width, gt_height))

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0
        pred = pred[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred)

        if opt.enable_median_scaling:
            pred *= ratio

        pred[pred < MIN_DEPTH] = MIN_DEPTH
        pred[pred > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred))

    errors = np.array(errors)
    mean_errors = errors.mean(0)

    print("\n|" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("|{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "|")
    print("\n-> Done!")


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options")
        self.parser.add_argument(
            "--data_path",
            type=str,
            help="path to the training data",
            default=os.path.join("/data/Kitti/"),
        )
        self.parser.add_argument(
            "--num_layers",
            type=int,
            help="number of resnet layers",
            default=50,
            choices=[18, 34, 50, 101, 152],
        )
        self.parser.add_argument(
            "--jpg",
            help="if set, trains from raw KITTI jpg files (instead of png)",
            action="store_true",
        )
        self.parser.add_argument("--height", type=int, help="input image height", default=192)
        self.parser.add_argument("--width", type=int, help="input image width", default=640)
        self.parser.add_argument(
            "--scales",
            nargs="+",
            type=int,
            help="scales used in the loss",
            default=[0, 1, 2, 3],
        )
        self.parser.add_argument("--load_weights_folder", type=str, default="weights/", help="name of model to load")
        self.parser.add_argument(
            "--eval_split",
            type=str,
            default="eigen_benchmark",
            choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
            help="which split to run eval on",
        )
        self.parser.add_argument(
            "--rotations",
            nargs=3,
            type=int,
            default=[0, 0, 0],
            help="Rotations to use for the perspective transform at test time",
        )
        self.parser.add_argument(
            "--camera",
            choices=["front", "front_left", "front_right", "back", "back_left", "back_right"],
            help="Camera to use on ddad dataset",
        )
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=12)
        self.parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
        self.parser.add_argument("--max_depth", type=float, help="maximum depth", default=100.0)
        self.parser.add_argument(
            "--enable_median_scaling", help="if set enables median scaling in evaluation", action="store_true"
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


if __name__ == "__main__":
    options = Parser()
    evaluate(options.parse())
