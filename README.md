# GroCo: Ground Constraint for Metric Self-Supervised Monocular Depth

This repository contains the inference code for the GroCo method (ECCV24).

## Example

The model can be run on an example using the notebook `example.ipynb`.

## Weights and Benchmark

Model and annotations are provided in this [google drive](https://drive.google.com/drive/folders/1-4GVXrCe-5UMcSidK6seAsJh5SmwabYk?usp=sharing)

Weights should be placed in a created `weights` folder and annotations in `splits/eigen_benchmark/gt_depths.npz`.

## Evaluation

Evaluation on KITTI can be run using the following command:

```bash
python evaluate_depth.py
```

Camera Rotations can be added with the --rotations flag, with the format [Pitch, Yaw, Roll].
Rotations are limited to 5 degrees for pitch and roll and 15 degrees for yaw to not create black borders.

```bash
python evaluate_depth.py --rotations 0 0 5
```

## Dataset preparation

Download the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).
Data structure should be as follows:

```
Kitti
├── 2011_09_26
│   ├── 2011_09_26_drive_0001_sync
│   │   ├── image_02
│   │   │   └── data
│   │   ├── image_03
│   │   │   └── data
│   │   ├── oxts
│   │   │   └── data
│   │   └── velodyne_points
│   │       └── data
    ...
└── 2011_09_28
    ...
└── 2011_09_29
    ...
└── 2011_09_30
    ...
└── 2011_10_03
```
