#!/usr/bin/env python3


from config import get_config
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import seaborn as sns
import torch
from dataset import BraTS
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from monai import transforms
from monai.transforms import Transform
from torch.utils.data import DataLoader


class AddChanneld(Transform):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.unsqueeze(data[key], dim=0)
        return data


def plt_res() -> None:
    config = get_config(args=None)

    data = [
        {
            'image': [
                f'{config.DATA.DIR}/BraTS-GLI-00494-000/BraTS-GLI-00494-000-t2f.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00494-000/BraTS-GLI-00494-000-t1c.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00494-000/BraTS-GLI-00494-000-t1n.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00494-000/BraTS-GLI-00494-000-t2w.nii',
            ],
            'label': f'{config.DATA.DIR}/BraTS-GLI-00494-000/BraTS-GLI-00494-000-seg.nii'
        },
        {
            'image': [
                f'{config.DATA.DIR}/BraTS-GLI-00704-000/BraTS-GLI-00704-000-t2f.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00704-000/BraTS-GLI-00704-000-t1c.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00704-000/BraTS-GLI-00704-000-t1n.nii',
                f'{config.DATA.DIR}/BraTS-GLI-00704-000/BraTS-GLI-00704-000-t2w.nii',
            ],
            'label': f'{config.DATA.DIR}/BraTS-GLI-00704-000/BraTS-GLI-00704-000-seg.nii'
        },
        {
            'image': [
                f'{config.DATA.DIR}/BraTS-GLI-01248-000/BraTS-GLI-01248-000-t2f.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01248-000/BraTS-GLI-01248-000-t1c.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01248-000/BraTS-GLI-01248-000-t1n.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01248-000/BraTS-GLI-01248-000-t2w.nii',
            ],
            'label': f'{config.DATA.DIR}/BraTS-GLI-01248-000/BraTS-GLI-01248-000-seg.nii'
        },
        {
            'image': [
                f'{config.DATA.DIR}/BraTS-GLI-01346-000/BraTS-GLI-01346-000-t2f.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01346-000/BraTS-GLI-01346-000-t1c.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01346-000/BraTS-GLI-01346-000-t1n.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01346-000/BraTS-GLI-01346-000-t2w.nii',
            ],
            'label': f'{config.DATA.DIR}/BraTS-GLI-01346-000/BraTS-GLI-01346-000-seg.nii'
        },
        {
            'image': [
                f'{config.DATA.DIR}/BraTS-GLI-01449-000/BraTS-GLI-01449-000-t2f.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01449-000/BraTS-GLI-01449-000-t1c.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01449-000/BraTS-GLI-01449-000-t1n.nii',
                f'{config.DATA.DIR}/BraTS-GLI-01449-000/BraTS-GLI-01449-000-t2w.nii',
            ],
            'label': f'{config.DATA.DIR}/BraTS-GLI-01449-000/BraTS-GLI-01449-000-seg.nii'
        },
    ]

    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["label"]),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[143, 186, 142],
            ),
        ]
    )

    dataset = BraTS(data=data, transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    for batch in loader:
        image, label = batch['image'], batch['label']

    unet3d_res = torch.load(f="unet3d.pt")
    unetr_res = torch.load(f="unetr.pt")
    unetr_pp_res = torch.load(f="unetr_pp.pt")
    swin_res = torch.load(f="swin.pt")

    labels = ["Background", "Tumor Core", "Whole Tumor", "Enhancing Tumor"]
    colors = sns.color_palette("colorblind")
    colors = [(0.0, 0.0, 0.0), colors[0], colors[3], colors[6]]
    cmap = ListedColormap(colors=colors)

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=True)

    fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, figsize=(14, 10))

    k = [75, 75, 75, 75, 80]

    for i in range(5):
        axes[i, 0].imshow(X=image[i][0][:, :, k[i]], cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(X=unet3d_res[i][:, :, k[i]], cmap=cmap)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(X=unetr_res[i][:, :, k[i]], cmap=cmap)
        axes[i, 2].axis("off")
        axes[i, 3].imshow(X=unetr_pp_res[i][:, :, k[i]], cmap=cmap)
        axes[i, 3].axis("off")
        axes[i, 4].imshow(X=swin_res[i][:, :, k[i]], cmap=cmap)
        axes[i, 4].axis("off")
        axes[i, 5].imshow(X=label[i][0][:, :, k[i]], cmap=cmap)
        axes[i, 5].axis("off")

        if i == 0:
            axes[i, 0].set_title("Image")
            axes[i, 1].set_title("3D-UNet")
            axes[i, 2].set_title("UNETR")
            axes[i, 3].set_title("UNETR++")
            axes[i, 4].set_title("Swin-UNETR")
            axes[i, 5].set_title("Ground Truth")

    legend_patches = [
        mpatches.Patch(color=colors[i], label=labels[i])
        for i in range(len(labels))
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncols=len(labels),
        fontsize=10,
    )

    plt.tight_layout(rect=[0.00, 0.01, 1.00, 1.00])

    plt.savefig(f"figures/results.png", dpi=500)
    plt.show()

    return


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plt_res",
        description="plot segmentation results",
    )

    plt_res()

    return


if __name__ == '__main__':
    main()