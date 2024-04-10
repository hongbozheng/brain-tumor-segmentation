import os
import random
from dataset import BraTS
from monai import transforms
from torch.utils.data import DataLoader
from typing import Sequence


def __image_label(ids: list[str], data_dir: str) -> list:
    """
    create list of dictionaries for images and label
    [{                                  {
        'img': ['image1.nii.gz', ...],      'img': ['image2.nii.gz', ...],
        'seg': 'label1.nii.gz',             'seg': 'label2.nii.gz',
     },                                 }, ...]

    :param ids: subject ids under dataset folder
    :param data_dir: data directory
    :return: list of dictionaries
    """
    data = []

    for id in ids:
        files = os.listdir(path=os.path.join(data_dir, id))
        files.sort()

        d = {}
        images = []

        for file in files:
            filename, _ = os.path.splitext(p=file)
            if filename.endswith("seg"):
                d["label"] = os.path.join(data_dir, id, file)
            else:
                images.append(os.path.join(data_dir, id, file))
        d["image"] = images

        data.append(d)

    return data


def train_val_split(data_dir: str, seed: int, val_pct: float) -> tuple[list[dict], list[dict]]:
    ids = os.listdir(path=data_dir)
    ids.sort()

    random.seed(a=seed)
    random.shuffle(x=ids)

    val_ids = ids[:int(len(ids)*val_pct)]
    train_ids = ids[int(len(ids)*val_pct):]

    val_data = __image_label(ids=val_ids, data_dir=data_dir)
    train_data = __image_label(ids=train_ids, data_dir=data_dir)

    assert len(val_ids) == len(val_data)
    assert len(train_ids) == len(train_data)

    return train_data, val_data


def loader(
        train_data: list[dict],
        val_data: list[dict],
        roi: Sequence[int] | int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    # train & val transform
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[1, 1, 1],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=roi,
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # train & val dataset
    train_dataset = BraTS(data=train_data, transform=train_transform)
    val_dataset = BraTS(data=val_data, transform=val_transform)

    # train & val dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader