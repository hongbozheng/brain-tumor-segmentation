import os
import random
from config import get_config
from monai import transforms


config = get_config(args=None)


# train transform
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
            roi_size=config.DATA.ROI,
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


# val (test) transform
val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


def image_label(paths: list[str]) -> list[dict]:
    """
    create list of dictionaries for images and label
    [{
        'sub_id': 'BraTS-GLI-XXXXX-XXX'
        'image': ['BraTS-GLI-XXXXX-XXX-XXX.nii.gz', ...],
        'label': 'BraTS-GLI-XXXXX-XXX-seg.nii.gz',
     },
     {
        'sub_id': 'BraTS-GLI-XXXXX-XXX'
        'image': ['BraTS-GLI-XXXXX-XXX-XXX.nii.gz', ...],
        'label': 'BraTS-GLI-XXXXX-XXX-seg.nii.gz',
     }, ...]

    :param paths: subject id paths
    :return: list of dictionaries
    """
    data = []

    for path in paths:
        files = os.listdir(path=path)

        d = {}
        images = []

        for file in files:
            filename, _ = os.path.splitext(p=file)
            if filename.endswith("seg"):
                d["label"] = os.path.join(path, file)
            else:
                images.append(os.path.join(path, file))
        d["image"] = images

        data.append(d)

    return data


def train_val_split(data_dir: str, seed: int, val_pct: float) -> tuple[list[dict], list[dict]]:
    ids = os.listdir(path=data_dir)
    ids.sort()

    paths = [os.path.join(data_dir, id) for id in ids]
    data = image_label(paths=paths)

    random.seed(a=seed)
    random.shuffle(x=ids)

    val_data = data[:int(len(data)*val_pct)]
    train_data = data[int(len(data)*val_pct):]

    return train_data, val_data