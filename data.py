import logger
import os
import random
from config import SEED, get_config
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
        transforms.SpatialPadd(
            keys=["image", "label"],
            method="symmetric",
            spatial_size=config.DATA.ROI,
            mode="constant",
            value=0,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
train_transform.set_random_state(seed=SEED)


# val (test) transform
val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[1, 1, 1],
        ),
        # transforms.SpatialPadd(
        #     keys=["image", "label"],
        #     method="symmetric",
        #     spatial_size=[167, 208, 153],
        #     mode="constant",
        #     value=0,
        # ),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
val_transform.set_random_state(seed=SEED)


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
        images = ["", "", "", ""]
        label = ""

        for file in files:
            filename, _ = os.path.splitext(p=file)
            if filename.endswith("seg"):
                label = os.path.join(path, file)
            else:
                if filename.endswith("t2f"):
                    images[0] = os.path.join(path, file)
                elif filename.endswith("t1c"):
                    images[1] = os.path.join(path, file)
                elif filename.endswith("t1n"):
                    images[2] = os.path.join(path, file)
                elif filename.endswith("t2w"):
                    images[3] = os.path.join(path, file)
                else:
                    logger.log_error("Invalid file!")

        d["image"] = images
        d["label"] = label

        data.append(d)

    return data


def train_val_split(
        data_dir: str,
        seed: int,
        val_pct: float
) -> tuple[list[dict], list[dict]]:
    ids = os.listdir(path=data_dir)
    ids.sort()

    paths = [
        os.path.join(data_dir, id) for id in ids
        if id not in config.DATA.SKIP_IDS
    ]
    data = image_label(paths=paths)

    random.seed(a=seed)
    random.shuffle(x=data)

    val_data = data[:int(len(data)*val_pct)]
    train_data = data[int(len(data)*val_pct):]

    return train_data, val_data
