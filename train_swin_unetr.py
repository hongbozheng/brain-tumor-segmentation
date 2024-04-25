#!/usr/bin/env python3


from config import SEED, DEVICE, get_config
from data import train_val_split, train_transform, val_transform
from dataset import BraTS
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch import optim as optim
from torch.utils.data import DataLoader
from train import train_model
from swin_unetr import SwinUNETR


def main() -> None:
    config = get_config(args=None)

    # split data into train & val in dict format
    train_data, val_data = train_val_split(
        data_dir=config.DATA.DIR,
        seed=SEED,
        val_pct=config.DATA.VAL_PCT,
    )

    # dataset
    train_dataset = BraTS(data=train_data, transform=train_transform)
    val_dataset = BraTS(data=val_data, transform=val_transform)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.LOADER.NUM_WORKERS_TRAIN,
        pin_memory=config.LOADER.PIN_MEMORY,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=config.LOADER.NUM_WORKERS_VAL,
        pin_memory=config.LOADER.PIN_MEMORY,
    )

    # define model
    model = SwinUNETR(
        img_size=config.MODEL.SWIN.ROI,
        in_channels=config.MODEL.SWIN.IN_CHANNELS,
        out_channels=config.MODEL.SWIN.OUT_CHANNELS,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        feature_size=config.MODEL.SWIN.FEATURE_SIZE,
        norm_name=config.MODEL.SWIN.NORM_NAME,
        drop_rate=config.MODEL.SWIN.DROP_RATE,
        attn_drop_rate=config.MODEL.SWIN.ATTN_DROP_RATE,
        dropout_path_rate=config.MODEL.SWIN.DROPOUT_PATH_RATE,
        normalize=config.MODEL.SWIN.NORMALIZE,
        use_checkpoint=config.MODEL.SWIN.USE_CHECKPOINT,
        spatial_dims=config.MODEL.SWIN.SPATIAL_DIMS,
        downsample=config.MODEL.SWIN.DOWNSAMPLE,
        use_v2=config.MODEL.SWIN.USE_V2,
    )

    # define optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.MODEL.SWIN.LR,
        weight_decay=config.MODEL.SWIN.WEIGHT_DECAY,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=300,
        warmup_start_lr=config.TRAIN.WARMUP_START_LR,
        eta_min=config.TRAIN.ETA_MIN,
    )

    # loss fn (train)
    loss_fn = DiceLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        softmax=False,
    )

    # acc fn (val)
    acc_fn = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    train_model(
        model=model,
        device=DEVICE,
        ckpt_filepath=config.BEST_MODEL.SWIN,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        roi=config.MODEL.SWIN.ROI,
        sw_batch_size=config.VAL.SW_BATCH_SIZE,
        overlap=config.VAL.OVERLAP,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        stats_filepath="swin_" + config.TRAIN.STATS_FILEPATH,
    )

    return


if __name__ == "__main__":
    main()
