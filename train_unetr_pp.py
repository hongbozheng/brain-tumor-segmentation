#!/usr/bin/env python3


from config import SEED, DEVICE, get_config
from data import *
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch import optim as optim
from train import train_model
from unetr_pp.unetr_pp import UNETR_PP


def main() -> None:
    config = get_config(args=None)

    # split data into train & val in dict format
    train_data, val_data = train_val_split(
        data_dir=config.DATA.DIR,
        seed=SEED,
        val_pct=config.DATA.VAL_PCT,
    )

    # create train & val dataloaders
    train_loader, val_loader = loader(
        train_data=train_data,
        val_data=val_data,
        train_batch_size=config.TRAIN.BATCH_SIZE,
        val_batch_size=config.VAL.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEMORY,
        roi=config.MODEL.NNFORMER.CROP_SIZE,
    )

    # define model (has to be these parameters, otherwise doesn't work, i donno why)
    model = UNETR_PP(
        in_channels=config.MODEL.UNETR_PP.IN_CHANNELS,
        out_channels=config.MODEL.UNETR_PP.OUT_CHANNELS,
        feature_size=config.MODEL.UNETR_PP.FEATURE_SIZE,
        hidden_size=config.MODEL.UNETR_PP.HIDDEN_SIZE,
        num_heads=config.MODEL.UNETR_PP.NUM_HEADS,
        pos_embed=config.MODEL.UNETR_PP.POS_EMBED,
        norm_name=config.MODEL.UNETR_PP.NORM_NAME,
        dropout_rate=config.MODEL.UNETR_PP.DROPOUT_RATE,
        depths=config.MODEL.UNETR_PP.DEPTHS,
        dims=config.MODEL.UNETR_PP.DIMS,
        # conv_op=config.CONV_OP,
        do_ds=config.MODEL.UNETR_PP.DO_DS,
    )

    # define optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.MODEL.UNETR_PP.LR,
        momentum=config.MODEL.UNETR_PP.MOMENTUM,
        weight_decay=config.MODEL.UNETR_PP.WEIGHT_DECAY,
        nesterov=config.MODEL.UNETR_PP.NESTEROV,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=config.TRAIN.N_EPOCHS,
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
        ckpt_filepath=config.BEST_MODEL.UNETR_PP,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        roi=config.MODEL.UNETR_PP.ROI,
        sw_batch_size=config.VAL.SW_BATCH_SIZE,
        overlap=config.VAL.OVERLAP,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return


if __name__ == "__main__":
    main()