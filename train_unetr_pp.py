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
        data_dir=config.DATA.DATA_DIR,
        seed=SEED,
        val_pct=config.DATA.VAL_PCT,
    )

    # create train & val dataloaders
    train_loader, val_loader = loader(
        train_data=train_data,
        val_data=val_data,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEMORY,
        roi=config.MODEL.UNETRPP.ROI,
    )

    # define model (has to be these parameters, otherwise doesn't work, i donno why)
    model = UNETR_PP(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        feature_size=16,
        hidden_size=256,
        num_heads=4,
        pos_embed="perceptron",
        norm_name=config.NORM_NAME,
        dropout_rate=config.DROPOUT_RATE,
        depths=[3, 3, 3, 3],
        dims=config.DIMS,
        conv_op=config.CONV_OP,
        do_ds=False,
    )

    # define optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.MODEL.UNETRPP.LR,
        momentum=config.MODEL.UNETRPP.MOMENTUM,
        weight_decay=config.MODEL.UNETRPP.WEIGHT_DECAY,
        nesterov=config.MODEL.UNETRPP.NESTEROV,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=config.TRAIN.N_EPOCHS,
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
        ckpt_filepath=config.SAVE.UNETRPP_BEST_MODEL,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return


if __name__ == "__main__":
    main()