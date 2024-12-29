#!/usr/bin/env python3


from config import get_config
import matplotlib.pyplot as plt
import seaborn as sns
from lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import optim as optim
from unetr_pp.unetr_pp import UNETR_PP


def lr(config, model, scheduler: str) -> list[float]:
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.MODEL.UNETR_PP.LR,
        momentum=config.MODEL.UNETR_PP.MOMENTUM,
        weight_decay=config.MODEL.UNETR_PP.WEIGHT_DECAY,
        nesterov=config.MODEL.UNETR_PP.NESTEROV,
    )

    if scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.TRAIN.N_EPOCHS,
            eta_min=config.TRAIN.ETA_MIN,
        )
    elif scheduler == "lwcos":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            max_epochs=config.TRAIN.N_EPOCHS,
            warmup_start_lr=config.TRAIN.WARMUP_START_LR,
            eta_min=config.TRAIN.ETA_MIN,
        )
    elif scheduler == "poly":
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer,
            total_iters=config.TRAIN.N_EPOCHS,
            power=config.MODEL.UNETR_PP.POWER,
        )

    lrs = []

    for _ in range(config.TRAIN.N_EPOCHS):
        lrs.extend(scheduler.get_last_lr())
        scheduler.step()

    return lrs


def plt_lr() -> None:
    config = get_config(args=None)

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

    cos = lr(config, model, scheduler="cos")
    lwcos = lr(config, model, scheduler="lwcos")
    poly = lr(config, model, scheduler="poly")

    assert config.TRAIN.N_EPOCHS == len(cos) == len(lwcos) == len(poly)

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=True)

    colors = sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(5, 4))

    ax.plot(range(config.TRAIN.N_EPOCHS), cos, color=colors[0],
            label="CosineAnnealingLR")
    ax.plot(range(config.TRAIN.N_EPOCHS), lwcos, color=colors[6],
            label="LinearWarmupCosineAnnealingLR")
    ax.plot(range(config.TRAIN.N_EPOCHS), poly, color=colors[3],
            label="PolynomialLR")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate vs. Epochs")
    ax.legend()

    plt.tight_layout()

    plt.savefig(f"figures/lr.png", dpi=600)
    plt.show()

    return


def main() -> None:
    plt_lr()

    return


if __name__ == '__main__':
    main()
