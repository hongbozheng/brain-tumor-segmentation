#!/usr/bin/env python3


import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot(filepath: str) -> None:
    file = open(file=filepath, mode='r')

    data = json.load(fp=file)

    epochs = data['epoch']
    avg_dice_loss = data['avg_dice_loss']
    dice_score = np.array(object=data['dice_score'], dtype=np.float32)
    avg_dice_score = data['avg_dice_score']

    file.close()

    assert (len(epochs) == len(avg_dice_score)
            == dice_score.shape[0] == len(avg_dice_score))

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=True)

    colors = sns.color_palette("colorblind")

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 3))

    axes[0].plot(epochs, avg_dice_loss)
    axes[0].set_xlabel("Epochs")
    axes[0].set_title("Average Dice Loss")

    axes[1].plot(epochs, dice_score[:, 0], color=colors[0],
                 label="Tumor Core")
    axes[1].plot(epochs, dice_score[:, 1], color=colors[3],
                 label="Whole Tumor")
    axes[1].plot(epochs, dice_score[:, 2], color=colors[6],
                 label="Enhancing Tumor")
    axes[1].set_xlabel("Epochs")
    axes[1].set_title("Dice Scores")
    axes[1].legend()

    axes[2].plot(epochs, avg_dice_score)
    axes[2].set_xlabel("Epochs")
    axes[2].set_title("Average Dice Score")

    plt.tight_layout()

    filename, _ = os.path.splitext(p=os.path.basename(p=filepath))
    plt.savefig(f"figures/{filename}.png", dpi=600)
    plt.show()

    return


def main() -> None:
    parser = argparse.ArgumentParser(prog="plt", description="plot statistics")
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="path to json file"
    )
    args = parser.parse_args()
    filepath = args.filepath

    plot(filepath=filepath)

    return


if __name__ == '__main__':
    main()
