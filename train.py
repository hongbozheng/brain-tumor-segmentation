import config
import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from data import *
from tqdm import tqdm
from val import val_epoch


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn,
        optimizer: optim.Optimizer,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=1)
    loader_tqdm.set_description(
        desc=f"[Batch 0]",
        refresh=True
    )

    loss_meter = AverageMeter()

    for idx, batch in enumerate(iterable=loader_tqdm):
        data = batch["image"].to(device=config.DEVICE)
        target = batch["label"].to(device=config.DEVICE)
        logits = model(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        loss_meter.update(val=loss.item(), n=data.shape[0])
        loader_tqdm.set_description(
            desc=f"[Batch {idx+1}]: train loss {loss_meter.avg:.6f}",
            refresh=True
        )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        device,
        ckpt_filepath: str,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        n_epochs: int,
        loss_fn,
        roi: tuple[int, int, int],
        sw_batch_size: int,
        overlap: float,
        acc_fn,
        train_loader: DataLoader,
        val_loader: DataLoader,
) -> None:
    path, _ = os.path.split(p=ckpt_filepath)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)

    start_epoch = 0
    avg_dice_losses = []
    avg_dice_scores = []

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]+1
        avg_dice_scores.append(ckpt["best_dice"])
        filename = os.path.basename(ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")

    epoch_tqdm = tqdm(iterable=range(start_epoch, n_epochs), position=0)

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(desc=f"[Epoch {epoch}]", refresh=True)
        avg_dice_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        avg_dice_losses.append(avg_dice_loss)

        dice_scores = val_epoch(
            model=model,
            val_loader=val_loader,
            roi=roi,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            acc_fn=acc_fn,
        )
        avg_dice_score = np.mean(a=dice_scores, dtype=np.float32)
        avg_dice_scores.append(avg_dice_score)

        # if avg_dice_scores[-1] == max(avg_dice_scores):
        #     torch.save(
        #         obj={
        #             "model_description": str(model),
        #             "model_state": model.state_dict(),
        #             "optimizer_state": optimizer.state_dict(),
        #             "scheduler_state": scheduler.state_dict(),
        #             "epoch": epoch,
        #             "best_dice": avg_dice_scores[-1],
        #         },
        #         f=ckpt_filepath,
        #     )

        # scheduler.step()

    return