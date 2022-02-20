import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

from .. import log, Metrics


def train(model: Module, train_loader: DataLoader,
          test_loader: DataLoader, criterion, optimizer, scheduler,
          epochs: int, metrics: Metrics,
          device, tensorboard=None, test_step=10):
    for epoch in range(1, epochs + 1):
        log.debug(f"Epoch: {epoch}")
        loader = tqdm(train_loader, desc=f"| training [{epoch}|{epochs}]", file=sys.stdout)
        avg_loss = _train_step(model, loader, criterion, optimizer, scheduler, metrics, device)
        if tensorboard:
            _tensorboard_plot(tensorboard, "train", avg_loss, metrics, epoch)
        if test_loader and epoch % test_step == 0:
            loader = tqdm(test_loader, desc=f"| testing [{epoch}|{epochs}]", file=sys.stdout)
            avg_loss = _test_step(model, loader, criterion, metrics, device)
            if tensorboard:
                _tensorboard_plot(tensorboard, "test", avg_loss, metrics, epoch)


def _train_step(model: Module, loader:tqdm, criterion,
                optimizer, scheduler, metrics, device):
    model.train()
    model.to(device)
    metrics.reset()
    Loss = 0
    count = 0

    for batch in loader:
        data, target = batch[0].to(device), batch[1].to(device)
        if len(target.size()) > 1:
            target = target.argmax(1)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss += loss
        count += 1
        metrics(output, target)
        tqdm.set_postfix(loader,
                         {
                             "loss": f"{Loss / count:.6f}",
                             "acc": f"{getattr(metrics, 'accuracy'):.4%}"
                         })
    if scheduler:
        scheduler.step()
    return Loss / count


def _test_step(model: Module, loader: tqdm, criterion,
               metrics, device):
    model.eval()
    model.to(device)
    metrics.reset()
    Loss = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0].to(device), batch[1].to(device)
            if len(target.size()) > 1:
                target = target.argmax(1)

            output = model(data)
            loss = criterion(output, target)

            Loss += loss
            count += 1
            metrics(output, target)
            tqdm.set_postfix(loader,
                             {
                                 "loss": f"{Loss / count:.6f}",
                                 "acc": f"{getattr(metrics, 'accuracy'):.4%}"
                             })
    return Loss / count


def _tensorboard_plot(tensorboard: SummaryWriter, prefix: str,
                      avg_loss: float, metrics: Metrics, epoch: int):
    log.debug(f"Loss: {avg_loss}")
    tensorboard.add_scalar(f"{prefix}_loss", avg_loss, epoch)
    names = metrics.get_names()
    for name in names:
        if hasattr(metrics, name):
            log.debug(f"{name}: {getattr(metrics, name)}")
            tensorboard.add_scalar(f"{prefix}_{name}", getattr(metrics, name), epoch)