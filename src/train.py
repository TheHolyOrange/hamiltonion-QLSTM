"""Training / evaluation loop for QLSTMRegressor on ETTh1."""
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.qlstm_model import QLSTMRegressor
from src.utils import rmse, mae, mape, resolve_device


def make_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, loss_fn, optimizer=None, device=None, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if is_train:
                loss.backward()
                # Optional global-norm gradient clipping: a cheap stabilizer for
                # VQC training, where occasional large gradients (e.g. near a
                # rotation-angle wraparound) can otherwise cause a single Adam
                # step that overshoots a good val-loss region it had just
                # reached. Off by default (grad_clip=None) so existing runs
                # (ETTh1 QLSTM/H-QLSTM) are unaffected.
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def collect_predictions(model, loader, device=None):
    model.eval()
    preds, actuals = [], []
    for x, y in loader:
        if device is not None:
            x, y = x.to(device), y.to(device)
        out = model(x)
        preds.append(out)
        actuals.append(y)
    return torch.cat(preds).cpu().numpy(), torch.cat(actuals).cpu().numpy()


def train_model(
    model_cfg,
    train_ds,
    val_ds,
    test_ds=None,
    lr=0.01,
    batch_size=32,
    num_epochs=25,
    patience=6,
    verbose=True,
    log_fn=print,
    checkpoint_path=None,
    device=None,
    model_cls=QLSTMRegressor,
    grad_clip=None,
    use_scheduler=False,
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-4,
):
    """If checkpoint_path is given, progress (model/optimizer/epoch/history/best
    state) is saved to it after every epoch, and training resumes from it on
    the next call if the file already exists (interrupted-run recovery).

    grad_clip: optional global gradient-norm clip (see run_epoch).
    use_scheduler: if True, wraps optimizer in ReduceLROnPlateau(mode='min',
        factor=scheduler_factor, patience=scheduler_patience,
        min_lr=scheduler_min_lr), stepped on val_loss each epoch. Targets
        oscillating (rather than diverging) val loss -- a fixed lr that is
        too large to let Adam settle into a val-loss basin once it finds one
        -- by cutting lr once val_loss stalls for scheduler_patience epochs,
        independent of the early-stopping patience/epochs_no_improve count.
        Both default off/unset so existing runs are bit-for-bit unaffected.
    """
    device = resolve_device(model_cls, device)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)

    model = model_cls(**model_cfg).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr)

    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        history.setdefault("lr", [])
        best_val = ckpt["best_val"]
        best_state = ckpt["best_state"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if verbose:
            log_fn(f"resumed from checkpoint at epoch {start_epoch} (best_val={best_val:.5f})")

    if verbose:
        log_fn(f"training on device: {device}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device=device, grad_clip=grad_clip)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(cur_lr)

        if verbose:
            log_fn(f"epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.5f}  "
                   f"val_loss={val_loss:.5f}  lr={cur_lr:.5f}  ({time.time()-t0:.1f}s)")

        if scheduler is not None:
            prev_lr = cur_lr
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if verbose and new_lr < prev_lr - 1e-12:
                log_fn(f"  lr reduced: {prev_lr:.5f} -> {new_lr:.5f}")

        stop_early = False
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    log_fn(f"early stopping at epoch {epoch+1} (best val_loss={best_val:.5f})")
                stop_early = True

        if checkpoint_path:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_val": best_val,
                "best_state": best_state,
                "epochs_no_improve": epochs_no_improve,
            }, checkpoint_path)

        if stop_early:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {"model": model, "history": history, "best_val_loss": best_val}

    if test_ds is not None:
        test_loss = run_epoch(model, test_loader, loss_fn, optimizer=None, device=device)
        result["test_loss"] = test_loss

    return result
