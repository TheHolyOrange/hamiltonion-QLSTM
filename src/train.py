"""Training / evaluation loop for QLSTMRegressor on ETTh1."""
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.qlstm_model import QLSTMRegressor
from src.utils import rmse, mae, mape


def make_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, loss_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    preds, actuals = [], []
    for x, y in loader:
        out = model(x)
        preds.append(out)
        actuals.append(y)
    return torch.cat(preds).numpy(), torch.cat(actuals).numpy()


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
):
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)

    model = QLSTMRegressor(**model_cfg)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            log_fn(f"epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.5f}  "
                   f"val_loss={val_loss:.5f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    log_fn(f"early stopping at epoch {epoch+1} (best val_loss={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {"model": model, "history": history, "best_val_loss": best_val}

    if test_ds is not None:
        test_loss = run_epoch(model, test_loader, loss_fn, optimizer=None)
        result["test_loss"] = test_loss

    return result
