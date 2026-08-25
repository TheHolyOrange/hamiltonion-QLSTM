import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(model_cls, requested=None):
    """A model class's PREFERRED_DEVICE (if it declares one) always wins over
    `requested` -- used for models like QLSTM whose PennyLane quantum layers
    only run on CPU, so callers don't have to remember to special-case them."""
    preferred = getattr(model_cls, "PREFERRED_DEVICE", None)
    if preferred is not None:
        return torch.device(preferred)
    return torch.device(requested) if requested is not None else get_device()


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
