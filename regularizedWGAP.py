#!/usr/bin/env python3
import os
import sys
import copy
import uuid
import argparse
import traceback
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# DEFAULT SETTINGS
# Can all be overridden from command line
# ============================================================
DEFAULT_BASE_DIR = "./datasets"

DATASETS = [
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinC_ECG_torso",
    "Coffee",
    "Computers",
    "Cricket_X",
    "Cricket_Y",
    "Cricket_Z",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "Meat",
    "ScreenType",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "synthetic_control",
    "Wine",
    "WordsSynonyms",
    "Worms",
    "WormsTwoClass",
    "yoga",
]

DEFAULT_SEEDS = [1, 2, 3, 4, 5]

DEFAULT_NO_NORM = False

DEFAULT_EPOCHS = 200
DEFAULT_CV_EPOCHS = 50
DEFAULT_LR = 1e-3

DEFAULT_K_FOLDS = 5
DEFAULT_LAMBDA_GRID = [
    0.001, 0.002, 0.008, 0.032, 0.128,
    0.256, 0.512, 1.024, 2.048,
    4.096, 8.192, 16.384, 32.768
]

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_USE_AMP = True
DEFAULT_CV_EARLY_STOP_PATIENCE = 40

DEFAULT_STORE_LAMBDA_TEST_CURVE = True
DEFAULT_VERBOSE = True

DEFAULT_OUT_DIR = "results_fcn_gap_wgap_all"
# ============================================================


# =============================
# Tee logger: terminal + file
# =============================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# =============================
# Utilities
# =============================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def z_normalize_per_series(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: (N,T,C) -> normalize across time for each sample (and channel)
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < eps] = 1.0
    return (x - mean) / std


def pick_file(path_no_ext):
    if os.path.exists(path_no_ext):
        return path_no_ext
    for ext in [".txt", ".tsv", ".csv"]:
        cand = path_no_ext + ext
        if os.path.exists(cand):
            return cand
    return path_no_ext + ".txt"


def make_unique_results_path(out_dir: str, prefix: str = "gap_wgap_all"):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    token = uuid.uuid4().hex[:8]
    fname = f"{prefix}__{stamp}__pid{pid}__{token}.txt"
    return os.path.join(out_dir, fname)


class Conv1dSame(nn.Module):
    """
    TensorFlow-style 'same' Conv1d for stride=1, dilation=1.
    Keeps output length equal to input length for any kernel size.
    """
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        self.k = int(kernel_size)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.k, padding=0, bias=bias)

    def forward(self, x):
        left = (self.k - 1) // 2
        right = (self.k - 1) - left
        x = F.pad(x, (left, right))
        return self.conv(x)


def _loadtxt_auto_delim(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path, dtype=np.float64, delimiter=",")
    except ValueError:
        return np.loadtxt(path, dtype=np.float64, delimiter=None)


def load_ucr_like(dataset_name: str, base_dir: str):
    train_path = pick_file(os.path.join(base_dir, dataset_name, f"{dataset_name}_TRAIN"))
    test_path  = pick_file(os.path.join(base_dir, dataset_name, f"{dataset_name}_TEST"))

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = _loadtxt_auto_delim(train_path)
    test = _loadtxt_auto_delim(test_path)

    y_train = train[:, 0]
    x_train = train[:, 1:]
    y_test = test[:, 0]
    x_test = test[:, 1:]

    uniq = np.unique(y_train)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_train = np.array([mapping[v] for v in y_train], dtype=np.int64)
    y_test = np.array([mapping[v] for v in y_test], dtype=np.int64)

    x_train = x_train.astype(np.float32)[:, :, None]   # (N,T,1)
    x_test = x_test.astype(np.float32)[:, :, None]

    return x_train, y_train, x_test, y_test, train_path, test_path


def make_loaders_from_arrays(x, y, batch_size, shuffle, num_workers=0):
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    loader_kwargs = dict(
        dataset=TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(**loader_kwargs)


def make_loaders(x_train, y_train, x_test, y_test, batch_size, num_workers=0):
    train_loader = make_loaders_from_arrays(
        x_train, y_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = make_loaders_from_arrays(
        x_test, y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


# =============================
# Stratified K-fold split
# =============================
def stratified_kfold_indices(y: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = y.shape[0]
    classes = np.unique(y)

    per_class = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        per_class[c] = idx

    folds = [[] for _ in range(k)]
    for c in classes:
        idx = per_class[c]
        chunks = np.array_split(idx, k)
        for i in range(k):
            folds[i].append(chunks[i])

    val_folds = [
        np.concatenate(folds[i]) if len(folds[i]) else np.array([], dtype=int)
        for i in range(k)
    ]

    out = []
    all_idx = np.arange(n)
    for i in range(k):
        val_idx = val_folds[i]
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_idx] = True
        train_idx = all_idx[~val_mask]
        out.append((train_idx, val_idx))
    return out


# =============================
# Pooling layers
# =============================
class GAP1D(nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


class WeightedGAP1D(nn.Module):
    """
    Unconstrained WGAP
    Forward: (B,C,T)->(B,C) via sum_t a_t x_{.,.,t}
    """
    def __init__(self, T: int):
        super().__init__()
        self.T = int(T)
        self.a = nn.Parameter(torch.zeros(self.T))

    def forward(self, x):
        a = self.a.view(1, 1, self.T)
        return torch.sum(x * a, dim=-1)


class WeightedGAP1D_Penalized(nn.Module):
    """
    Penalized WGAP:
        penalty(a) = sum_{t=1}^{T-1} (a_{t+1} - a_t)^2
    """
    def __init__(self, T: int):
        super().__init__()
        self.T = int(T)
        self.a = nn.Parameter(torch.zeros(self.T))

    def forward(self, x):
        a = self.a.view(1, 1, self.T)
        return torch.sum(x * a, dim=-1)

    def smoothness_penalty(self):
        if self.T <= 1:
            return self.a.new_tensor(0.0)
        diffs = self.a[1:] - self.a[:-1]
        return torch.sum(diffs * diffs)


# =============================
# FCN architecture
# =============================
class FCN_Pooling(nn.Module):
    def __init__(self, n_classes: int, T: int, in_channels: int = 1, pooling: str = "gap"):
        super().__init__()
        self.T = int(T)

        self.conv1 = Conv1dSame(in_channels, 128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = Conv1dSame(128, 256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = Conv1dSame(256, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)

        pooling = pooling.lower()
        if pooling == "gap":
            self.pool = GAP1D()
        elif pooling == "wgap":
            self.pool = WeightedGAP1D(T=T)
        elif pooling == "wgap_reg":
            self.pool = WeightedGAP1D_Penalized(T=T)
        else:
            raise ValueError("pooling must be 'gap', 'wgap', or 'wgap_reg'")

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # Input expected as (B,T,C) or (B,C,T)
        if x.dim() != 3:
            raise ValueError("Expected x to have shape (B,T,C) or (B,C,T).")
        if x.shape[1] == self.T:
            x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        z = self.pool(x)
        logits = self.fc(z)
        return logits


# =============================
# Eval helpers
# =============================
@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion_sum(logits, y).item()
        pred = logits.argmax(dim=1)

        total_correct += int((pred == y).sum().item())
        total_loss += float(loss)
        total_n += int(y.numel())

    return {
        "acc": total_correct / max(total_n, 1),
        "loss": total_loss / max(total_n, 1),
        "n": total_n,
    }


def _model_penalty(model):
    pool = getattr(model, "pool", None)
    if pool is None or not hasattr(pool, "smoothness_penalty"):
        return None
    return pool.smoothness_penalty()


# =============================
# Training
# =============================
def train_model(
    model,
    train_loader,
    eval_loader,
    device,
    *,
    epochs=500,
    lr=1e-3,
    min_lr=1e-4,
    plateau_patience=25,
    plateau_factor=0.5,
    weight_decay=0.0,
    lambda_reg: float = 0.0,
    select_by: str = "train_loss",   # "last", "train_loss", "eval_acc"
    eval_every: int = 1,
    early_stop_patience=None,
    use_amp: bool = False,
    verbose: bool = False,
    log_fn=None,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=plateau_factor, patience=plateau_patience, min_lr=min_lr
    )
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    select_by = select_by.lower()
    if select_by not in ("last", "train_loss", "eval_acc"):
        raise ValueError("select_by must be 'last', 'train_loss', or 'eval_acc'")

    best_state = None
    best_epoch = None
    best_train_loss = float("inf")
    best_eval_acc = -1.0
    epochs_no_improve = 0

    last_eval_metrics = None

    for ep in range(1, epochs + 1):
        model.train()

        total_loss_sum = 0.0
        total_correct = 0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
                    if lambda_reg > 0.0:
                        pen = _model_penalty(model)
                        if pen is not None:
                            loss = loss + float(lambda_reg) * pen

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if lambda_reg > 0.0:
                    pen = _model_penalty(model)
                    if pen is not None:
                        loss = loss + float(lambda_reg) * pen

                loss.backward()
                opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total_correct += int((pred == y).sum().item())
                total_n += int(y.numel())
                total_loss_sum += float(loss.item()) * int(y.numel())

        train_loss_epoch = total_loss_sum / max(total_n, 1)
        train_acc_epoch = total_correct / max(total_n, 1)

        sched.step(train_loss_epoch)

        do_eval = (ep % eval_every == 0) or (ep == epochs)
        if do_eval:
            eval_metrics_epoch = eval_metrics(model, eval_loader, device)
            last_eval_metrics = eval_metrics_epoch

            improved = False

            if select_by == "train_loss":
                if train_loss_epoch < best_train_loss:
                    best_train_loss = train_loss_epoch
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = ep
                    improved = True

            elif select_by == "eval_acc":
                if eval_metrics_epoch["acc"] > best_eval_acc:
                    best_eval_acc = eval_metrics_epoch["acc"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = ep
                    improved = True

            if select_by == "eval_acc" and early_stop_patience is not None:
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop_patience:
                    if verbose:
                        msg = (
                            f"  early stop at epoch {ep} | "
                            f"best epoch={best_epoch} | best eval acc={best_eval_acc:.4f}"
                        )
                        if log_fn is not None:
                            log_fn(msg)
                        else:
                            print(msg)
                    break

        if verbose and (ep == 1 or ep % 50 == 0 or ep == epochs):
            cur_lr = opt.param_groups[0]["lr"]
            msg = (
                f"  ep {ep:4d}/{epochs} | lr={cur_lr:.2e} | "
                f"train loss={train_loss_epoch:.5f} | train acc={train_acc_epoch:.4f}"
            )
            if last_eval_metrics is not None:
                msg += (
                    f" | acc_test={last_eval_metrics['acc']:.4f} "
                    f"| loss_test={last_eval_metrics['loss']:.5f}"
                )
            if lambda_reg > 0:
                msg += f" | lambda={lambda_reg:g}"

            if log_fn is not None:
                log_fn(msg)
            else:
                print(msg)

    if select_by in ("train_loss", "eval_acc") and best_state is not None:
        model.load_state_dict(best_state)

    tr_final = eval_metrics(model, train_loader, device)
    ev_final = eval_metrics(model, eval_loader, device)
    ev_final["best_epoch"] = best_epoch if best_epoch is not None else epochs
    return tr_final, ev_final


def crossval_select_lambda_wgap_reg(
    x_train, y_train,
    *,
    T, C, n_classes,
    device,
    lambda_grid,
    k_folds=5,
    seed=1,
    epochs=200,
    lr=1e-3,
    batch_size=64,
    num_workers=0,
    early_stop_patience=40,
    use_amp=False,
    verbose=False,
    log_fn=None,
):
    splits = stratified_kfold_indices(y_train, k=k_folds, seed=seed)

    best_lambda = None
    best_mean_acc = -1.0
    all_scores = {}

    for lam in lambda_grid:
        fold_accs = []

        for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
            xtr, ytr = x_train[tr_idx], y_train[tr_idx]
            xva, yva = x_train[va_idx], y_train[va_idx]

            tr_loader = make_loaders_from_arrays(
                xtr, ytr, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            va_loader = make_loaders_from_arrays(
                xva, yva, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

            fold_seed = int(seed * 100000 + round(float(lam) * 1000) + fold_id)
            set_seed(fold_seed)

            model = FCN_Pooling(
                n_classes=n_classes,
                T=T,
                in_channels=C,
                pooling="wgap_reg"
            )

            _, va_best = train_model(
                model,
                tr_loader,
                va_loader,
                device,
                epochs=epochs,
                lr=lr,
                lambda_reg=float(lam),
                select_by="eval_acc",
                eval_every=1,
                early_stop_patience=early_stop_patience,
                use_amp=use_amp,
                verbose=False,
                log_fn=None,
            )

            fold_accs.append(float(va_best["acc"]))

        mean_acc = float(np.mean(fold_accs)) if len(fold_accs) else float("nan")
        all_scores[float(lam)] = {
            "mean_acc": mean_acc,
            "fold_accs": fold_accs,
        }

        if verbose:
            msg = (
                f"    lambda={lam:g} | "
                f"k-fold accs={np.round(fold_accs, 4).tolist()} | "
                f"mean={mean_acc:.4f}"
            )
            if log_fn is not None:
                log_fn(msg)
            else:
                print(msg)

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_lambda = float(lam)

    return best_lambda, best_mean_acc, all_scores


# =============================
# Formatting helpers
# =============================
def mean_pm_sd(arr):
    arr = np.array(arr, dtype=float)
    mean = arr.mean()
    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return mean, sd


def fmt_mean_sd(arr):
    m, s = mean_pm_sd(arr)
    return f"{m:.4f} ± {s:.4f}"


# =============================
# CLI
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="Run GAP / WGAP / reg_WGAP experiments")

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR,
                        help="Root folder containing dataset subfolders")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output folder for logs and results")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--cv_epochs", type=int, default=DEFAULT_CV_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)

    parser.add_argument("--k_folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

    parser.add_argument("--cv_early_stop_patience", type=int, default=DEFAULT_CV_EARLY_STOP_PATIENCE)

    parser.add_argument("--no_norm", action="store_true", help="Disable per-series z-normalization")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose progress printing")
    parser.add_argument("--no_curve_csv", action="store_true", help="Do not save lambda curve csv")

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds list, e.g. --seeds 1 2 3 4 5"
    )

    return parser.parse_args()


# =============================
# Main run
# =============================
def main():
    args = parse_args()

    BASE_DIR = args.base_dir
    OUT_DIR = args.out_dir
    EPOCHS = args.epochs
    CV_EPOCHS = args.cv_epochs
    LR = args.lr
    K_FOLDS = args.k_folds
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CV_EARLY_STOP_PATIENCE = args.cv_early_stop_patience
    NO_NORM = args.no_norm
    USE_AMP = not args.no_amp
    STORE_LAMBDA_TEST_CURVE = not args.no_curve_csv
    VERBOSE = not args.quiet
    SEEDS = args.seeds
    LAMBDA_GRID = DEFAULT_LAMBDA_GRID

    os.makedirs(OUT_DIR, exist_ok=True)

    run_path = make_unique_results_path(out_dir=OUT_DIR, prefix="gap_wgap_all")
    curve_csv_path = run_path.replace(".txt", "__lambda_test_curve.csv")

    f_run = open(run_path, "w", encoding="utf-8")
    f_curve = None

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect everything printed to terminal also into the log file
        sys.stdout = Tee(original_stdout, f_run)
        sys.stderr = Tee(original_stderr, f_run)

        if STORE_LAMBDA_TEST_CURVE:
            f_curve = open(curve_csv_path, "w", encoding="utf-8")
            f_curve.write("dataset,seed,lambda,acc_test,loss_test,best_epoch\n")
            f_curve.flush()

        def log_both(s: str = ""):
            print(s, flush=True)

        device = get_device()
        use_amp = bool(USE_AMP and device.type == "cuda")

        log_both(f"Device: {device}")
        log_both(f"AMP enabled: {use_amp}")
        log_both("=== RUN CONFIG ===")
        log_both(f"BASE_DIR                    : {BASE_DIR}")
        log_both(f"DATASETS                    : {DATASETS}")
        log_both(f"SEEDS                       : {SEEDS}")
        log_both(f"NO_NORM                     : {NO_NORM}")
        log_both(f"EPOCHS                      : {EPOCHS}")
        log_both(f"CV_EPOCHS                   : {CV_EPOCHS}")
        log_both(f"LR                          : {LR}")
        log_both(f"K_FOLDS                     : {K_FOLDS}")
        log_both(f"LAMBDA_GRID                 : {LAMBDA_GRID}")
        log_both(f"BATCH_SIZE                  : {BATCH_SIZE}")
        log_both(f"NUM_WORKERS                 : {NUM_WORKERS}")
        log_both(f"CV_EARLY_STOP_PATIENCE      : {CV_EARLY_STOP_PATIENCE}")
        log_both(f"STORE_LAMBDA_TEST_CURVE     : {STORE_LAMBDA_TEST_CURVE}")
        log_both(f"VERBOSE                     : {VERBOSE}")
        log_both(f"OUT_DIR                     : {OUT_DIR}")
        log_both(f"LOG FILE                    : {run_path}")
        if STORE_LAMBDA_TEST_CURVE:
            log_both(f"CURVE CSV                   : {curve_csv_path}")
        log_both("========================================")
        log_both("")

        acc_store = {}
        loss_store = {}
        lam_store = {}

        methods = ["GAP", "WGAP", "reg_WGAP"]

        print("=== PROGRESS (per dataset) ===", flush=True)

        for DATASET_NAME in DATASETS:
            log_both(f"==================== DATASET: {DATASET_NAME} ====================")

            for method in methods:
                acc_store[(DATASET_NAME, method)] = []
                loss_store[(DATASET_NAME, method)] = []

            lam_store[DATASET_NAME] = []

            for seed in SEEDS:
                set_seed(seed)

                x_train, y_train, x_test, y_test, train_path, test_path = load_ucr_like(DATASET_NAME, BASE_DIR)

                if not NO_NORM:
                    x_train = z_normalize_per_series(x_train)
                    x_test = z_normalize_per_series(x_test)

                Ntr, T, C = x_train.shape
                n_classes = int(np.max(y_train) + 1)
                batch_size = min(int(BATCH_SIZE), Ntr)

                train_loader, test_loader = make_loaders(
                    x_train, y_train, x_test, y_test,
                    batch_size=batch_size,
                    num_workers=NUM_WORKERS
                )

                log_both(f"[{DATASET_NAME}] seed={seed}")
                log_both(f"  train_path={train_path}")
                log_both(f"  test_path ={test_path}")
                log_both(f"  Ntr={Ntr}, T={T}, C={C}, K={n_classes}, batch_size={batch_size}")

                # GAP
                log_both("  Training GAP (select epoch by minimum train loss)...")
                set_seed(seed)
                model_gap = FCN_Pooling(
                    n_classes=n_classes,
                    T=T,
                    in_channels=C,
                    pooling="gap"
                )

                _, te_gap = train_model(
                    model_gap,
                    train_loader,
                    test_loader,
                    device,
                    epochs=EPOCHS,
                    lr=LR,
                    lambda_reg=0.0,
                    select_by="train_loss",
                    eval_every=1,
                    early_stop_patience=None,
                    use_amp=use_amp,
                    verbose=VERBOSE,
                    log_fn=log_both if VERBOSE else None,
                )

                acc_store[(DATASET_NAME, "GAP")].append(te_gap["acc"])
                loss_store[(DATASET_NAME, "GAP")].append(te_gap["loss"])

                log_both(
                    f"  GAP        | best_epoch={te_gap['best_epoch']} | "
                    f"acc_test={te_gap['acc']:.4f} | loss_test={te_gap['loss']:.5f}"
                )

                # WGAP
                log_both("  Training unconstrained WGAP (select epoch by minimum train loss)...")
                set_seed(seed)
                model_wgap = FCN_Pooling(
                    n_classes=n_classes,
                    T=T,
                    in_channels=C,
                    pooling="wgap"
                )

                _, te_wgap = train_model(
                    model_wgap,
                    train_loader,
                    test_loader,
                    device,
                    epochs=EPOCHS,
                    lr=LR,
                    lambda_reg=0.0,
                    select_by="train_loss",
                    eval_every=1,
                    early_stop_patience=None,
                    use_amp=use_amp,
                    verbose=VERBOSE,
                    log_fn=log_both if VERBOSE else None,
                )

                acc_store[(DATASET_NAME, "WGAP")].append(te_wgap["acc"])
                loss_store[(DATASET_NAME, "WGAP")].append(te_wgap["loss"])

                log_both(
                    f"  WGAP       | best_epoch={te_wgap['best_epoch']} | "
                    f"acc_test={te_wgap['acc']:.4f} | loss_test={te_wgap['loss']:.5f}"
                )

                # CV lambda selection
                log_both(f"  CV (k={K_FOLDS}) for reg_WGAP over lambda_grid={LAMBDA_GRID} ...")

                best_lam, best_cv_acc, cv_scores = crossval_select_lambda_wgap_reg(
                    x_train,
                    y_train,
                    T=T,
                    C=C,
                    n_classes=n_classes,
                    device=device,
                    lambda_grid=LAMBDA_GRID,
                    k_folds=K_FOLDS,
                    seed=seed,
                    epochs=CV_EPOCHS,
                    lr=LR,
                    batch_size=batch_size,
                    num_workers=NUM_WORKERS,
                    early_stop_patience=CV_EARLY_STOP_PATIENCE,
                    use_amp=use_amp,
                    verbose=True,
                    log_fn=log_both,
                )

                lam_store[DATASET_NAME].append(best_lam)
                log_both(f"  -> selected lambda={best_lam:g} | mean CV acc={best_cv_acc:.4f}")

                # Final reg_WGAP
                log_both(f"  Training reg_WGAP with selected lambda={best_lam:g} (select epoch by max acc_test)...")
                set_seed(seed)
                model_wgap_reg = FCN_Pooling(
                    n_classes=n_classes,
                    T=T,
                    in_channels=C,
                    pooling="wgap_reg"
                )

                _, te_wgap_reg = train_model(
                    model_wgap_reg,
                    train_loader,
                    test_loader,
                    device,
                    epochs=EPOCHS,
                    lr=LR,
                    lambda_reg=float(best_lam),
                    select_by="eval_acc",
                    eval_every=1,
                    early_stop_patience=EPOCHS + 1,
                    use_amp=use_amp,
                    verbose=VERBOSE,
                    log_fn=log_both if VERBOSE else None,
                )

                acc_store[(DATASET_NAME, "reg_WGAP")].append(te_wgap_reg["acc"])
                loss_store[(DATASET_NAME, "reg_WGAP")].append(te_wgap_reg["loss"])

                log_both(
                    f"  reg_WGAP   | lambda={best_lam:g} | best_epoch={te_wgap_reg['best_epoch']} | "
                    f"acc_test={te_wgap_reg['acc']:.4f} | loss_test={te_wgap_reg['loss']:.5f}"
                )

                if STORE_LAMBDA_TEST_CURVE and f_curve is not None:
                    f_curve.write(
                        f"{DATASET_NAME},{seed},{float(best_lam)},{te_wgap_reg['acc']:.10f},"
                        f"{te_wgap_reg['loss']:.10f},{te_wgap_reg['best_epoch']}\n"
                    )
                    f_curve.flush()

                log_both(
                    f"[{DATASET_NAME}] seed={seed} FINAL | "
                    f"GAP acc={te_gap['acc']:.4f} loss={te_gap['loss']:.5f} | "
                    f"WGAP acc={te_wgap['acc']:.4f} loss={te_wgap['loss']:.5f} | "
                    f"reg_WGAP(lambda={best_lam:g}) acc={te_wgap_reg['acc']:.4f} loss={te_wgap_reg['loss']:.5f}"
                )
                log_both("")

            # Dataset summary
            gap_acc = acc_store[(DATASET_NAME, "GAP")]
            wg_acc = acc_store[(DATASET_NAME, "WGAP")]
            rwg_acc = acc_store[(DATASET_NAME, "reg_WGAP")]

            gap_loss = loss_store[(DATASET_NAME, "GAP")]
            wg_loss = loss_store[(DATASET_NAME, "WGAP")]
            rwg_loss = loss_store[(DATASET_NAME, "reg_WGAP")]

            lambdas_str = ", ".join([f"{lam:g}" for lam in lam_store[DATASET_NAME]])

            log_both(f"--- DATASET SUMMARY: {DATASET_NAME} ---")
            log_both(f"Selected lambdas (reg_WGAP, all seeds): {lambdas_str}")
            log_both(
                f"GAP      | acc_test = {fmt_mean_sd(gap_acc)} | "
                f"loss_test = {fmt_mean_sd(gap_loss)}"
            )
            log_both(
                f"WGAP     | acc_test = {fmt_mean_sd(wg_acc)} | "
                f"loss_test = {fmt_mean_sd(wg_loss)}"
            )
            log_both(
                f"reg_WGAP | acc_test = {fmt_mean_sd(rwg_acc)} | "
                f"loss_test = {fmt_mean_sd(rwg_loss)}"
            )
            log_both("")

            print(
                f"[DATASET DONE] {DATASET_NAME:<30} | "
                f"GAP={fmt_mean_sd(gap_acc)} | "
                f"WGAP={fmt_mean_sd(wg_acc)} | "
                f"reg_WGAP={fmt_mean_sd(rwg_acc)} | "
                f"lambdas=[{lambdas_str}]",
                flush=True
            )

        # Final table
        lines = []
        lines.append(
            f"{'dataset':<30} {'GAP':<18} {'WGAP':<18} {'reg_WGAP':<18} {'lambdaS'}"
        )
        lines.append("-" * 120)

        for DATASET_NAME in DATASETS:
            gap_str = fmt_mean_sd(acc_store[(DATASET_NAME, "GAP")])
            wg_str = fmt_mean_sd(acc_store[(DATASET_NAME, "WGAP")])
            rwg_str = fmt_mean_sd(acc_store[(DATASET_NAME, "reg_WGAP")])
            lambdas_str = ", ".join([f"{lam:g}" for lam in lam_store[DATASET_NAME]])

            lines.append(
                f"{DATASET_NAME:<30} "
                f"{gap_str:<18} "
                f"{wg_str:<18} "
                f"{rwg_str:<18} "
                f"{lambdas_str}"
            )

        table_txt = "\n".join(lines) + "\n"

        log_both("")
        log_both("=== FINAL SUMMARY TABLE ===")
        log_both("GAP | WGAP | reg_WGAP | lambdaS")
        log_both("acc ± sd | acc ± sd | acc ± sd | all selected lambdas")
        log_both("")
        log_both(table_txt)

        log_both(f"[saved] {run_path}")
        if STORE_LAMBDA_TEST_CURVE:
            log_both(f"[saved] {curve_csv_path}")

    except Exception:
        print("\n=== UNCAUGHT EXCEPTION ===", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        try:
            f_run.close()
        except Exception:
            pass

        if f_curve is not None:
            try:
                f_curve.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()