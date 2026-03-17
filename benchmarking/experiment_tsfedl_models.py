"""
Experiment script for TSFEDL time-series anomaly detection models.

Uses all 23 TSFEDL models available in the RADAR platform, evaluated on
two UCI datasets (ai4i and metro_interstate), following the same structure
as experiment_transformers_models.py.

Each TSFEDL model is a PyTorch Lightning autoencoder.  The platform class
``TsfedlAnomalyDetection`` wraps them uniformly and exposes fit /
decision_function / predict.

Key TSFEDL parameters
---------------------
- algorithm_     : str   – model name (key into the tsfedl_algorithms dict).
- top_module     : nn.Module – a *Forecaster* head that defines the output
                   reconstruction shape.
- in_features    : int   – in the faithful TSFEDL path this means the number
                   of channels / input features for CNN models, and the
                   per-timestep feature size for GenMinxing (LSTM-only).
- input_shape    : tuple – used by YildirimOzal instead of `in_features`.
- loss           : Loss  – reconstruction loss (MSELoss for autoencoders).
- max_epochs     : int   – PyTorch-Lightning Trainer epochs.
- batch_size     : int   – DataLoader batch size.

Dimensional mapping (faithful TSFEDL mode)
------------------------------------------
    RADAR windows:        (N, window_size, n_features)
    CNN models receive:   (N, n_features, window_size)
    GenMinxing receives:  (N, window_size, n_features)

"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _time_series_experiment_utils import (
    DEFAULT_METRO_HIGH_Q,
    DEFAULT_METRO_LOW_Q,
    DEFAULT_STEP_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_WINDOW_SIZE,
    build_windowed_dataset_configs,
    build_windowed_dataset_summary,
    ensure_project_root_on_path,
    print_dataframe,
    resolve_project_root,
)

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = resolve_project_root(Path(__file__))
ensure_project_root_on_path(PROJECT_ROOT)

from RADAR.time_series.algorithms import tsfedl

# Forecaster heads (output layers) for each TSFEDL model
from TSFEDL.models_pytorch import (
    OhShuLih_Forecaster,
    YiboGao_Forecaster,
    LihOhShu_Forecaster,
    YaoQihang_Forecaster,
    HtetMyetLynn_Forecaster,
    YildirimOzal_Forecaster,
    CaiWenjuan_Forecaster,
    ZhangJin_Forecaster,
    KongZhengmin_Forecaster,
    WeiXiaoyan_Forecaster,
    GaoJunLi_Forecaster,
    KhanZulfiqar_Forecaster,
    ZhengZhenyu_Forecaster,
    WangKejun_Forecaster,
    ChenChen_Forecaster,
    KimTaeYoung_Forecaster,
    GenMinxing_Forecaster,
    FuJiangmeng_Forecaster,
    ShiHaotian_Forecaster,
    HuangMeiLing_Forecaster,
    HongTan_Forecaster,
    SharPar_Forecaster,
    DaiXiLi_Forecaster,
)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
# Minimum sequence length required by each architecture in faithful TSFEDL mode.
# In other words: minimum `window_size` so the conv/pooling stack does not
# collapse the time axis.
MODEL_MIN_WINDOW_SIZE = {
    "ohshulih": 1,      "gaojunli": 1,      "kongzhengmin": 6,
    "caiwenjuan": 8,     "wangkejun": 8,     "zhengzhenyu": 8,
    "kimtaeyoung": 5,   "fujiangmeng": 2,   "shihaotian": 15,
    "sharpar": 1,        "hongtan": 12,      "htetmyetlynn": 16,
    "liohshu": 261,      "yibogao": 1000,    "yaoqihang": 243,
    "yildirimozal": 8,   "zhangjin": 243,    "weixiaoyan": 96,
    "khanzulfiqar": 261, "chenchen": 1000,   "genminxing": 1,
    "huangmeiling": 1000, "daixili": 1000,
}

# Ordered list used as the default --models value
ALL_MODELS = [
    "ohshulih",     "gaojunli",      "kongzhengmin",
    "caiwenjuan",   "wangkejun",     "zhengzhenyu",
    "kimtaeyoung",  "fujiangmeng",   "shihaotian",
    "sharpar",      "hongtan",       "htetmyetlynn",
    "liohshu",      "yibogao",       "yaoqihang",
    "yildirimozal", "zhangjin",      "weixiaoyan",
    "khanzulfiqar", "chenchen",      "genminxing",
    "huangmeiling", "daixili",
]

# Maps each model key → its Forecaster class and any special kwargs
FORECASTER_REGISTRY: dict[str, dict] = {
    "ohshulih":      {"forecaster": OhShuLih_Forecaster},
    "gaojunli":      {"forecaster": GaoJunLi_Forecaster},
    "kongzhengmin":  {"forecaster": KongZhengmin_Forecaster},
    "caiwenjuan":    {"forecaster": CaiWenjuan_Forecaster},
    "wangkejun":     {"forecaster": WangKejun_Forecaster},
    "zhengzhenyu":   {"forecaster": ZhengZhenyu_Forecaster,  "top_extra": {"in_features": 256}},
    "kimtaeyoung":   {"forecaster": KimTaeYoung_Forecaster},
    "fujiangmeng":   {"forecaster": FuJiangmeng_Forecaster},
    "shihaotian":    {"forecaster": ShiHaotian_Forecaster},
    "sharpar":       {"forecaster": SharPar_Forecaster},
    "hongtan":       {"forecaster": HongTan_Forecaster},
    "htetmyetlynn":  {"forecaster": HtetMyetLynn_Forecaster},
    "liohshu":       {"forecaster": LihOhShu_Forecaster},
    "yibogao":       {"forecaster": YiboGao_Forecaster},
    "yaoqihang":     {"forecaster": YaoQihang_Forecaster},
    "yildirimozal":  {"forecaster": YildirimOzal_Forecaster, "top_extra": {"in_features": 32}},
    "zhangjin":      {"forecaster": ZhangJin_Forecaster,     "top_extra": {"in_features": 24}},
    "weixiaoyan":    {"forecaster": WeiXiaoyan_Forecaster},
    "khanzulfiqar":  {"forecaster": KhanZulfiqar_Forecaster},
    "chenchen":      {"forecaster": ChenChen_Forecaster},
    "genminxing":    {"forecaster": GenMinxing_Forecaster},
    "huangmeiling":  {"forecaster": HuangMeiLing_Forecaster},
    "daixili":       {"forecaster": DaiXiLi_Forecaster},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy(values):
    """Safely convert torch tensors to numpy arrays."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def summarize_mse(scores):
    """Mean of finite scores, or NaN if none are finite."""
    scores = np.asarray(scores, dtype=float).ravel()
    finite = scores[np.isfinite(scores)]
    return float(finite.mean()) if finite.size else np.nan


def split_direct_model_and_trainer_params(model_params: dict) -> tuple[dict, dict]:
    """Split RADAR wrapper params into direct-model kwargs and Trainer kwargs."""
    trainer_signature = signature(pl.Trainer.__init__)
    trainer_param_names = {name for name in trainer_signature.parameters if name != "self"}

    trainer_kwargs = {}
    direct_model_kwargs = {}
    for key, value in model_params.items():
        if key in trainer_param_names:
            trainer_kwargs[key] = value
        elif key not in {"algorithm_", "batch_size"}:
            direct_model_kwargs[key] = value

    return direct_model_kwargs, trainer_kwargs


def build_direct_tsfedl_model(model_name: str, direct_model_kwargs: dict):
    """Instantiate the TSFEDL model class directly, without the RADAR wrapper."""
    algorithm_cls = tsfedl.tsfedl_algorithms[model_name]
    return algorithm_cls(**direct_model_kwargs)


def train_direct_tsfedl_model(
    direct_model,
    X_train_tensor: torch.Tensor,
    batch_size: int,
    trainer_kwargs: dict,
):
    """Train a TSFEDL model directly through PyTorch Lightning."""
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(direct_model, train_dataloaders=train_loader)
    return direct_model


def compute_direct_scores(direct_model, X_test_tensor: torch.Tensor) -> np.ndarray:
    """Compute per-sample reconstruction MSE for a direct TSFEDL model."""
    direct_model.eval()
    try:
        device = next(direct_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    X_test_device = X_test_tensor.to(device)
    with torch.no_grad():
        predictions = direct_model(X_test_device)
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        predictions = predictions.detach().cpu()
        targets = X_test_tensor.detach().cpu()

        if predictions.shape == targets.shape:
            errors = (predictions - targets) ** 2
            reduction_dims = tuple(range(1, errors.ndim))
            return torch.mean(errors, dim=reduction_dims).numpy()

        predictions_flat = predictions.reshape(predictions.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)
        min_width = min(predictions_flat.shape[1], targets_flat.shape[1])
        errors = (predictions_flat[:, :min_width] - targets_flat[:, :min_width]) ** 2
        return torch.mean(errors, dim=1).numpy()


def average(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list, else NaN."""
    return float(np.mean(values)) if values else np.nan


def get_model_layout(model_name: str) -> str:
    """Return the data layout expected by the faithful experiment path."""
    if model_name == "genminxing":
        return "lstm"
    if model_name == "yildirimozal":
        return "yildirim"
    return "cnn"


def prepare_model_tensors(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert RADAR windows to the layout expected by each TSFEDL model.

    RADAR windows are `(N, window_size, n_features)`.
    - CNN TSFEDL models expect `(N, n_features, window_size)`.
    - GenMinxing expects `(N, window_size, n_features)`.
    - YildirimOzal also expects CNN-style `(N, n_features, window_size)`.
    """
    layout = get_model_layout(model_name)
    if layout in {"cnn", "yildirim"}:
        return (
            np.ascontiguousarray(np.transpose(X_train, (0, 2, 1))),
            np.ascontiguousarray(np.transpose(X_test, (0, 2, 1))),
        )
    return np.ascontiguousarray(X_train), np.ascontiguousarray(X_test)


class ReduceTimeForecasterAdapter(nn.Module):
    """Reduce a temporal dimension before delegating to a TSFEDL forecaster."""

    def __init__(self, forecaster: nn.Module, reduce: str = "last"):
        super().__init__()
        self.forecaster = forecaster
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if self.reduce == "mean":
                x = x.mean(dim=1)
            else:
                x = x[:, -1, :]
        return self.forecaster(x)


@lru_cache(maxsize=None)
def infer_backbone_output_shape(
    model_name: str,
    input_dim: int,
    seq_len: int,
) -> tuple[int, ...]:
    """Infer the raw backbone output shape for one TSFEDL model in faithful mode."""
    algorithm_cls = tsfedl.tsfedl_algorithms[model_name]
    layout = get_model_layout(model_name)

    model_kwargs: dict = {"top_module": None}
    if model_name == "yildirimozal":
        model_kwargs["input_shape"] = (input_dim, seq_len)
        dummy = torch.zeros(2, input_dim, seq_len, dtype=torch.float32)
    elif layout == "lstm":
        model_kwargs["in_features"] = input_dim
        dummy = torch.zeros(2, seq_len, input_dim, dtype=torch.float32)
    else:
        model_kwargs["in_features"] = input_dim
        dummy = torch.zeros(2, input_dim, seq_len, dtype=torch.float32)

    model = algorithm_cls(**model_kwargs)
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    return tuple(output.shape)


# ---------------------------------------------------------------------------
# Build model parameters
# ---------------------------------------------------------------------------

def build_model_params(
    model_name: str,
    input_dim: int,
    seq_len: int,
    max_epochs: int,
    batch_size: int,
) -> dict:
    """
    Build the kwargs dict for ``TsfedlAnomalyDetection(**params)``.

    Parameters
    ----------
    model_name : str
        Key from ALL_MODELS / tsfedl_algorithms.
    input_dim : int
        Number of features (columns) per time step.
    seq_len : int
        Window size (number of time steps per sample).
    max_epochs : int
        Training epochs for the PyTorch Lightning Trainer.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict ready to pass to ``TsfedlAnomalyDetection(**params)``.
    """
    registry = FORECASTER_REGISTRY[model_name]
    forecaster_cls = registry["forecaster"]

    layout = get_model_layout(model_name)

    # -- Build the top_module (Forecaster head) --
    # Faithful reconstruction targets:
    # - CNN models reconstruct (N, n_features, window_size)  -> n_pred=n_features, out_features=window_size
    # - GenMinxing reconstructs (N, window_size, n_features) -> n_pred=window_size, out_features=n_features
    if layout == "cnn":
        top_kwargs = {"out_features": seq_len, "n_pred": input_dim}
    else:
        top_kwargs = {"out_features": input_dim, "n_pred": seq_len}

    backbone_shape = None
    if model_name in {"yibogao", "zhangjin", "huangmeiling"}:
        backbone_shape = infer_backbone_output_shape(model_name, input_dim, seq_len)

    if model_name in {"yibogao", "huangmeiling"} and backbone_shape is not None:
        top_kwargs["in_features"] = backbone_shape[-1]

    if model_name == "zhangjin" and backbone_shape is not None:
        top_kwargs["in_features"] = backbone_shape[-1]

    top_kwargs.update(registry.get("top_extra", {}))
    top_module = forecaster_cls(**top_kwargs)

    if model_name == "zhangjin":
        top_module = ReduceTimeForecasterAdapter(top_module, reduce="last")

    # -- Model kwargs for TsfedlAnomalyDetection --
    params = {
        "algorithm_": model_name,
        "loss": torch.nn.MSELoss(),
        "top_module": top_module,
        "in_features": input_dim,
        "batch_size": batch_size,
        # Trainer kwargs (forwarded via pytorch_params_)
        "max_epochs": max_epochs,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }

    # Special cases
    if model_name == "yildirimozal":
        params["input_shape"] = (input_dim, seq_len)
        del params["in_features"]
    if model_name == "genminxing":
        params["in_features"] = input_dim

    return params


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train every selected TSFEDL model on every selected dataset."""

    # 1. Prepare windowed datasets
    dataset_configs = build_windowed_dataset_configs(
        window_size=args.window_size,
        step_size=args.step_size,
        test_size=args.test_size,
        low_q=args.metro_low_q,
        high_q=args.metro_high_q,
        dataset_keys=args.datasets,
    )
    dataset_summary = build_windowed_dataset_summary(dataset_configs)
    print_dataframe(dataset_summary, title="Dataset summary")

    results = []
    timing_results = []

    # 2. Iterate datasets × models
    for dataset_key, config in dataset_configs.items():
        raw_features = config["X_train_windows"].shape[2]
        seq_len = config["window_size"]

        print(f"\n{'='*60}")
        print(f"Dataset: {config['dataset']}")
        print(f"  Features: {raw_features} | Window: {seq_len}")
        print(f"  Train windows: {config['train_windows']} | Test windows: {config['test_windows']}")
        print(f"{'='*60}")

        for idx, model_name in enumerate(args.models, start=1):
            min_window = MODEL_MIN_WINDOW_SIZE.get(model_name, 1)
            layout = get_model_layout(model_name)

            # --- Decide if we skip ---
            if seq_len < min_window:
                print(f"  [{idx:02d}] {model_name:18s} -> SKIPPED (needs window >= {min_window}, has {seq_len})")
                results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "window_size": seq_len,
                    "n_features": raw_features,
                    "layout": layout,
                    "status": "skipped",
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "train_time_s": np.nan,
                    "inference_time_s": np.nan,
                    "mse": np.nan,
                    "error_msg": f"Needs window_size >= {min_window}",
                })
                if args.run_timing:
                    timing_results.append({
                        "dataset_key": dataset_key,
                        "dataset_name": config["dataset"],
                        "algorithm": model_name,
                        "timing_repetitions": 1,
                        "platform_status": "skipped",
                        "direct_status": "skipped",
                        "platform_time_s": np.nan,
                        "direct_time_s": np.nan,
                        "overhead_s": np.nan,
                        "speedup_direct_over_platform": np.nan,
                        "platform_mse": np.nan,
                        "direct_mse": np.nan,
                        "mse_diff": np.nan,
                        "platform_error_msg": f"Needs window_size >= {min_window}",
                        "direct_error_msg": f"Needs window_size >= {min_window}",
                    })
                continue

            if model_name == "yildirimozal" and raw_features != 1:
                print(f"  [{idx:02d}] {model_name:18s} -> SKIPPED (faithful autoencoder path is single-channel; dataset has {raw_features})")
                results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "window_size": seq_len,
                    "n_features": raw_features,
                    "layout": layout,
                    "status": "skipped",
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "train_time_s": np.nan,
                    "inference_time_s": np.nan,
                    "mse": np.nan,
                    "error_msg": "YildirimOzal faithful autoencoder currently supports only single-channel reconstruction",
                })
                if args.run_timing:
                    timing_results.append({
                        "dataset_key": dataset_key,
                        "dataset_name": config["dataset"],
                        "algorithm": model_name,
                        "timing_repetitions": 1,
                        "platform_status": "skipped",
                        "direct_status": "skipped",
                        "platform_time_s": np.nan,
                        "direct_time_s": np.nan,
                        "overhead_s": np.nan,
                        "speedup_direct_over_platform": np.nan,
                        "platform_mse": np.nan,
                        "direct_mse": np.nan,
                        "mse_diff": np.nan,
                        "platform_error_msg": "YildirimOzal faithful autoencoder currently supports only single-channel reconstruction",
                        "direct_error_msg": "YildirimOzal faithful autoencoder currently supports only single-channel reconstruction",
                    })
                continue

            if model_name == "daixili" and seq_len != 1000:
                print(f"  [{idx:02d}] {model_name:18s} -> SKIPPED (architecture is hard-coded for window_size=1000; got {seq_len})")
                results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "window_size": seq_len,
                    "n_features": raw_features,
                    "layout": layout,
                    "status": "skipped",
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "train_time_s": np.nan,
                    "inference_time_s": np.nan,
                    "mse": np.nan,
                    "error_msg": "DaiXiLi uses Linear layers hard-coded for window_size=1000",
                })
                if args.run_timing:
                    timing_results.append({
                        "dataset_key": dataset_key,
                        "dataset_name": config["dataset"],
                        "algorithm": model_name,
                        "timing_repetitions": 1,
                        "platform_status": "skipped",
                        "direct_status": "skipped",
                        "platform_time_s": np.nan,
                        "direct_time_s": np.nan,
                        "overhead_s": np.nan,
                        "speedup_direct_over_platform": np.nan,
                        "platform_mse": np.nan,
                        "direct_mse": np.nan,
                        "mse_diff": np.nan,
                        "platform_error_msg": "DaiXiLi uses Linear layers hard-coded for window_size=1000",
                        "direct_error_msg": "DaiXiLi uses Linear layers hard-coded for window_size=1000",
                    })
                continue

            X_train, X_test = prepare_model_tensors(
                model_name=model_name,
                X_train=config["X_train_windows"],
                X_test=config["X_test_windows"],
            )
            layout_note = " (CNN faithful layout)" if layout == "cnn" else " (LSTM layout)"

            print(f"  [{idx:02d}] {model_name:18s}{layout_note} ... ", end="", flush=True)
            platform_status = "pending"
            platform_error_msg = None
            direct_status = "pending"
            direct_error_msg = None
            direct_time = np.nan
            direct_mse = np.nan
            train_time = np.nan
            mse = np.nan

            # --- Create, train, and score ---
            try:
                model_params = build_model_params(
                    model_name, raw_features, seq_len,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                )
                model = tsfedl.TsfedlAnomalyDetection(**model_params)
                X_train_tensor = torch.from_numpy(np.ascontiguousarray(X_train)).to(torch.float32).contiguous()
                X_test_tensor = torch.from_numpy(np.ascontiguousarray(X_test)).to(torch.float32).contiguous()

                # Train
                t0 = time.time()
                model.fit(X_train_tensor, X_train_tensor)
                train_time = time.time() - t0

                # Inference (decision_function → MSE per sample)
                t0 = time.time()
                scores = tensor_to_numpy(model.decision_function(X_test_tensor)).ravel()
                inference_time = time.time() - t0

                finite = bool(np.isfinite(scores).all())
                mse = summarize_mse(scores) if finite else np.nan
                platform_status = "ok"

                if np.isfinite(mse):
                    print(f"MSE={mse:.6f}  train={train_time:.2f}s  infer={inference_time:.2f}s")
                else:
                    print(f"MSE=nan (non-finite scores)  train={train_time:.2f}s")

                results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "window_size": seq_len,
                    "n_features": raw_features,
                    "layout": layout,
                    "status": "ok",
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "train_time_s": round(train_time, 4),
                    "inference_time_s": round(inference_time, 4),
                    "mse": round(float(mse), 6) if np.isfinite(mse) else np.nan,
                    "error_msg": None,
                })

            except Exception as exc:
                print(f"FAILED -> {exc}")
                platform_status = "failed"
                platform_error_msg = str(exc).strip() or exc.__class__.__name__
                results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "window_size": seq_len,
                    "n_features": raw_features,
                    "layout": layout,
                    "status": "failed",
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "train_time_s": np.nan,
                    "inference_time_s": np.nan,
                    "mse": np.nan,
                    "error_msg": platform_error_msg,
                })

            if args.run_timing:
                if platform_status == "ok":
                    try:
                        platform_timing_runs = [float(train_time)]
                        for _ in range(max(args.timing_repetitions - 1, 0)):
                            timing_model = tsfedl.TsfedlAnomalyDetection(**model_params)
                            platform_start = time.perf_counter()
                            timing_model.fit(X_train_tensor, X_train_tensor)
                            platform_timing_runs.append(time.perf_counter() - platform_start)

                        train_time = average(platform_timing_runs)

                        direct_model_kwargs, trainer_kwargs = split_direct_model_and_trainer_params(model_params)
                        direct_timing_runs = []
                        direct_model = None
                        for _ in range(args.timing_repetitions):
                            direct_model = build_direct_tsfedl_model(model_name, direct_model_kwargs)
                            direct_start = time.perf_counter()
                            train_direct_tsfedl_model(
                                direct_model=direct_model,
                                X_train_tensor=X_train_tensor,
                                batch_size=args.batch_size,
                                trainer_kwargs=trainer_kwargs,
                            )
                            direct_timing_runs.append(time.perf_counter() - direct_start)

                        direct_time = average(direct_timing_runs)

                        direct_scores = compute_direct_scores(direct_model, X_test_tensor)
                        if np.isfinite(direct_scores).all():
                            direct_mse = summarize_mse(direct_scores)
                        direct_status = "ok"
                    except Exception as exc:
                        direct_status = "failed"
                        direct_error_msg = str(exc).strip() or exc.__class__.__name__
                else:
                    direct_status = "skipped"
                    direct_error_msg = "Platform run failed; direct timing skipped"

                timing_results.append({
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": model_name,
                    "timing_repetitions": args.timing_repetitions,
                    "platform_status": platform_status,
                    "direct_status": direct_status,
                    "platform_time_s": round(float(train_time), 4) if np.isfinite(train_time) else np.nan,
                    "direct_time_s": round(float(direct_time), 4) if np.isfinite(direct_time) else np.nan,
                    "overhead_s": round(float(train_time - direct_time), 4)
                    if np.isfinite(train_time) and np.isfinite(direct_time)
                    else np.nan,
                    "speedup_direct_over_platform": round(float(direct_time / train_time), 4)
                    if np.isfinite(train_time) and np.isfinite(direct_time) and train_time > 0
                    else np.nan,
                    "platform_mse": round(float(mse), 6) if np.isfinite(mse) else np.nan,
                    "direct_mse": round(float(direct_mse), 6) if np.isfinite(direct_mse) else np.nan,
                    "mse_diff": round(float(mse - direct_mse), 6)
                    if np.isfinite(mse) and np.isfinite(direct_mse)
                    else np.nan,
                    "platform_error_msg": platform_error_msg,
                    "direct_error_msg": direct_error_msg,
                })

    # 3. Build DataFrames
    results_df = (
        pd.DataFrame(results)
        .sort_values(["dataset_name", "mse"], ascending=[True, True], na_position="last")
        .reset_index(drop=True)
    )

    summary_df = (
        results_df[results_df["status"] == "ok"]
        .groupby(["dataset_name", "algorithm"], as_index=False)
        .agg({"mse": "min", "train_time_s": "mean", "inference_time_s": "mean"})
        .sort_values(["dataset_name", "mse"], ascending=[True, True], na_position="last")
        .reset_index(drop=True)
    )

    timing_df = pd.DataFrame(timing_results)
    if args.run_timing and not timing_df.empty:
        timing_df = timing_df.sort_values(
            ["dataset_name", "algorithm"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)

    return results_df, summary_df, timing_df



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment: evaluate all TSFEDL models on UCI time-series datasets.",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["ai4i", "metro_interstate"],
        default=["ai4i", "metro_interstate"],
        help="Datasets to evaluate on (default: both).",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="TSFEDL models to evaluate (default: all 23).",
    )
    parser.add_argument("--window-size", type=int, default=1024,
                        help="Sliding window size (default: 2048 for faithful TSFEDL mode).")
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--metro-low-q", type=float, default=DEFAULT_METRO_LOW_Q)
    parser.add_argument("--metro-high-q", type=float, default=DEFAULT_METRO_HIGH_Q)
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="PyTorch Lightning training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DataLoader batch size (default: 32).")
    parser.add_argument("--timing-repetitions", type=int, default=5,
                        help="Number of repeated training runs used to average timing overhead measurements.")
    parser.add_argument("--run-timing", action="store_true", default=True,
                        help="Also measure direct TSFEDL vs RADAR-wrapper training times.")
    parser.add_argument("--skip-timing", action="store_false", dest="run_timing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TSFEDL Models Experiment")
    print("=" * 60)
    print(f"  Datasets : {args.datasets}")
    print(f"  Models   : {len(args.models)} selected")
    print(f"  Window   : {args.window_size}  |  Epochs: {args.max_epochs}  |  Batch: {args.batch_size}")
    print("=" * 60)

    results_df, summary_df, timing_df = run_experiment(args)

    print_dataframe(results_df, title="Detailed results")
    print_dataframe(summary_df, title="Summary (best MSE per dataset × model)")

    results_path = args.results_dir / "uci_tsfedl_results.csv"
    summary_path = args.results_dir / "uci_tsfedl_summary.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if args.run_timing:
        timing_path = args.results_dir / "uci_tsfedl_timing_results.csv"
        print_dataframe(timing_df, title="Timing comparison (RADAR vs direct TSFEDL)")
        timing_df.to_csv(timing_path, index=False)

    print(f"\nSaved results  -> {results_path}")
    print(f"Saved summary  -> {summary_path}")
    if args.run_timing:
        print(f"Saved timing   -> {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
