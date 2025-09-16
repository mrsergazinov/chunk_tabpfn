#!/usr/bin/env python3
import argparse
import json
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabrepo.benchmark.task.openml import OpenMLTaskWrapper
import openml

# --- Your local TabPFN repo ---
import sys
import os
sys.path.append(os.environ.get("PYTHONPATH", ".") + "/src")
from tabpfn.classifier import TabPFNClassifier
from tabpfn.regressor import TabPFNRegressor
from tabpfn.extensions.replace_chunk_attn_computation import enable_chunked_attention

# ------------------------------------------------------------------
# Feature encoding (safe for unseen categories)
# ------------------------------------------------------------------
def fit_feature_encoders(df: pd.DataFrame):
    encoders = {}
    for c in df.columns:
        if df[c].dtype in ("object", "category"):
            # Map known categories to ints
            cats = pd.Categorical(df[c].astype(str)).categories.tolist()
            encoders[c] = {"cats": {v: i for i, v in enumerate(cats)}}
    return encoders

def transform_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if c in encoders:
            mapping = encoders[c]["cats"]
            # Unknown -> new index
            ser = df[c].astype(str).map(mapping)
            unk_mask = ser.isna()
            if unk_mask.any():
                start = len(mapping)
                # assign incremental ids to unknowns
                unknown_vals = df[c].astype(str)[unk_mask].unique().tolist()
                for v in unknown_vals:
                    mapping[v] = start
                    start += 1
                ser = df[c].astype(str).map(mapping)
            cols.append(ser.astype("int64").rename(c))
        else:
            cols.append(df[c])
    return pd.concat(cols, axis=1)

# ------------------------------------------------------------------
# Target encoding
# ------------------------------------------------------------------
def encode_target(y_train_raw, y_test_raw, problem_type):
    if problem_type == "regression":
        return y_train_raw, y_test_raw, None  # no encoder
    le = LabelEncoder().fit(y_train_raw)
    y_tr = le.transform(y_train_raw)
    y_te = le.transform(y_test_raw)
    return y_tr, y_te, le

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def metrics_classification_binary(y_true_enc, proba):
    pos = proba[:, 1]
    auc = roc_auc_score(y_true_enc, pos)
    auc = 1 - auc if auc < 0.5 else auc
    preds = (pos >= 0.5).astype(int)
    acc = accuracy_score(y_true_enc, preds)
    return {"auc": auc, "acc": acc}

def metrics_classification_multiclass(y_true_enc, proba):
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y_true_enc, preds)
    return {"acc": acc}

def metrics_regression(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2": r2}

# ------------------------------------------------------------------
# Misc utils
# ------------------------------------------------------------------
def get_gpu_mem_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        m = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
        return m
    return np.nan

def build_grid(n_rows: int):
    if n_rows <= 1000:
        return [n_rows]
    if n_rows <= 10000:
        return list(range(1000, n_rows + 1, 1000))
    grid = list(range(1000, 10001, 1000)) + list(range(20000, n_rows + 1, 10000))
    return sorted(set(grid))

def get_dataset_meta(task_id, tasks_entry=None):
    if tasks_entry and "dataset_name" in tasks_entry and "dataset_size" in tasks_entry:
        return tasks_entry["dataset_name"], tasks_entry["dataset_size"]
    task = OpenMLTaskWrapper.from_task_id(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    return dataset.name, len(task.X)

# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------
def run_single(task_id: int,
               context_len: int,
               use_chunk: bool,
               chunk_size: int,
               seed: int,
               device: str,
               out_csv: Path,
               tasks_entry=None):

    task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
    X, y = task.X, task.y
    problem_type = task.problem_type

    # split
    X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if problem_type != "regression" else None
    )

    # enforce same order after slicing
    if context_len > 0:
        X_tr = X_tr.iloc[:context_len].copy().reset_index(drop=True)
        y_tr_raw = y_tr_raw.iloc[:context_len].copy().reset_index(drop=True)
    else:
        X_tr = X_tr.reset_index(drop=True)
        y_tr_raw = y_tr_raw.reset_index(drop=True)

    X_te = X_te.reset_index(drop=True)
    y_te_raw = y_te_raw.reset_index(drop=True)

    # feature encoding
    encoders = fit_feature_encoders(X_tr)
    X_tr = transform_features(X_tr, encoders)
    X_te = transform_features(X_te, encoders)

    # target encoding / normalization
    y_tr, y_te, le_y = encode_target(y_tr_raw, y_te_raw, problem_type)
    y_mean = y_std = None
    if problem_type == "regression":
        y_mean, y_std = np.mean(y_tr), np.std(y_tr) or 1.0
        y_tr = (y_tr - y_mean) / y_std

    # chunked attention
    if use_chunk:
        enable_chunked_attention(chunk_size=chunk_size)
        
    # model
    if problem_type in ("binary", "multiclass"):
        model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    else:
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

    # fit
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.fit(X_tr.values, y_tr)
    fit_time = time.perf_counter() - t0
    fit_mem = get_gpu_mem_mb()

    # eval
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if problem_type == "regression":
        y_pred = model.predict(X_te.values) * y_std + y_mean
        metrics = metrics_regression(y_te_raw.values, y_pred)
    else:
        proba = model.predict_proba(X_te.values)
        if problem_type == "binary":
            metrics = metrics_classification_binary(y_te, proba)
        else:
            metrics = metrics_classification_multiclass(y_te, proba)
    eval_time = time.perf_counter() - t0
    eval_mem = get_gpu_mem_mb()

    dataset_name, _ = get_dataset_meta(task_id, tasks_entry)

    row = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "problem_type": problem_type,
        "context_len": context_len,
        "use_chunk": int(use_chunk),
        "chunk_size": chunk_size,
        "seed": seed,
        "fit_time_s": fit_time,
        "eval_time_s": eval_time,
        "fit_mem_mb": fit_mem,
        "eval_mem_mb": eval_mem,
        "auc": 0,
        "acc": 0,
        "r2": 0,
        "rmse": 0,
    }
    row.update(metrics)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--tasks-json", type=str, default="tasks_long.json")
    ap.add_argument("--context-lens", type=str, default="auto")
    ap.add_argument("--use-chunk", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-csv", type=str, default="results.csv")
    args = ap.parse_args()

    tasks_entry = None
    if args.task_id is None:
        assert args.task_index is not None, "Provide --task-id or --task-index"
        tasks = json.load(open(args.tasks_json))
        tasks_entry = tasks[args.task_index]
        task_id = tasks_entry["task_id"]
        dataset_size = tasks_entry["dataset_size"]
    else:
        task_id = args.task_id
        tmp = OpenMLTaskWrapper.from_task_id(task_id=task_id)
        dataset_size = len(tmp.X)

    # context grid
    if args.context_lens.strip().lower() == "auto":
        grid = build_grid(dataset_size)
    else:
        grid = [int(x) for x in args.context_lens.split(",") if x.strip()]

    print(f"Task {task_id} size={dataset_size}; grid={grid}; chunk={args.use_chunk}")

    out_csv = Path(args.out_csv)
    for L in grid:
        run_single(task_id, L, args.use_chunk, args.chunk_size, args.seed, args.device, out_csv, tasks_entry)


if __name__ == "__main__":
    main()
