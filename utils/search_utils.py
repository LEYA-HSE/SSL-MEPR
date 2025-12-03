# coding: utf-8

import copy
import os
import logging
from itertools import product
from typing import Any

import numpy as np

# ── Metric display order (priority first, then secondary) ───────────
METRIC_ORDER = [
    "mean_combo",
    "mean_emo", "mUAR", "mF1",          # emotion block
    "mean_pkl", "ACC", "CCC",           # personality block
]

# ──────────────────────────── Helpers ───────────────────────────────
def _pick_score(metrics: dict, metric_name: str = "mean_emo") -> float:
    """Safely extract a metric value for selection, otherwise return 0."""
    return float(metrics.get(metric_name, 0.0))


def format_result_box_dual(step_num: int,
                           param_name: str,
                           candidate: Any,
                           fixed_params: dict[str, Any],
                           dev_metrics: dict[str, Any],
                           test_metrics: dict[str, Any],
                           is_best: bool = False,
                           selection_metric: str = "mean_emo",
                           early_stop_on: str = "dev") -> str:
    """Pretty ASCII box for displaying dev/test results."""
    title = f"Step {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]

    # Add mean_combo
    if "mean_emo" in dev_metrics and "mean_pkl" in dev_metrics:
        dev_metrics["mean_combo"] = 0.5 * (dev_metrics["mean_emo"] + dev_metrics["mean_pkl"])
    if "mean_emo" in test_metrics and "mean_pkl" in test_metrics:
        test_metrics["mean_combo"] = 0.5 * (test_metrics["mean_emo"] + test_metrics["mean_pkl"])

    def format_metrics_block(metrics: dict[str, Any], label: str) -> list[str]:
        lines = [f"  Results ({label.upper()}):"]
        ordered = METRIC_ORDER + sorted(set(metrics) - set(METRIC_ORDER))
        for k in ordered:
            if k in metrics:
                val = metrics[k]
                line = (f"    {k.upper():12} = {val:.4f}"
                        if isinstance(val, (int, float)) else
                        f"    {k.upper():12} = {val}")
                if is_best and label == early_stop_on and k == selection_metric:
                    line += " ✅"
                lines.append(line)
        return lines

    content_lines = [title, "  Fixed parameters:"]
    content_lines += [f"    {line}" for line in fixed_lines]
    content_lines += format_metrics_block(dev_metrics, "dev")
    content_lines.append("")
    content_lines += format_metrics_block(test_metrics, "test")

    max_width = max(len(line) for line in content_lines)
    border_top = "┌" + "─" * (max_width + 2) + "┐"
    border_bot = "└" + "─" * (max_width + 2) + "┘"

    box = [border_top]
    for line in content_lines:
        box.append(f"│ {line.ljust(max_width)} │")
    box.append(border_bot)
    return "\n".join(box)


# ─────────────────────────── Greedy search ──────────────────────────
def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
):
    current_best_params = copy.deepcopy(default_values)
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")
    selection_metric = getattr(base_config, "selection_metric", "mean_emo")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Greedy hyperparameter search (Dev-based) ===\n")
        f.write(f"Model: {model_name}\n")

    for i, param_name in enumerate(all_param_names):
        candidates = param_grid[param_name]
        tried_value = current_best_params[param_name]
        candidates_now = candidates if i == 0 else [v for v in candidates if v != tried_value]

        best_val_for_param = tried_value
        best_metric_for_param = float("-inf")

        # Evaluate default value (from second step onward)
        if i != 0:
            cfg_def = copy.copy(base_config)
            for k, v in current_best_params.items():
                setattr(cfg_def, k, v)

            combo_dir = os.path.join(base_config.checkpoint_dir,
                                     f"greedy_{param_name}_{tried_value}")
            os.makedirs(combo_dir, exist_ok=True)
            cfg_def.checkpoint_dir = combo_dir

            dev_met_def, test_met_def = train_fn(cfg_def, train_loader, dev_loader, test_loader)
            dev_score = _pick_score(dev_met_def, selection_metric)

            box = format_result_box_dual(
                i + 1, param_name, tried_value,
                {k: v for k, v in current_best_params.items() if k != param_name},
                dev_met_def, test_met_def,
                is_best=True, selection_metric=selection_metric,
                early_stop_on=base_config.early_stop_on
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box + "\n")

            _log_dataset_metrics(dev_met_def, overrides_file, "dev")
            _log_dataset_metrics(test_met_def, overrides_file, "test")

            best_metric_for_param = dev_score

        # Iterate through remaining candidates
        for cand in candidates_now:
            cfg = copy.copy(base_config)
            for k, v in current_best_params.items():
                setattr(cfg, k, v)
            setattr(cfg, param_name, cand)

            logging.info(f"[STEP {i+1}] {param_name} = {cand}, others {current_best_params}")

            dev_met, test_met = train_fn(cfg, train_loader, dev_loader, test_loader)
            dev_score = _pick_score(dev_met, selection_metric)
            is_better = dev_score > best_metric_for_param

            box = format_result_box_dual(
                i + 1, param_name, cand,
                {k: v for k, v in current_best_params.items() if k != param_name},
                dev_met, test_met,
                is_best=is_better, selection_metric=selection_metric,
                early_stop_on=base_config.early_stop_on
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box + "\n")

            _log_dataset_metrics(dev_met, overrides_file, "dev")
            _log_dataset_metrics(test_met, overrides_file, "test")

            if is_better:
                best_val_for_param = cand
                best_metric_for_param = dev_score

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n>> [Step {i+1}] best {param_name}={best_val_for_param}, "
                    f"dev_{selection_metric}={best_metric_for_param:.4f}\n")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Final combination (Dev-based) ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")
    logging.info("Done! Greedy search finished.")


# ────────────────────────── Exhaustive search ────────────────────────
def exhaustive_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
):
    all_param_names = list(param_grid.keys())
    selection_metric = base_config.selection_metric

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Exhaustive hyperparameter search (Dev-based) ===\n")
        f.write(f"Model: {base_config.model_name}\n")

    best_config = None
    best_score = float("-inf")
    combo_id = 0

    for combo in product(*(param_grid[p] for p in all_param_names)):
        combo_id += 1
        param_combo = dict(zip(all_param_names, combo))

        cfg = copy.copy(base_config)
        for k, v in param_combo.items():
            setattr(cfg, k, v)

        combo_dir = os.path.join(base_config.checkpoint_dir, f"combo_{combo_id}")
        os.makedirs(combo_dir, exist_ok=True)
        cfg.checkpoint_dir = combo_dir

        logging.info(f"\n[Combination #{combo_id}] {param_combo}")

        train_out = train_fn(cfg, train_loader, dev_loader, test_loader)

        # If train() returns (dev, test), unpack; otherwise assume dev only
        if isinstance(train_out, tuple) and len(train_out) == 2:
            dev_met, test_met = train_out
        else:
            dev_met, test_met = train_out, {}
        dev_score = _pick_score(dev_met, selection_metric)
        is_better = dev_score > best_score

        box = format_result_box_dual(
            combo_id, " + ".join(all_param_names), str(combo),
            {}, dev_met, test_met,
            is_best=is_better, selection_metric=selection_metric,
            early_stop_on=base_config.early_stop_on
        )
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box + "\n")

        _log_dataset_metrics(dev_met, overrides_file, "dev")
        _log_dataset_metrics(test_met, overrides_file, "test")

        if is_better:
            best_score = dev_score
            best_config = param_combo

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Best combination (Dev-based) ===\n")
        for k, v in best_config.items():
            f.write(f"{k} = {v}\n")
    logging.info("Exhaustive search completed. Best parameters selected.")
    return best_score, best_config


# ────────────────────────── Dataset logging ──────────────────────────
def _log_dataset_metrics(metrics: dict, file_path: str, label: str = "dev") -> None:
    if "by_dataset" not in metrics:
        return
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Detailed metrics by dataset ({label})\n")
        for ds in metrics["by_dataset"]:
            name = ds.get("name", "unknown")
            f.write(f"  - {name}:\n")
            ordered = METRIC_ORDER + sorted(set(ds) - set(METRIC_ORDER))
            for k in ordered:
                if k in ds:
                    f.write(f"      {k.upper():8} = {ds[k]:.4f}\n")
        f.write(f"<<< End of detailed metrics ({label})\n")
