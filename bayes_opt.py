import argparse
import copy
import json
import tempfile
from pathlib import Path

import optuna
import yaml

from train import Trainer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def objective(trial, base_config: dict, args) -> float:
    # 深拷贝基础配置，防止污染原 config
    cfg = copy.deepcopy(base_config)

    model_type = args.model or cfg["model"].get("type", "lstm").lower()
    cfg["model"]["type"] = model_type

    # 训练超参
    cfg["training"]["epochs"] = args.tune_epochs
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    # 搜索空间：学习率、数据权重、dropout
    cfg["training"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    cfg["training"]["data_weight"] = trial.suggest_float("data_weight", 0.2, 0.9)
    cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

    if model_type == "mlp":
        h1 = trial.suggest_int("hidden1", 64, 512, step=64)
        h2 = trial.suggest_int("hidden2", 32, 256, step=32)
        cfg["model"]["mlp_hidden_sizes"] = [h1, h2]
    else:  # lstm
        cfg["model"]["hidden_size"] = trial.suggest_int("hidden_size", 64, 512, step=64)
        cfg["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 4)

    # 缩短 early stopping 容忍度，避免调参阶段过长
    base_patience = cfg["training"].get("early_stopping_patience")
    if base_patience:
        cfg["training"]["early_stopping_patience"] = max(3, min(base_patience, args.tune_epochs // 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        trainer = Trainer(cfg)
        # 将输出路径指向临时目录，避免大量 trial 文件覆盖
        trainer.best_checkpoint_path = tmpdir / f"best_{model_type}_trial.pth"
        trainer.loss_plot_path = tmpdir / f"loss_{model_type}_trial.png"
        trainer.scatter_plot_path = tmpdir / f"scatter_{model_type}_trial.png"
        trainer.metrics_path = tmpdir / "metrics_trial.csv"

        trainer.train()

        if not trainer.metric_history:
            raise RuntimeError("No metrics recorded; check training loop.")
        best_val = min(m["val_total"] for m in trainer.metric_history)
        return best_val


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Bayesian Optimization for PINN hyperparameters")
    parser.add_argument("--config", default="config.yaml", help="Path to base config.yaml")
    parser.add_argument("--model", choices=["lstm", "mlp"], help="Override model type (default: config value)")
    parser.add_argument("--trials", type=int, default=20, help="Number of optuna trials")
    parser.add_argument("--tune-epochs", type=int, default=20, help="Epochs per trial (shorter for tuning)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study-name", default="pinn_bayes_opt", help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--timeout", type=int, default=None, help="Time limit seconds for the whole optimization")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    base_config = load_config(config_path)

    study_kwargs = {
        "direction": "minimize",
        "study_name": args.study_name,
    }
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.optimize(lambda t: objective(t, base_config, args), n_trials=args.trials, timeout=args.timeout)

    print("\nBest trial:")
    print(f"  Value (val_loss): {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")

    # 将最佳参数保存到文件，便于更新 config
    output = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "model": args.model or base_config["model"].get("type", "lstm").lower(),
        "tune_epochs": args.tune_epochs,
    }
    out_path = Path("results") / "bayes_opt_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Best params saved to {out_path}")


if __name__ == "__main__":
    main()
