import csv
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
import yaml

from models.lstm_phy import LSTMPhyModel
from models.mlp_phy import MLPPhyModel
from utils.data_loader import FEATURE_COLUMNS, get_data_loaders
from utils.loss import PhysicsInformedLoss
from utils.registry import upsert_registry


DEPTH_IDX = FEATURE_COLUMNS.index('Depth')
AC_IDX = FEATURE_COLUMNS.index('AC')


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config):
        self.config = config
        self.seed = self.config['training'].get('seed', 42)
        set_global_seed(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = self.config['training'].get('grad_clip')
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience')
        self.model_type = self.config['model'].get('type', 'lstm').lower()
        self.timestamp = datetime.now()
        self.run_name = self.build_run_name()
        self.run_dir = Path('runs') / self.run_name
        self.best_checkpoint_path = self.run_dir / 'best_model.pth'
        self.loss_plot_path = self.run_dir / 'training_loss.png'
        self.scatter_plot_path = self.run_dir / 'actual_vs_pred.png'
        self.depth_profile_plot_path = self.run_dir / 'pred_vs_target_depth.png'
        self.metrics_path = self.run_dir / 'training_metrics.csv'
        self.scaler_path = self.run_dir / 'scaler.pkl'
        self.config_snapshot_path = self.run_dir / 'config.yaml'
        self.legacy_checkpoint_path = Path('checkpoints') / f'best_{self.model_type}_model.pth'
        self.legacy_loss_plot_path = Path('results') / f'training_loss_{self.model_type}.png'
        self.legacy_scatter_plot_path = Path('results') / f'actual_vs_pred_{self.model_type}.png'
        self.legacy_depth_profile_plot_path = Path('results') / f'pred_vs_target_depth_{self.model_type}.png'
        self.legacy_metrics_path = Path('results') / 'training_metrics.csv'
        self.legacy_scaler_path = Path('models') / 'scaler.pkl'
        self.registry_alias = (
            self.config.get('experiment', {}).get('alias')
            or f"{Path(self.config['data']['csv_file']).stem}_{self.model_type}"
        )
        self.metric_history = []

        self.setup_directories()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer_loss()

    def build_run_name(self):
        dataset_tag = Path(self.config['data']['csv_file']).stem
        user_tag = self.config.get('experiment', {}).get('name')
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        parts = [ts, dataset_tag, self.model_type]
        if user_tag:
            parts.append(user_tag)
        return "_".join(parts)

    def setup_directories(self):
        """确保检查点与结果输出目录已创建。"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        Path('checkpoints').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        with self.config_snapshot_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)

    def setup_data(self):
        """构建数据加载器并缓存标准化参数。"""
        self.train_loader, self.val_loader, self.scaler = get_data_loaders(
            csv_file=self.config['data']['csv_file'],
            sequence_length=self.config['data']['sequence_length'],
            batch_size=self.config['training']['batch_size'],
            train_ratio=self.config['data']['train_ratio'],
        )
        self.scaler_mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
        self.scaler_scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)
        joblib.dump(self.scaler, self.scaler_path)
        self.legacy_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.legacy_scaler_path)

    def setup_model(self):
        """��������ѡ�� LSTM-Phy �� MLP-Phy ģ�͡�"""
        model_cfg = self.config['model']
        if self.model_type == 'mlp':
            hidden_sizes = tuple(model_cfg.get('mlp_hidden_sizes', [128, 64]))
            self.model = MLPPhyModel(
                input_size=model_cfg['input_size'],
                hidden_sizes=hidden_sizes,
                dropout=model_cfg['dropout'],
            ).to(self.device)
        else:
            self.model = LSTMPhyModel(
                input_size=model_cfg['input_size'],
                hidden_size=model_cfg['hidden_size'],
                num_layers=model_cfg['num_layers'],
                dropout=model_cfg['dropout'],
            ).to(self.device)

    def setup_optimizer_loss(self):
        """创建优化器、损失函数与调度器。"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
        )
        physics_cfg = self.config['physics']
        self.criterion = PhysicsInformedLoss(
            data_weight=self.config['training']['data_weight'],
            eaton_exponent=physics_cfg['eaton_exponent'],
            fluid_density=physics_cfg['fluid_density'],
            gravity=physics_cfg['gravity'],
            rho_rock=physics_cfg['rho_rock'],
            nac_a=physics_cfg['nac_a'],
            nac_b=physics_cfg['nac_b'],
        )
        scheduler_cfg = self.config['training'].get('scheduler', {})
        patience = scheduler_cfg.get('patience')
        if patience:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_cfg.get('factor', 0.5),
                patience=patience,
                min_lr=scheduler_cfg.get('min_lr', 1e-6),
            )
        else:
            self.scheduler = None

    def prepare_physical_params(self, batch_features):
        """从标准化后的输入恢复原始的深度、声波时差等物理参数。"""
        current_features = batch_features[:, -1, :]
        scaler_mean = self.scaler_mean.to(dtype=current_features.dtype)
        scaler_scale = self.scaler_scale.to(dtype=current_features.dtype)
        unscaled_features = current_features * scaler_scale + scaler_mean

        physical_params = {
            'Depth': unscaled_features[:, DEPTH_IDX : DEPTH_IDX + 1],
            'AC': unscaled_features[:, AC_IDX : AC_IDX + 1],
        }
        return physical_params

    def train_epoch(self):
        """执行一次完整的训练轮。"""
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features, targets = features.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            physical_params = self.prepare_physical_params(features)
            loss_dict = self.criterion(predictions, targets, physical_params)
            loss = loss_dict['total_loss']

            loss.backward()
            if self.grad_clip is not None:
                nn_utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_data_loss += loss_dict['data_loss'].item()
            total_physics_loss += loss_dict['physics_loss'].item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')

        num_batches = len(self.train_loader)
        return (
            total_loss / num_batches,
            total_data_loss / num_batches,
            total_physics_loss / num_batches,
        )

    def validate(self):
        """在验证集上评估模型表现。"""
        self.model.eval()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0

        with torch.no_grad():
            for features, targets in self.val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                predictions = self.model(features)
                physical_params = self.prepare_physical_params(features)
                loss_dict = self.criterion(predictions, targets, physical_params)

                total_loss += loss_dict['total_loss'].item()
                total_data_loss += loss_dict['data_loss'].item()
                total_physics_loss += loss_dict['physics_loss'].item()

        num_batches = len(self.val_loader)
        return (
            total_loss / num_batches,
            total_data_loss / num_batches,
            total_physics_loss / num_batches,
        )

    def train(self):
        """执行包含日志与检查点的完整训练流程。"""
        print("Starting LSTM-Phy training...")
        print(f'Run directory: {self.run_dir}')
        train_losses = []
        val_losses = []
        train_data_losses = []
        train_physics_losses = []
        val_data_losses = []
        val_physics_losses = []

        epochs = self.config['training']['epochs']
        best_val_loss = None
        epochs_without_improve = 0

        for epoch in range(epochs):
            train_loss, data_loss, physics_loss = self.train_epoch()
            val_loss, val_data_loss, val_physics_loss = self.validate()
            if self.scheduler:
                self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_data_losses.append(data_loss)
            train_physics_losses.append(physics_loss)
            val_data_losses.append(val_data_loss)
            val_physics_losses.append(val_physics_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (Data: {data_loss:.6f}, Physics: {physics_loss:.6f})')
            print(f'  Val Loss:   {val_loss:.6f} (Data: {val_data_loss:.6f}, Physics: {val_physics_loss:.6f})')
            print(f'  LR: {current_lr:.6e}')

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_total': train_loss,
                'train_data': data_loss,
                'train_physics': physics_loss,
                'val_total': val_loss,
                'val_data': val_data_loss,
                'val_physics': val_physics_loss,
                'lr': current_lr,
            }
            self.metric_history.append(epoch_metrics)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improve = 0
                checkpoint_payload = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'scaler_mean': self.scaler.mean_.tolist(),
                    'scaler_scale': self.scaler.scale_.tolist(),
                }
                torch.save(checkpoint_payload, self.best_checkpoint_path)
                shutil.copy2(self.best_checkpoint_path, self.legacy_checkpoint_path)
                print(f'  [*] Updated best model checkpoint: {self.best_checkpoint_path}')
            else:
                epochs_without_improve += 1
            if (
                self.early_stopping_patience
                and epochs_without_improve >= self.early_stopping_patience
            ):
                print(f'触发早停机制，在第 {epoch + 1} 轮提前结束训练。')
                break

        best_loss_text = f"{best_val_loss:.6f}" if best_val_loss is not None else 'N/A'
        self.best_val_loss = best_val_loss
        self.plot_losses(train_losses, val_losses)
        self.save_metrics()
        self.plot_actual_vs_pred_scatter()
        self.plot_depth_profile()
        self.update_registry(best_val_loss)
        print(f"Training complete! Best val loss: {best_loss_text}. Artifacts: {self.run_dir}")

    def plot_losses(self, train_losses, val_losses):
        """绘制训练与验证损失曲线。"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(self.loss_plot_path)
        shutil.copy2(self.loss_plot_path, self.legacy_loss_plot_path)
        plt.close()

    def plot_actual_vs_pred_scatter(self):
        """���� Train / Val / All ��ǩ�͹۲�ֵ�Եĵ���ͼ��"""
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from {self.best_checkpoint_path} for scatter plot.")
        self.model.eval()

        def collect_preds(loader):
            ys, ps = [], []
            with torch.no_grad():
                for features, targets in loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    preds = self.model(features)
                    ys.append(targets.detach().cpu().numpy().ravel())
                    ps.append(preds.detach().cpu().numpy().ravel())
            return np.concatenate(ys), np.concatenate(ps)

        def r2_score(y_true, y_pred):
            if y_true.size == 0:
                return float('nan')
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

        y_train, p_train = collect_preds(self.train_loader)
        y_val, p_val = collect_preds(self.val_loader)
        y_all = np.concatenate([y_train, y_val])
        p_all = np.concatenate([p_train, p_val])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plots = [
            ("Training", y_train, p_train, 'tab:blue'),
            ("Validation", y_val, p_val, 'tab:green'),
            ("All", y_all, p_all, 'tab:orange'),
        ]

        for ax, (name, y, p, color) in zip(axes, plots):
            ax.scatter(y, p, alpha=0.6, s=20, edgecolors='none', color=color)
            vmin, vmax = min(y.min(), p.min()), max(y.max(), p.max())
            ax.plot([vmin, vmax], [vmin, vmax], color='red', linewidth=2)
            if y.size > 1:
                m, b = np.polyfit(y, p, 1)
                ax.plot([vmin, vmax], [m * vmin + b, m * vmax + b], color='gray', linestyle='--')
            r2 = r2_score(y, p)
            ax.set_title(f'{name} (R²={r2:.4f})', fontsize=16)
            ax.set_xlabel('Target', fontsize=14)
            ax.set_ylabel('Prediction', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        self.scatter_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.scatter_plot_path, dpi=300, bbox_inches='tight')
        self.legacy_scatter_plot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.scatter_plot_path, self.legacy_scatter_plot_path)
        plt.close(fig)
        print(f"Saved scatter plot to {self.scatter_plot_path}")

    def plot_depth_profile(self):
        """Plot prediction and target versus depth (shallow at top, deep at bottom)."""
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from {self.best_checkpoint_path} for depth profile.")
        self.model.eval()

        def collect_depth_series(loader):
            depths, targets, preds = [], [], []
            with torch.no_grad():
                for features, targets_batch in loader:
                    features = features.to(self.device)
                    targets_batch = targets_batch.to(self.device)
                    preds_batch = self.model(features)
                    physical_params = self.prepare_physical_params(features)
                    depth_batch = physical_params['Depth'].detach().cpu().numpy().ravel()
                    depths.append(depth_batch)
                    targets.append(targets_batch.detach().cpu().numpy().ravel())
                    preds.append(preds_batch.detach().cpu().numpy().ravel())
            if not depths:
                return None, None, None
            return (
                np.concatenate(depths),
                np.concatenate(targets),
                np.concatenate(preds),
            )

        train_depths, train_targets, train_preds = collect_depth_series(self.train_loader)
        val_depths, val_targets, val_preds = collect_depth_series(self.val_loader)

        all_depths = []
        all_targets = []
        all_preds = []
        for d, t, p in [
            (train_depths, train_targets, train_preds),
            (val_depths, val_targets, val_preds),
        ]:
            if d is not None:
                all_depths.append(d)
                all_targets.append(t)
                all_preds.append(p)

        if not all_depths:
            print("No data available to plot depth profile.")
            return

        depths = np.concatenate(all_depths)
        targets = np.concatenate(all_targets)
        preds = np.concatenate(all_preds)

        sort_idx = np.argsort(depths)
        depths_sorted = depths[sort_idx]
        targets_sorted = targets[sort_idx]
        preds_sorted = preds[sort_idx]

        plt.figure(figsize=(6, 10))
        plt.plot(preds_sorted, depths_sorted, label='Prediction', color='tab:blue')
        plt.plot(targets_sorted, depths_sorted, label='Target', color='tab:orange', linestyle='--')
        plt.gca().invert_yaxis()
        plt.xlabel('Pressure')
        plt.ylabel('Depth')
        plt.title('Prediction vs Target by Depth')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()

        self.depth_profile_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.depth_profile_plot_path, dpi=300, bbox_inches='tight')
        self.legacy_depth_profile_plot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.depth_profile_plot_path, self.legacy_depth_profile_plot_path)
        plt.close()
        print(f"Saved depth profile plot to {self.depth_profile_plot_path}")

    def save_metrics(self):
        """将每轮指标写入 CSV 以便后续分析。"""
        if not self.metric_history:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.metric_history[0].keys())
        with self.metrics_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metric_history)
        self.legacy_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.metrics_path, self.legacy_metrics_path)

    def update_registry(self, best_val_loss):
        if not self.best_checkpoint_path.exists():
            return
        entry = {
            'alias': self.registry_alias,
            'run_dir': str(self.run_dir),
            'checkpoint_path': str(self.best_checkpoint_path),
            'config_path': str(self.config_snapshot_path),
            'scaler_path': str(self.scaler_path),
            'model_type': self.model_type,
            'data_csv': self.config['data']['csv_file'],
            'run_name': self.run_name,
            'created_at': self.timestamp.isoformat(timespec='seconds'),
            'best_val_loss': best_val_loss,
        }
        notes = self.config.get('experiment', {}).get('notes')
        if notes:
            entry['notes'] = notes
        upsert_registry(self.registry_alias, entry)
        print(f"[+] Updated registry alias '{self.registry_alias}' -> {self.best_checkpoint_path}")



if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()
