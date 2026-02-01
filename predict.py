import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter

from models.lstm_phy import LSTMPhyModel
from models.mlp_phy import MLPPhyModel
from utils.registry import get_registry_entry

# Matplotlib font settings (keep SimHei for Chinese labels if available)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_COLUMNS = ['Depth', 'AC', 'DEN', 'Ip']
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_SCALER_PATH = "models/scaler.pkl"
DEFAULT_REGISTRY_PATH = "models/registry.json"


def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="LSTM-Phy pore pressure inference and plotting")
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='Path to config file (YAML)')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint path; defaults to best_<type> under checkpoints/')
    parser.add_argument('--alias', default="N873_fortrain_Ip_mlp", help='Alias to lookup in models/registry.json')
    parser.add_argument('--registry', default=DEFAULT_REGISTRY_PATH, help='Path to model registry JSON')
    parser.add_argument('--input_csv', default=r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\5-PINN\data\raw\W585_filtered_ip.csv", help='Input CSV for prediction')
    parser.add_argument('--output_csv', default='results/W585_filtered_ip_predictions_mlp2201.csv', help='Where to save prediction CSV')
    parser.add_argument('--plot_path', default='results/W585_filtered_ip_predictions_mlp2201.png', help='Where to save pressure-depth plot')
    parser.add_argument('--sequence_length', type=int, help='Sequence length (fallback to config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (fallback to config)')
    parser.add_argument('--scaler_path', default=DEFAULT_SCALER_PATH, help='Path to StandardScaler used in training')
    return parser.parse_args()


def load_config(path: str):
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def restore_scaler(scaler_path: Path, checkpoint: dict) -> StandardScaler:
    """Load scaler from file; fall back to stats stored in checkpoint."""
    if scaler_path.exists():
        return joblib.load(scaler_path)
    if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        return scaler
    raise FileNotFoundError('StandardScaler not found. Provide models/scaler.pkl or checkpoint stats.')


def resolve_model_sources(args):
    """Resolve config/checkpoint/scaler paths from alias or explicit args."""
    registry_entry = None
    if args.alias:
        registry_path = Path(args.registry)
        if not registry_path.exists():
            raise FileNotFoundError(f'Registry file not found: {registry_path}')
        try:
            registry_entry = get_registry_entry(args.alias, registry_path)
        except KeyError as exc:
            raise SystemExit(f"Alias '{args.alias}' not found in registry: {registry_path}") from exc

    use_config_from_registry = registry_entry is not None and args.config == DEFAULT_CONFIG_PATH
    config_path = Path(registry_entry['config_path']) if use_config_from_registry else Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    config = load_config(config_path)
    model_type = config['model'].get('type', 'lstm').lower()
    default_checkpoint = Path('checkpoints') / f'best_{model_type}_model.pth'

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif registry_entry:
        checkpoint_path = Path(registry_entry.get('checkpoint_path', default_checkpoint))
    else:
        checkpoint_path = default_checkpoint

    scaler_path = Path(args.scaler_path)
    if registry_entry and args.scaler_path == DEFAULT_SCALER_PATH:
        scaler_path = Path(registry_entry.get('scaler_path', scaler_path))

    return config, config_path, checkpoint_path, scaler_path, registry_entry


def build_sequences(df: pd.DataFrame, seq_len: int, scaler: StandardScaler):
    """Standardize input and build sliding windows ending at each row."""
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if len(df) < seq_len:
        raise ValueError(f'Input rows {len(df)} < sequence_length {seq_len}, cannot build sequences')

    features = df[FEATURE_COLUMNS].values
    scaled = scaler.transform(features)
    sequences, idx = [], []
    for end_idx in range(seq_len, len(df) + 1):
        sequences.append(scaled[end_idx - seq_len:end_idx])
        idx.append(end_idx - 1)

    tensor = torch.from_numpy(np.stack(sequences)).float()
    metadata = df.iloc[idx].reset_index(drop=True)
    return tensor, metadata


def load_model(config, checkpoint_path, device):
    """Instantiate model and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = config['model']
    model_type = model_cfg.get('type', 'lstm').lower()
    if model_type == 'mlp':
        model = MLPPhyModel(
            input_size=model_cfg['input_size'],
            hidden_sizes=tuple(model_cfg.get('mlp_hidden_sizes', [128, 64])),
            dropout=model_cfg['dropout'],
        ).to(device)
    elif model_type == 'lstm':
        model = LSTMPhyModel(
            input_size=model_cfg['input_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def run_prediction(model, sequences, batch_size, device):
    """Run batched forward pass and return numpy predictions."""
    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            outputs = model(batch.to(device))
            preds.append(outputs.cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze(-1)


def compute_pw(depths: np.ndarray, physics_cfg: dict) -> np.ndarray:
    """Compute hydrostatic pressure Pw in MPa."""
    rho = physics_cfg['fluid_density']
    g = physics_cfg['gravity']
    return rho * g * depths / 1000.0


def compute_pv(depths: np.ndarray, physics_cfg: dict) -> np.ndarray:
    """Compute overburden pressure Pv in MPa."""
    rho_rock = physics_cfg['rho_rock']
    g = physics_cfg['gravity']
    return rho_rock * g * depths / 1000.0


def compute_nac(depths: np.ndarray, physics_cfg: dict) -> np.ndarray:
    """Compute normalized AC curve with exponential model."""
    a = physics_cfg['nac_a']
    b = physics_cfg['nac_b']
    return np.exp(a * depths + b)


def plot_profile(metadata: pd.DataFrame, physics_cfg: dict, plot_path: Path):
    """Plot predicted Pp, optional true Pp, hydrostatic Pw, and overburden Pv vs depth."""
    depths = metadata['Depth'].values
    preds = metadata['Pp_pred'].values
    pw_line = metadata['Pw'].values if 'Pw' in metadata.columns else compute_pw(depths, physics_cfg)


    # 2. 应用 Savitzky-Golay 滤波器平滑
    # ----------------------------
    # 参数说明：
    # - window_length: 必须是奇数，代表滑动窗口大小（点数）
    # - polyorder: 多项式阶数，通常 2 或 3
    # - mode: 边界处理方式（'interp', 'mirror', 'constant' 等）
    window_length = 201   # 窗口大小（根据数据密度调整，越大越平滑）
    polyorder = 3        # 多项式阶数（必须 < window_length）

    smoothed_pp = savgol_filter(preds, window_length=window_length, polyorder=polyorder)

    plt.figure(figsize=(6, 10))
    plt.plot(preds, depths, label='Pred Pp (raw)', color='tab:blue')
    plt.plot(smoothed_pp, depths, label='Pred Pp (smoothed)', color='tab:red')
    if 'Pp' in metadata.columns:
        plt.plot(metadata['Pp'].values, depths, linestyle='--', label='True Pp', color='tab:orange')
    plt.plot(pw_line, depths, linestyle='-.', label='Pw (hydrostatic)', color='tab:green')
    if 'Pv' in metadata.columns:
        plt.plot(metadata['Pv'].values, depths, linestyle=':', label='Pv (overburden)', color='tab:red')

    measured_pp = []
    measured_depth = []
    plt.plot(measured_pp, measured_depth, 'o', label='Measured Pp', color='tab:purple')

    plt.gca().invert_yaxis()
    plt.xlabel('Pressure (MPa)')
    plt.ylabel('Depth (m)')
    plt.title('Pore pressure vs depth')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    config, config_path, checkpoint_path, scaler_path, registry_entry = resolve_model_sources(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = args.sequence_length or config['data']['sequence_length']
    batch_size = args.batch_size or config['training']['batch_size']

    model, checkpoint = load_model(config, checkpoint_path, device)
    scaler = restore_scaler(scaler_path, checkpoint)
    expected_features = len(FEATURE_COLUMNS)
    if getattr(scaler, "n_features_in_", expected_features) != expected_features:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but current FEATURE_COLUMNS has {expected_features}. "
            "Please retrain to generate a matching scaler."
        )

    df = pd.read_csv(args.input_csv)
    sequences, metadata = build_sequences(df, seq_len, scaler)

    predictions = run_prediction(model, sequences, batch_size, device)
    metadata = metadata.copy()
    metadata['nAC'] = compute_nac(metadata['Depth'].values, config['physics'])
    metadata['Pp_pred'] = predictions
    metadata['Pv'] = compute_pv(metadata['Depth'].values, config['physics'])
    metadata['Pw'] = compute_pw(metadata['Depth'].values, config['physics'])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_csv, index=False)

    plot_path = Path(args.plot_path)
    plot_profile(metadata, config['physics'], plot_path)

    print(f'Predicted {len(predictions)} rows')
    print(f'Output CSV: {output_csv}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Plot: {plot_path}')
    print(f'Config used: {config_path}')
    print(f'Scaler: {scaler_path}')
    if args.alias:
        print(f'Using registry alias: {args.alias} ({args.registry})')


if __name__ == '__main__':
    main()
