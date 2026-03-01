"""
Interactive Visualization Script for AnyQuantileForecasterExogWithSeries Model
FIXED VERSION - handles YAML tuple syntax issue
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import sys
from datetime import datetime
import argparse
import yaml

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from model.models import AnyQuantileForecasterExogWithSeries
from dataset.datasets import EMHIRESUnivariateDataModule

# Configuration
OUTPUT_DIR = "results/exog_series_visualizations"
NUM_SAMPLES = 20

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f5f5f5'
plt.rcParams['font.size'] = 10

def find_exog_series_checkpoints():
    """Find checkpoints for exog-series experiments"""
    print("🔍 Searching for Exog+Series checkpoints...")
    
    lightning_logs = Path("lightning_logs")
    if not lightning_logs.exists():
        raise FileNotFoundError(f"Directory not found: {lightning_logs}")
    
    exog_series_checkpoints = []
    
    for log_dir in lightning_logs.glob("nbeatsaq-exog-series*"):
        if not log_dir.is_dir():
            continue
        ckpt_dir = log_dir / "checkpoints"
        if ckpt_dir.exists():
            for ckpt in ckpt_dir.glob("*.ckpt"):
                exog_series_checkpoints.append({
                    'path': ckpt,
                    'mtime': ckpt.stat().st_mtime,
                    'size_mb': ckpt.stat().st_size / (1024 * 1024),
                    'experiment': log_dir.name,
                    'name': ckpt.name
                })
    
    if not exog_series_checkpoints:
        print("⚠️  No exog-series checkpoints found. Searching all checkpoints...")
        for log_dir in lightning_logs.glob("*"):
            if not log_dir.is_dir():
                continue
            ckpt_dir = log_dir / "checkpoints"
            if ckpt_dir.exists():
                for ckpt in ckpt_dir.glob("*.ckpt"):
                    exog_series_checkpoints.append({
                        'path': ckpt,
                        'mtime': ckpt.stat().st_mtime,
                        'size_mb': ckpt.stat().st_size / (1024 * 1024),
                        'experiment': log_dir.name,
                        'name': ckpt.name
                    })
    
    if not exog_series_checkpoints:
        raise FileNotFoundError("No checkpoint files found")
    
    exog_series_checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    return exog_series_checkpoints

def list_checkpoints(checkpoints):
    """Display available checkpoints"""
    print("\n" + "="*100)
    print("📋 AVAILABLE CHECKPOINTS")
    print("="*100)
    
    for i, ckpt in enumerate(checkpoints, 1):
        mtime = datetime.fromtimestamp(ckpt['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{i:2d}] {ckpt['experiment']}")
        print(f"     File: {ckpt['name']}")
        print(f"     Modified: {mtime}")
        print(f"     Size: {ckpt['size_mb']:.1f} MB")
    
    print("\n" + "="*100)

def select_checkpoint(checkpoints):
    """Let user select a checkpoint"""
    list_checkpoints(checkpoints)
    
    while True:
        try:
            choice = input(f"\nSelect checkpoint [1-{len(checkpoints)}] or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                sys.exit(0)
            
            idx = int(choice) - 1
            
            if 0 <= idx < len(checkpoints):
                selected = checkpoints[idx]
                print(f"\n✅ Selected: {selected['experiment']}/{selected['name']}")
                return str(selected['path']), selected['experiment']
            else:
                print(f"❌ Invalid choice. Enter 1-{len(checkpoints)}")
        except ValueError:
            print("❌ Invalid input. Enter a number or 'q'")
        except KeyboardInterrupt:
            sys.exit(0)

def load_model_and_config(checkpoint_path, config_path=None):
    """Load model - FIXED to handle YAML tuple syntax"""
    
    if config_path and config_path.exists():
        print(f"📄 Loading config from: {config_path}")
        # Read and fix YAML
        with open(config_path, 'r') as f:
            yaml_content = f.read()
        
        # Remove problematic python/tuple syntax
        yaml_content = yaml_content.replace('!!python/tuple', '')
        
        # Load with safe_load then convert to OmegaConf
        try:
            cfg_dict = yaml.safe_load(yaml_content)
            cfg = OmegaConf.create(cfg_dict)
            print("✅ Config loaded successfully")
        except Exception as e:
            print(f"⚠️  Config load failed: {e}")
            print("Using minimal default config...")
            cfg = None
    else:
        cfg = None
    
    if cfg is None:
        # Create minimal default
        cfg = OmegaConf.create({
            'model': {
                'input_horizon_len': 168,
                'max_norm': True,
                'num_series': 35,
                'series_embed_dim': 32,
                'series_embed_scale': 0.08,
            },
            'dataset': {
                'name': 'MHLV',
                'train_batch_size': 512,
                'eval_batch_size': 512,
                'num_workers': 0,
                'persistent_workers': False,
                'horizon_length': 24,
                'history_length': 168,
                'split_boundaries': ['2006-01-01', '2017-12-30', '2018-01-01', '2019-01-01'],
                'fillna': 'ffill',
                'train_step': 1,
                'eval_step': 24,
            }
        })
    
    print(f"\n📦 Loading model from: {checkpoint_path}")
    
    try:
        model = AnyQuantileForecasterExogWithSeries.load_from_checkpoint(
            checkpoint_path, cfg=cfg, strict=False, map_location='cpu'
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Model loaded on {device}\n")
        return model, cfg, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def load_test_data(cfg):
    """Load test dataset"""
    print("📊 Loading test dataset...")
    
    dm = EMHIRESUnivariateDataModule(
        name=cfg.dataset.name,
        train_batch_size=cfg.dataset.train_batch_size,
        eval_batch_size=cfg.dataset.eval_batch_size,
        num_workers=0,  # Avoid multiprocessing for viz
        persistent_workers=False,
        horizon_length=cfg.dataset.horizon_length,
        history_length=cfg.dataset.history_length,
        split_boundaries=cfg.dataset.split_boundaries,
        fillna=cfg.dataset.fillna,
        train_step=cfg.dataset.train_step,
        eval_step=cfg.dataset.eval_step,
    )
    
    dm.setup('test')
    test_dataloader = dm.test_dataloader()
    
    print(f"✅ Test dataset loaded: {len(dm.test_dataset)} samples\n")
    return test_dataloader

@torch.no_grad()
def generate_predictions(model, dataloader, device, num_samples=100):
    """Generate predictions"""
    print(f"🔮 Generating predictions for {num_samples} samples...")
    
    model.eval()
    quantiles = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], 
                            dtype=torch.float32, device=device)
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    for batch in dataloader:
        if sample_count >= num_samples:
            break
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        batch_size = batch['history'].shape[0]
        batch_predictions = []
        
        for q in quantiles:
            q_batch = q.unsqueeze(0).expand(batch_size, 1).to(device)
            batch['quantiles'] = q_batch
            pred = model(batch)
            batch_predictions.append(pred.squeeze(-1))
        
        batch_predictions = torch.stack(batch_predictions, dim=-1)
        
        samples_to_take = min(batch_size, num_samples - sample_count)
        all_predictions.append(batch_predictions[:samples_to_take].cpu().numpy())
        all_targets.append(batch['target'][:samples_to_take].cpu().numpy())
        sample_count += samples_to_take
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    quantiles_array = quantiles.cpu().numpy()
    
    print(f"✅ Generated: {predictions.shape} (samples, horizon, quantiles)\n")
    return predictions, targets, [quantiles_array]

def plot_quantile_fan_charts(predictions, targets, quantiles, num_plots=5):
    """Plot quantile fan charts"""
    print("📈 Creating quantile fan charts...")
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    q = quantiles[0]
    
    for idx in range(min(num_plots, len(predictions))):
        ax = axes[idx]
        pred = predictions[idx]
        target = targets[idx]
        h = np.arange(len(target))
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(q)//2))
        
        for i in range(len(q)//2):
            lower_q = q[i]
            upper_q = q[-(i+1)]
            ax.fill_between(h, pred[:, i], pred[:, -(i+1)],
                          alpha=0.3, color=colors[i], 
                          label=f'{int((upper_q-lower_q)*100)}% PI')
        
        median_idx = len(q)//2
        ax.plot(h, pred[:, median_idx], 'b-', linewidth=2.5, label='Median', zorder=3)
        ax.plot(h, target, 'ro-', linewidth=2, markersize=5, label='Actual', alpha=0.8, zorder=4)
        
        ax.set_xlabel('Hour', fontsize=11)
        ax.set_ylabel('Load (MW)', fontsize=11)
        ax.set_title(f'Sample {idx+1} - Quantile Fan Chart', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/01_quantile_fan_charts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_point_forecasts(predictions, targets, quantiles, num_plots=5):
    """Plot point forecasts"""
    print("📈 Creating point forecast plots...")
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    q = quantiles[0]
    median_idx = len(q)//2
    
    for idx in range(min(num_plots, len(predictions))):
        ax = axes[idx]
        pred = predictions[idx][:, median_idx]
        target = targets[idx]
        h = np.arange(len(target))
        
        ax.plot(h, target, 'ro-', linewidth=2, markersize=5, label='Actual', alpha=0.8)
        ax.plot(h, pred, 'b-', linewidth=2, label='Predicted', alpha=0.8)
        
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target)**2))
        
        ax.set_xlabel('Hour', fontsize=11)
        ax.set_ylabel('Load (MW)', fontsize=11)
        ax.set_title(f'Sample {idx+1} - Point Forecast (MAE: {mae:.2f}, RMSE: {rmse:.2f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/02_point_forecasts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_quantile_analysis(predictions, targets, quantiles):
    """Plot quantile calibration"""
    print("📈 Creating quantile analysis...")
    
    q = quantiles[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calibration
    ax = axes[0]
    empirical_coverage = []
    for i, q_val in enumerate(q):
        pred_q = predictions[:, :, i]
        coverage = np.mean(targets <= pred_q)
        empirical_coverage.append(coverage)
    
    ax.plot(q, empirical_coverage, 'o-', linewidth=2, markersize=8, label='Empirical')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Nominal Coverage', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Quantile Calibration', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Coverage error
    ax = axes[1]
    coverage_errors = np.array(empirical_coverage) - q
    ax.bar(range(len(q)), coverage_errors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Quantile', fontsize=11)
    ax.set_ylabel('Coverage Error', fontsize=11)
    ax.set_title('Coverage Error', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(q)))
    ax.set_xticklabels([f'{q_val:.2f}' for q_val in q], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/03_quantile_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def save_summary_stats(predictions, targets, quantiles, checkpoint_info):
    """Save summary statistics"""
    print("📝 Saving summary statistics...")
    
    q = quantiles[0]
    median_idx = len(q)//2
    pred_median = predictions[:, :, median_idx]
    errors = pred_median - targets
    
    summary = []
    summary.append("=" * 80)
    summary.append("EXOG+SERIES MODEL - PERFORMANCE SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Checkpoint: {checkpoint_info['experiment']}/{checkpoint_info['name']}")
    summary.append(f"\nDataset: {len(predictions)} samples, {predictions.shape[1]}h horizon")
    summary.append(f"Quantiles: {q}")
    
    summary.append(f"\n\nPoint Metrics (Median):")
    summary.append(f"  MAE:  {np.mean(np.abs(errors)):.3f} MW")
    summary.append(f"  RMSE: {np.sqrt(np.mean(errors**2)):.3f} MW")
    summary.append(f"  MAPE: {np.mean(np.abs(errors / (targets + 1e-8))) * 100:.2f}%")
    
    summary.append(f"\n\nQuantile Coverage:")
    for i, q_val in enumerate(q):
        pred_q = predictions[:, :, i]
        empirical_cov = np.mean(targets <= pred_q)
        summary.append(f"  Q{q_val:.2f}: {empirical_cov:.3f} (error: {empirical_cov-q_val:+.3f})")
    
    summary.append("\n" + "=" * 80)
    
    output_path = f'{OUTPUT_DIR}/00_summary_statistics.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"   ✅ Saved: {output_path}")
    print("\n" + '\n'.join(summary))

def main():
    """Main visualization pipeline"""
    print("\n" + "="*100)
    print("🎨 EXOG+SERIES VISUALIZATION SUITE")
    print("="*100 + "\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint path')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES)
    args = parser.parse_args()
    
    try:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            experiment_name = Path(checkpoint_path).parent.parent.name
        else:
            checkpoints = find_exog_series_checkpoints()
            checkpoint_path, experiment_name = select_checkpoint(checkpoints)
        
        checkpoint_info = {
            'path': checkpoint_path,
            'experiment': experiment_name,
            'name': Path(checkpoint_path).name
        }
        
        config_path = Path("config/nbeatsaq-exog-series.yaml")
        
        model, cfg, device = load_model_and_config(checkpoint_path, config_path)
        test_dataloader = load_test_data(cfg)
        predictions, targets, quantiles = generate_predictions(
            model, test_dataloader, device, args.num_samples
        )
        
        print("\n📊 Generating plots...")
        print("-" * 100)
        plot_quantile_fan_charts(predictions, targets, quantiles, num_plots=5)
        plot_point_forecasts(predictions, targets, quantiles, num_plots=5)
        plot_quantile_analysis(predictions, targets, quantiles)
        
        print("-" * 100)
        save_summary_stats(predictions, targets, quantiles, checkpoint_info)
        
        print("\n" + "="*100)
        print(f"✨ COMPLETE! Output: {OUTPUT_DIR}/")
        print("="*100 + "\n")
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())