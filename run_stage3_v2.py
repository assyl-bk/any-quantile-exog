#!/usr/bin/env python3
"""
run-stage3.py
=============
Generic script for any stage that uses AnyQuantileForecasterExogSeriesAdaptive
and needs strict=False checkpoint loading (handles missing adaptive/series keys).

Works for:
  - Original Stage 3:  python run-stage3.py --config config/stage3.yaml
  - v2 Stage 2:        python run-stage3.py --config config/stage2-v2.yaml
  - v2 Stage 3:        python run-stage3.py --config config/stage3-v2.yaml

The key reason this script exists (vs run.py):
  PyTorch Lightning's ckpt_path=... requires an EXACT architecture match.
  When transitioning from AnyQuantileForecasterExogWithSeries to
  AnyQuantileForecasterExogSeriesAdaptive, new parameters (bin_probs etc.)
  are missing from the old checkpoint — strict=False handles this gracefully.
"""

import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.model_factory import instantiate
from model.models import AnyQuantileForecasterExogSeriesAdaptive


# ---------------------------------------------------------------------------
# CLI — accepts positional overrides for compatibility but ignores them
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Stage training with strict=False checkpoint loading"
)
parser.add_argument("--config", type=str, default="config/stage3.yaml",
                    help="Path to stage yaml config")
parser.add_argument("overrides", nargs="*",
                    help="key=value overrides (accepted for CLI compat, ignored — edit yaml instead)")
args = parser.parse_args()

if args.overrides:
    print(f"\n⚠️  Positional overrides ignored: {args.overrides}")
    print("   Edit the yaml directly for config changes.\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"ADAPTIVE STAGE TRAINING: {args.config}")
print("=" * 80)

cfg = OmegaConf.load(args.config)
print(f"\n✅ Config loaded from {args.config}")

seed = int(cfg.random.seed)
log_name = f"{cfg.logging.name}-seed{seed}"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print(f"\n🤖 Creating model: AnyQuantileForecasterExogSeriesAdaptive")
model = AnyQuantileForecasterExogSeriesAdaptive(cfg)
print(f"   - num_adaptive_quantiles: {model.num_adaptive_quantiles}")
print(f"   - temperature:            {model.temperature}")
print(f"   - momentum:               {model.momentum}")
print(f"   - series_embed_scale:     {cfg.model.series_embed_scale}")

# ---------------------------------------------------------------------------
# Load previous stage checkpoint with strict=False
# New keys (bin_probs, series weights) are initialized fresh if missing.
# ---------------------------------------------------------------------------
resume_ckpt = OmegaConf.select(cfg, "checkpoint.resume_ckpt")

if resume_ckpt:
    print(f"\n📦 Loading previous stage weights from:\n   {resume_ckpt}")
    try:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        result = model.load_state_dict(ckpt["state_dict"], strict=False)
        print("✅ Weights loaded successfully (strict=False)")
        if result.missing_keys:
            print(f"   Keys initialized fresh:      {result.missing_keys}")
        if result.unexpected_keys:
            print(f"   Keys ignored (not in model): {result.unexpected_keys}")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {resume_ckpt}")
        print("   Make sure the previous stage completed successfully.")
        raise
    except Exception as exc:
        print(f"❌ Error loading checkpoint: {exc}")
        print("   Continuing with fresh initialization...")
else:
    print("\n⚠️  No resume_ckpt set — training from scratch.")

# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
print(f"\n📊 Creating DataModule via instantiate()...")
dm = instantiate(cfg.dataset)
dm.prepare_data()
dm.setup(stage="fit")
print("✅ DataModule ready")

# ---------------------------------------------------------------------------
# Callbacks & Logger
# ---------------------------------------------------------------------------
print(f"\n💾 Setting up callbacks...")
checkpoint_callback = ModelCheckpoint(
    dirpath=f"lightning_logs/{log_name}/checkpoints",
    filename="model-epoch={epoch:02d}",
    monitor="val/crps",
    mode="min",
    save_top_k=OmegaConf.select(cfg, "checkpoint.save_top_k", default=1),
    auto_insert_metric_name=False,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
print("✅ Callbacks configured")

print(f"\n📝 Creating logger: {log_name}")
logger = TensorBoardLogger(
    save_dir=cfg.logging.path,
    version=log_name,
    name="",
)
print("✅ Logger ready")

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
print(f"\n🏋️  Creating Trainer ({cfg.trainer.max_epochs} max epochs)...")
trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    devices=cfg.trainer.devices,
    accelerator=cfg.trainer.accelerator,
    precision=cfg.trainer.precision,
    gradient_clip_val=cfg.trainer.gradient_clip_val,
    check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    log_every_n_steps=cfg.trainer.log_every_n_steps,
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
)
print("✅ Trainer configured")

# ---------------------------------------------------------------------------
# Seed & Train
# ---------------------------------------------------------------------------
print(f"\n🎲 Setting random seed: {seed}")
pl.seed_everything(seed, workers=True)

print(f"\n🚀 Starting training...")
print("=" * 80)
trainer.fit(model, datamodule=dm)

# ---------------------------------------------------------------------------
# Test using the best checkpoint from THIS stage
# ---------------------------------------------------------------------------
print(f"\n🧪 Running test evaluation (best checkpoint from this stage)...")
print("=" * 80)
test_results = trainer.test(datamodule=dm, ckpt_path="best")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"STAGE COMPLETE: {args.config}")
print("=" * 80)
print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
print(f"Best val/crps:   {checkpoint_callback.best_model_score:.4f}")
print("\nTest Results:")
for result_dict in test_results:
    for metric, value in result_dict.items():
        print(f"  {metric}: {value:.6f}")
print("\n" + "=" * 80)