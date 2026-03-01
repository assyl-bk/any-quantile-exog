#!/usr/bin/env python3
"""
run-stage3.py
=============
Manually loads Stage 2 checkpoint with strict=False to handle
model architecture mismatch (AnyQuantileForecasterExogWithSeries ->
AnyQuantileForecasterExogSeriesAdaptive).

Uses the same instantiate() utility as run.py so that the DataModule
is constructed exactly as the config describes — avoiding the
'unexpected keyword argument' errors that arise from manually
unpacking cfg.dataset fields.

Usage:
    python run-stage3.py
    python run-stage3.py --config config/stage3.yaml
"""

import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Reuse the same factory used by run.py — handles _target_ resolution
from utils.model_factory import instantiate
from model.models import AnyQuantileForecasterExogSeriesAdaptive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Stage 3 training with strict=False checkpoint loading")
parser.add_argument("--config", type=str, default="config/stage3.yaml",
                    help="Path to stage3 config file")
parser.add_argument("overrides", nargs="*",
                    help="Optional key=value overrides (accepted but ignored — edit the yaml instead)")
args = parser.parse_args()

if args.overrides:
    print(f"\n⚠️  Note: positional overrides are ignored in this script: {args.overrides}")
    print("   Edit the yaml directly or use run.py for override support.\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
print("=" * 80)
print("STAGE 3: Manual Training with Stage 2 Checkpoint Loading")
print("=" * 80)

cfg = OmegaConf.load(args.config)
print(f"\n✅ Config loaded from {args.config}")

seed = int(cfg.random.seed)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print(f"\n🤖 Creating model: AnyQuantileForecasterExogSeriesAdaptive")
model = AnyQuantileForecasterExogSeriesAdaptive(cfg)
print(f"   - num_adaptive_quantiles: {model.num_adaptive_quantiles}")
print(f"   - temperature:            {model.temperature}")
print(f"   - momentum:               {model.momentum}")

# ---------------------------------------------------------------------------
# Load Stage 2 checkpoint (strict=False — adaptive params are new)
# ---------------------------------------------------------------------------
stage2_ckpt = OmegaConf.select(cfg, "checkpoint.resume_ckpt")
if not stage2_ckpt:
    raise ValueError(
        "cfg.checkpoint.resume_ckpt is not set. "
        "Please set it to the Stage 2 checkpoint path in stage3.yaml."
    )

print(f"\n📦 Loading Stage 2 weights from:\n   {stage2_ckpt}")
try:
    ckpt = torch.load(stage2_ckpt, map_location="cpu")
    result = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("✅ Stage 2 weights loaded successfully!")
    if result.missing_keys:
        print(f"   Missing keys (initialized fresh): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"   Unexpected keys (ignored):        {result.unexpected_keys}")
except Exception as exc:
    print(f"❌ Error loading checkpoint: {exc}")
    print("   Continuing with fresh initialization...")

# ---------------------------------------------------------------------------
# DataModule  — use instantiate() so _target_ + all kwargs are handled
# correctly, exactly as run.py does.  This avoids manually unpacking
# cfg.dataset fields and hitting 'unexpected keyword argument' errors.
# ---------------------------------------------------------------------------
print(f"\n📊 Creating DataModule via instantiate()...")
dm = instantiate(cfg.dataset)
dm.prepare_data()
dm.setup(stage="fit")
print("✅ DataModule ready")

# ---------------------------------------------------------------------------
# Callbacks & Logger
# ---------------------------------------------------------------------------
log_name = f"{cfg.logging.name}-seed{seed}"

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

print(f"\n📝 Creating TensorBoard logger: {log_name}")
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

print(f"\n🚀 Starting Stage 3 training...")
print("=" * 80)
trainer.fit(model, datamodule=dm)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
# Use the best checkpoint saved during stage 3 (not the stage 2 one)
print(f"\n🧪 Running test evaluation (best Stage 3 checkpoint)...")
print("=" * 80)
test_results = trainer.test(datamodule=dm, ckpt_path="best")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STAGE 3 COMPLETE!")
print("=" * 80)
print("\nTest Results:")
for result_dict in test_results:
    for metric, value in result_dict.items():
        print(f"  {metric}: {value:.6f}")
print("\n" + "=" * 80)