"""
Extract predictions from trained model and save for CQR

This script loads a trained model checkpoint and saves the predictions
for later CQR application.
"""

import torch
import sys
import pickle
import numpy as np

# Add project to path
sys.path.insert(0, '.')

from model.models import AnyQuantileForecaster
from dataset.datasets import ElectricityUnivariateDataModule


def main():
    print("=" * 80)
    print("Extracting Model Predictions for CQR")
    print("=" * 80)
    
    # Model checkpoint
    checkpoint_path = "lightning_logs/MHLV/model=NBEATSAQ-MultiHead-blocks30-history=168-lr=0.0005-seed=0-seed0/checkpoints/model-epoch=14.ckpt"
    
    print(f"\nLoading model from: {checkpoint_path}")
    
    try:
        model = AnyQuantileForecaster.load_from_checkpoint(checkpoint_path)
        model.eval()
        print("[OK] Model loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ElectricityUnivariateDataModule(
        history_length=168,
        horizon_length=48,
        train_batch_size=1024,
        eval_batch_size=1024,
        num_workers=0,  # Avoid multiprocessing issues
        split_boundaries=['2006-01-01', '2017-12-30', '2018-01-01', '2019-01-01']
    )
    dataset.setup('fit')
    dataset.setup('test')
    print("[OK] Dataset loaded")
    
    # Get dataloaders
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    
    #quantiles
    quantile_levels = torch.tensor([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Collect validation predictions (for calibration)
    print("\nCollecting validation predictions...")
    cal_preds = []
    cal_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(val_loader)}")
            
            x = batch['history'].to(device)
            y = batch['target']
            
            # Use fixed quantiles
            quantiles = quantile_levels.unsqueeze(0).expand(x.size(0), -1).to(device)
            
            # Forward pass
            preds = model(x, quantiles)
            
            cal_preds.append(preds.cpu().numpy())
            cal_targets.append(y.numpy())
    
    cal_preds = np.concatenate(cal_preds, axis=0)
    cal_targets = np.concatenate(cal_targets, axis=0)
    print(f"[OK] Validation predictions: {cal_preds.shape}")
    
    # Collect test predictions
    print("\nCollecting test predictions...")
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(test_loader)}")
            
            x = batch['history'].to(device)
            y = batch['target']
            
            quantiles = quantile_levels.unsqueeze(0).expand(x.size(0), -1).to(device)
            
            preds = model(x, quantiles)
            
            test_preds.append(preds.cpu().numpy())
            test_targets.append(y.numpy())
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    print(f"[OK] Test predictions: {test_preds.shape}")
    
    # Save to file
    output_file = 'predictions_multihead.pkl'
    print(f"\nSaving predictions to {output_file}...")
    
    data = {
        'cal_preds': cal_preds,
        'cal_targets': cal_targets,
        'test_preds': test_preds,
        'test_targets': test_targets,
        'quantiles': quantile_levels.numpy()
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("[OK] Predictions saved")
    
    print("\n" + "=" * 80)
    print("Done! Now run: python apply_cqr_simple.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
