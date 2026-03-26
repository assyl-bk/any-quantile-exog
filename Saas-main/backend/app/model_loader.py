import os
import sys
import pickle
import logging
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from omegaconf import OmegaConf
from functools import lru_cache
from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_model_root(base_path: Path) -> Path:

    candidate = base_path.resolve()

    if (candidate / "model").is_dir() or (candidate / "model.py").exists():
        return candidate

    for _ in range(5):
        candidate = candidate.parent
        if (candidate / "model").is_dir() or (candidate / "model.py").exists():
            return candidate

    this_file = Path(__file__).resolve()
    for level in range(2, 6):
        candidate = this_file.parents[level] if level < len(this_file.parents) else this_file.parent
        if (candidate / "model").is_dir() or (candidate / "model.py").exists():
            return candidate

    logger.error(
        "Could not locate 'model/' package. "
        f"Searched from {base_path} and relative to {this_file}. "
        "Please set MODEL_BASE_PATH to the directory that contains model/"
    )
    return base_path.resolve()


class ModelLoader:
    
    _instance = None
    _model = None
    _calibrator = None
    _cfg = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    def load_config(self, config_path: str) -> OmegaConf:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            raw = f.read().replace("!!python/tuple", "")
        
        cfg = OmegaConf.create(yaml.safe_load(raw))
        logger.info(f"Config loaded: {config_path}")
        return cfg
    
    def load_model(self, checkpoint_path: str, config_path: str):
        """
        Load AQ_NBEATS++_S2 model from checkpoint.
        
        Args:
            checkpoint_path: Path to .ckpt file
            config_path: Path to .yaml configuration file
        """
        # Avoid reloading if already loaded
        if self._model is not None and self._cfg is not None:
            logger.info("Model already loaded")
            return self._model
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            self._cfg = self.load_config(config_path)
            
            # ── PATH FIX ──────────────────────────────────────────────────────
            # settings.MODEL_BASE_PATH may point to Saas-main/ but the model/
            # package lives in the parent repo root (any-quantile-exog/).
            # _resolve_model_root() walks upward until it finds the model/ dir.
            configured_base = Path(settings.MODEL_BASE_PATH)
            model_root = _resolve_model_root(configured_base)

            if str(model_root) not in sys.path:
                sys.path.insert(0, str(model_root))
                logger.info(f"Added {model_root} to sys.path")
            else:
                logger.info(f"sys.path already contains {model_root}")
            # ─────────────────────────────────────────────────────────────────

            # Import here to avoid dependency issues if model not used
            from model.models import AQNBEATSPlusPlus
            
            self._model = AQNBEATSPlusPlus.load_from_checkpoint(
                str(checkpoint_path),
                cfg=self._cfg,
                strict=False,
                map_location=self.device,
                weights_only=False
            )
            self._model.eval()
            self._model.to(self.device)
            
            logger.info(f"Model loaded from: {checkpoint_path}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return self._model
    
    def load_calibrator(self, calibrator_path: str):
        """
        Load CQR calibrator (offsets per quantile).
        
        Args:
            calibrator_path: Path to .pkl file containing calibrator dict
        """
        if self._calibrator is not None:
            logger.info("Calibrator already loaded")
            return self._calibrator
        
        calibrator_path = Path(calibrator_path)
        if not calibrator_path.exists():
            logger.warning(f"Calibrator not found: {calibrator_path}. Using raw predictions.")
            return None
        
        try:
            # Must import ConformalCalibrator BEFORE pickle.load so it can
            # resolve the class that was stored under __main__ when saved.
            model_root = _resolve_model_root(Path(settings.MODEL_BASE_PATH))
            if str(model_root) not in sys.path:
                sys.path.insert(0, str(model_root))
            from calibrator import ConformalCalibrator  # noqa: F401

            with open(calibrator_path, 'rb') as f:
                self._calibrator = pickle.load(f)
            logger.info(f"Calibrator loaded: {calibrator_path}")
            logger.info(f"Quantiles: {sorted(self._calibrator.q_per_level.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load calibrator ({e}). Using raw predictions without CQR calibration.")
            self._calibrator = None
        
        return self._calibrator
    
    def get_model(self):
        """Get loaded model instance."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    def _get_forecast_horizon(self) -> int:
        """Get forecast horizon from config."""
        if self._cfg is None:
            return 24  # Default fallback
        return int(self._cfg.get('forecast_horizon', self._cfg.get('history_length', 24)))
    
    def get_calibrator(self) -> Optional[Dict]:
        """Get loaded calibrator instance."""
        return self._calibrator
    
    def get_config(self) -> Optional[OmegaConf]:
        """Get loaded configuration."""
        return self._cfg
    
    @torch.no_grad()
    def predict(
        self,
        historical_data: np.ndarray,
        exogenous_data: Optional[np.ndarray] = None,
        quantiles: Optional[List[float]] = None,
        apply_cqr: bool = True,
        horizon: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate quantile forecasts.
        
        Args:
            historical_data: Historical univariate data (np.ndarray) shape (history_len,)
            exogenous_data: Exogenous features (optional) shape (history_len, num_features)
            quantiles: List of quantile levels (default: [0.1, 0.5, 0.9])
            apply_cqr: Apply CQR calibration if available (default: True)
            horizon: Optional forecast horizon override (default: use model config)
        
        Returns:
            Dict with keys:
                - 'forecasts': np.ndarray shape (num_quantiles, forecast_horizon)
                - 'quantiles': List of quantile values
                - 'cqr_applied': bool
        """
        model = self.get_model()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        forecast_horizon = horizon if horizon is not None else self._get_forecast_horizon()
        
        try:
            y_tensor = torch.from_numpy(historical_data).float().to(self.device)
            
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(0)
            
            x_tensor = None
            if exogenous_data is not None:
                x_tensor = torch.from_numpy(exogenous_data).float().to(self.device)
                if x_tensor.dim() == 2:
                    x_tensor = x_tensor.unsqueeze(0)
            
            batch_size = y_tensor.shape[0]
            
            per_q = []
            for q_val in quantiles:
                batch = {
                    "history": y_tensor,
                    "quantiles": torch.full(
                        (batch_size, 1), float(q_val),
                        dtype=torch.float32, device=self.device
                    ),
                    "target": torch.zeros(
                        (batch_size, forecast_horizon),
                        dtype=torch.float32, device=self.device
                    )
                }
                
                out = model.shared_forward(batch) if hasattr(model, 'shared_forward') else model(batch)
                
                if isinstance(out, dict):
                    forecast_vals = out["forecast"][..., 0].squeeze(0).cpu().numpy()
                else:
                    forecast_vals = out.squeeze(0).cpu().numpy()
                
                per_q.append(forecast_vals)
            
            forecasts_np = np.stack(per_q, axis=0)
            
            cqr_applied = False
            if apply_cqr and self._calibrator is not None:
                forecasts_np = self._apply_cqr(forecasts_np, quantiles)
                cqr_applied = True
            
            return {
                'forecasts': forecasts_np,
                'quantiles': quantiles,
                'cqr_applied': cqr_applied
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _apply_cqr(self, forecasts: np.ndarray, quantiles: List[float]) -> np.ndarray:
        """
        Apply CQR calibration by adding offsets to quantile predictions.
        
        Args:
            forecasts: Raw predictions shape (num_quantiles, forecast_horizon)
            quantiles: List of quantile levels
        
        Returns:
            Calibrated predictions
        """
        calibrated = forecasts.copy()
        
        for i, q in enumerate(quantiles):
            if q in self._calibrator.q_per_level:
                offset = self._calibrator.q_per_level[q]
                calibrated[i, :] += offset
                logger.debug(f"Q{q:.3f}: Applied offset {offset:.6f}")
            else:
                logger.warning(f"Quantile {q} not in calibrator, using raw prediction")
        
        return calibrated
    
    def reset(self):
        """Reset loader to unload model."""
        self._model = None
        self._calibrator = None
        self._cfg = None
        logger.info("Model loader reset")


# Global loader instance
_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create the model loader singleton."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def initialize_model(
    base_path: str,
    checkpoint: str = "lightning_logs/nbeatsaq-stage2-seed0/checkpoints/model-epoch=4.ckpt",
    config: str = "config/AQNBEATS++_S2.yaml",
    calibrator: str = "results/CQR/calibrator_stage2_v2.pkl"
):
    """
    Initialize model loader with paths.
    Call this once at application startup.
    
    Args:
        base_path: Base path to project directory (any-quantile-exog/)
        checkpoint: Relative path to checkpoint
        config: Relative path to config
        calibrator: Relative path to calibrator
    """
    loader = get_model_loader()
    
    base_path = Path(base_path)
    ckpt_path = base_path / checkpoint
    cfg_path  = base_path / config
    cal_path  = base_path / calibrator
    
    logger.info("Initializing model loader...")
    logger.info(f"Base path: {base_path}")
    
    loader.load_model(str(ckpt_path), str(cfg_path))
    
    if cal_path.exists():
        loader.load_calibrator(str(cal_path))
    else:
        logger.warning(f"Calibrator not found: {cal_path}. Using raw predictions.")
    
    logger.info("Model loader initialized successfully")