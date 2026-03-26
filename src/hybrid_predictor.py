"""
Hybrid prediction utilities (ACAGN-Hybrid)
=========================================

ACAGN-Hybrid is a probability-level ensemble:

  p_hybrid = w * p_base + (1 - w) * p_gate

where:
  - p_base is the calibrated probability from the ACAGN base ensemble
  - p_gate is the calibrated probability from ACAGN-Gate

Important:
  The gate model bundle must include saved weights (e.g. `seed_state_dicts` or
  `best_seed_state_dict`). Older bundles that only contain test predictions
  cannot be used for new-patient inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch

try:
    from config import GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, GATE_HIDDEN_DIM, GATE_DROPOUT
    from gated_fusion_model import TextGuidedGate
except ImportError:
    from .config import GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, GATE_HIDDEN_DIM, GATE_DROPOUT
    from .gated_fusion_model import TextGuidedGate


def hybrid_combine(p_base: float, p_gate: float, w_base: float = 0.5) -> float:
    p_base = float(p_base)
    p_gate = float(p_gate)
    w_base = float(w_base)
    return float(w_base * p_base + (1.0 - w_base) * p_gate)


def _resolve_gate_bundle_path(path: str = GATE_MODEL_PKL) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(GATE_MODEL_PKL_LEGACY):
        return GATE_MODEL_PKL_LEGACY
    return path


def _state_dict_from_numpy(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        else:
            arr = np.asarray(v)
            out[k] = torch.from_numpy(arr)
    return out


@dataclass(frozen=True)
class GateBundleSpec:
    tab_cols: List[str]
    emb_cols: List[str]
    text_dim: int
    tabular_dim: int
    calibrator: object
    seed_state_dicts: Dict[int, dict]
    best_seed_state_dict: Optional[dict]
    gate_hidden_dim: int
    gate_dropout: float


def load_gate_bundle(bundle_path: Optional[str] = None) -> Tuple[GateBundleSpec, dict]:
    path = _resolve_gate_bundle_path(bundle_path or GATE_MODEL_PKL)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gate model bundle not found at {path}. "
            f"Train an inference-ready bundle with: python3 -m src.gated_fusion_model --model-out {GATE_MODEL_PKL}"
        )

    bundle = joblib.load(path)

    tab_cols = bundle.get("tab_cols")
    emb_cols = bundle.get("emb_cols")
    text_dim = bundle.get("text_dim")
    tabular_dim = bundle.get("tabular_dim")
    calibrator = bundle.get("calibrator")

    seed_state_dicts = bundle.get("seed_state_dicts") or {}
    best_seed_state_dict = bundle.get("best_seed_state_dict")

    if tab_cols is None or text_dim is None or tabular_dim is None or calibrator is None:
        raise RuntimeError(f"Gate bundle at {path} is missing required fields for inference.")

    if emb_cols is None:
        raise RuntimeError(
            f"Gate bundle at {path} does not contain emb_cols; cannot map embeddings. "
            f"Re-train gate with updated code to generate an inference-ready bundle."
        )

    if not seed_state_dicts and best_seed_state_dict is None:
        raise RuntimeError(
            f"Gate bundle at {path} does not contain model weights (seed_state_dicts / best_seed_state_dict). "
            f"It likely only contains test predictions. Re-train gate to enable new-patient inference."
        )

    spec = GateBundleSpec(
        tab_cols=list(tab_cols),
        emb_cols=list(emb_cols),
        text_dim=int(text_dim),
        tabular_dim=int(tabular_dim),
        calibrator=calibrator,
        seed_state_dicts={int(k): v for k, v in seed_state_dicts.items()},
        best_seed_state_dict=best_seed_state_dict,
        gate_hidden_dim=int(bundle.get("gate_hidden_dim", GATE_HIDDEN_DIM)),
        gate_dropout=float(bundle.get("gate_dropout", GATE_DROPOUT)),
    )
    return spec, bundle


class GatePredictor:
    """
    Inference wrapper for ACAGN-Gate.

    Uses:
      - seed_state_dicts (preferred; averages probabilities across seeds), or
      - best_seed_state_dict (fallback; single model)

    Expects a fully-populated feature dict containing:
      - gate tabular features for spec.tab_cols
      - embedding features for spec.emb_cols (ct5_0..ct5_511 + metadata columns)
    """

    def __init__(
        self,
        bundle_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.spec, self._bundle = load_gate_bundle(bundle_path=bundle_path)
        self._models = self._build_models()

    def _build_models(self) -> List[torch.nn.Module]:
        models: List[torch.nn.Module] = []
        if self.spec.seed_state_dicts:
            items: Iterable[dict] = self.spec.seed_state_dicts.values()
        else:
            items = [self.spec.best_seed_state_dict]  # type: ignore[list-item]

        for state in items:
            model = TextGuidedGate(
                text_dim=self.spec.text_dim,
                tabular_dim=self.spec.tabular_dim,
                hidden_dim=self.spec.gate_hidden_dim,
                dropout=self.spec.gate_dropout,
            ).to(self.device)
            model.load_state_dict(_state_dict_from_numpy(state))
            model.eval()
            models.append(model)

        return models

    def predict_proba_from_full(self, full: dict) -> float:
        text_vec = np.asarray([full.get(c, 0.0) for c in self.spec.emb_cols], dtype=np.float32)
        tab_vec = np.asarray([full.get(c, 0.0) for c in self.spec.tab_cols], dtype=np.float32)

        text_t = torch.from_numpy(text_vec).unsqueeze(0).to(self.device)
        tab_t = torch.from_numpy(tab_vec).unsqueeze(0).to(self.device)

        probs = []
        with torch.no_grad():
            for m in self._models:
                p, _ = m(text_t, tab_t)
                probs.append(float(p.item()))

        raw = float(np.mean(probs)) if probs else 0.0
        cal = float(self.spec.calibrator.predict(np.array([raw], dtype=np.float32))[0])
        return cal
