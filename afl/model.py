"""Margin model: predicts home margin, turns it into a win probability.

margin  = linear model on the pre-game features (Ridge regression)
P(win)  = Normal CDF(predicted margin / sigma), sigma = spread of past errors

A linear model was chosen over XGBoost after the season-by-season test in
scripts/backtest.py: it was as accurate, better calibrated and far lighter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

TRAIN_FROM = 2003  # 2000-2002 only warm up the ELO ratings


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


@dataclass
class MarginModel:
    alpha: float = 10.0
    min_sigma: float = 30.0

    def fit(self, df: pd.DataFrame) -> "MarginModel":
        df = df[df["year"] >= TRAIN_FROM].dropna(subset=["margin"])
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe.fit(df[FEATURES], df["margin"])
        resid = df["margin"] - self.pipe.predict(df[FEATURES])
        self.sigma = max(float(resid.std()), self.min_sigma)
        self.n_train = len(df)
        return self

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(df[FEATURES])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _phi(self.predict_margin(df) / self.sigma)

    def contributions(self, row: pd.DataFrame) -> dict[str, float]:
        """Points of margin each feature adds for one game (vs an average game)."""
        scaler, ridge = self.pipe.named_steps.values()
        z = (row[FEATURES].to_numpy(dtype=float)[0] - scaler.mean_) / scaler.scale_
        return dict(zip(FEATURES, z * ridge.coef_))


def walk_forward(feat: pd.DataFrame, first_test: int, last_test: int | None = None,
                 **model_kw) -> pd.DataFrame:
    """Each season is predicted by a model trained only on earlier seasons."""
    done = feat.dropna(subset=["margin"])
    last_test = last_test or int(done["year"].max())
    parts = []
    for year in range(first_test, last_test + 1):
        test = done[done["year"] == year]
        if test.empty:
            continue
        m = MarginModel(**model_kw).fit(done[done["year"] < year])
        out = test.copy()
        out["pred_margin"] = m.predict_margin(test)
        out["prob"] = m.predict_proba(test)
        parts.append(out)
    return pd.concat(parts) if parts else done.iloc[0:0]


def score(df: pd.DataFrame, prob_col: str = "prob", margin_col: str | None = "pred_margin") -> dict:
    """Accuracy, log loss, Brier and margin error. Draws count as half."""
    y = np.where(df["margin"] > 0, 1.0, np.where(df["margin"] < 0, 0.0, 0.5))
    p = df[prob_col].clip(1e-6, 1 - 1e-6).to_numpy()
    tip_right = np.where(y == 0.5, 0.5, ((p > 0.5) == (y == 1.0)).astype(float))
    out = {
        "games": len(df),
        "accuracy": float(tip_right.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
    }
    if margin_col and margin_col in df:
        out["margin_mae"] = float((df[margin_col] - df["margin"]).abs().mean())
    return out
