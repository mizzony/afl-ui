"""Fair, season-by-season test of the model.

For each season from 2015, every model is trained only on earlier seasons and
then predicts that season. Compares:
  * ELO only
  * Ridge margin model (what the app uses)
  * XGBoost margin model
  * Bookmakers (average odds, overround removed) - seasons with odds data only

Writes data/backtest_summary.csv, which the app shows on the Track Record tab.

Usage:
    python scripts/backtest.py [path/to/odds_data_2009_to_present.xlsx]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from afl.data import load_history_csv  # noqa: E402
from afl.features import FEATURES, build_features  # noqa: E402
from afl.model import TRAIN_FROM, MarginModel, _phi, score, walk_forward  # noqa: E402
from afl.teams import canonical_team  # noqa: E402

FIRST_TEST = 2015


def xgb_walk_forward(feat: pd.DataFrame) -> pd.DataFrame:
    from xgboost import XGBRegressor
    done = feat.dropna(subset=["margin"])
    parts = []
    for year in range(FIRST_TEST, int(done.year.max()) + 1):
        train = done[(done.year < year) & (done.year >= TRAIN_FROM)]
        test = done[done.year == year].copy()
        m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
        m.fit(train[FEATURES], train["margin"])
        sigma = max(float((train["margin"] - m.predict(train[FEATURES])).std()), 30.0)
        test["pred_margin"] = m.predict(test[FEATURES])
        test["prob"] = _phi(test["pred_margin"] / sigma)
        parts.append(test)
    return pd.concat(parts)


def add_bookies(df: pd.DataFrame, odds_path: Path) -> pd.DataFrame:
    o = pd.read_excel(odds_path)
    o = pd.DataFrame({
        "day": pd.to_datetime(o["Date"]).dt.normalize(),
        "home": o["Home Team"].map(canonical_team),
        "away": o["Away Team"].map(canonical_team),
        "home_odds": o["Home Odds"], "away_odds": o["Away Odds"],
    })
    df = df.assign(day=df["date"].dt.normalize())
    out = df.merge(o, on=["day", "home", "away"], how="left")
    ih, ia = 1 / out["home_odds"], 1 / out["away_odds"]
    out["bookie_prob"] = ih / (ih + ia)
    return out


def main():
    games = load_history_csv()
    feat, _ = build_features(games)
    results = {
        "ELO only": walk_forward(feat, FIRST_TEST).assign(prob=lambda d: d["elo_prob"]),
        "Our model (Ridge)": walk_forward(feat, FIRST_TEST),
        "XGBoost": xgb_walk_forward(feat),
    }
    rows = []
    for name, df in results.items():
        for year, part in df.groupby("year"):
            rows.append({"model": name, "year": year,
                         **score(part, margin_col=None if name == "ELO only" else "pred_margin")})
        rows.append({"model": name, "year": "All",
                     **score(df, margin_col=None if name == "ELO only" else "pred_margin")})

    odds_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if odds_path and odds_path.exists():
        ours = add_bookies(results["Our model (Ridge)"], odds_path)
        both = ours.dropna(subset=["bookie_prob"])
        print(f"\nGames with bookmaker odds: {len(both)} of {len(ours)}")
        for year, part in both.groupby("year"):
            rows.append({"model": "Bookmakers", "year": year,
                         **score(part, prob_col="bookie_prob", margin_col=None)})
            rows.append({"model": "Our model (same games)", "year": year, **score(part)})
        rows.append({"model": "Bookmakers", "year": "All",
                     **score(both, prob_col="bookie_prob", margin_col=None)})
        rows.append({"model": "Our model (same games)", "year": "All", **score(both)})

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(summary[summary.year == "All"].round(4).to_string(index=False))
    print()
    print(summary.pivot(index="year", columns="model", values="accuracy").round(3).to_string())

    # Calibration check for our model: predicted vs actual win rate by bucket.
    ridge = results["Our model (Ridge)"]
    bins = pd.cut(ridge["prob"], [0, .2, .35, .5, .65, .8, 1])
    calib = ridge.groupby(bins, observed=True).agg(predicted=("prob", "mean"),
                                                   actual=("margin", lambda m: (m > 0).mean()),
                                                   games=("prob", "size"))
    print("\nCalibration (our model):\n", calib.round(3).to_string())

    out = summary.copy()
    out["year"] = out["year"].astype(str)
    out.round(4).to_csv(ROOT / "data" / "backtest_summary.csv", index=False)
    print(f"\nwrote {ROOT / 'data' / 'backtest_summary.csv'}")


if __name__ == "__main__":
    main()
