---
title: AFL Predictor & Betting Edge
emoji: 🏉
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
license: mit
---

# 🏉 AFL Predictor & Betting Edge

Predicts AFL match winners and margins, checks bookmaker odds for value, and shows
an honest track record.

## What's in the app

| Tab | What it does |
|---|---|
| 📅 This Round | Tips for every game in a round, loaded automatically. Played games show ✅/❌. |
| 🔮 Match Predictor | Pick any two teams and a ground. Shows win %, margin, *why*, and an optional value check against bookmaker odds. |
| 📈 ELO Ladder | Current team strength ratings and recent form. |
| 🎯 Track Record | Season-by-season test results vs ELO-only and bookmakers. |

## Model

- **Features** (all computed only from games *before* the match, so no leakage):
  ELO rating difference, interstate travel, experience at the ground (last 2 years),
  last-5 form and scoring, recent head-to-head, days of rest, finals flag.
- **Margin model**: Ridge regression → predicted margin.
  Win chance = Normal CDF(margin / σ), σ ≈ 36 points.
- **ELO settings** were tuned on 2003–2014 only.

### Test results (each season predicted by a model trained only on earlier seasons)

| 2015–2025 | Tips correct | Log loss | Brier |
|---|---|---|---|
| **Our model (Ridge)** | 67.5% | **0.599** | **0.205** |
| ELO only | 67.9% | 0.601 | 0.206 |
| XGBoost (same features) | 67.2% | 0.608 | 0.209 |
| Bookmakers (2015–2023 only) | 69.2% | 0.581 | 0.197 |

Our model on the same games as the bookmakers: 68.0%, log loss 0.599.
Ridge was kept over XGBoost: better calibrated, lighter, and no less accurate.
Bookmakers are still better — the value check in the app says so.

## Data

- Live results and fixtures: [Squiggle API](https://api.squiggle.com.au) (cached; refreshed every 3 hours).
- Backup history `data/games_history.csv` (2000 – early 2025), used if Squiggle is down:
  built from AFL Tables data in [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) (MIT).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Set `SQUIGGLE_USER_AGENT` to something with your contact details (Squiggle asks for this).

## Developer scripts

```bash
# rebuild the backup history file
python scripts/build_history.py /path/to/AFL-Data-Analysis

# re-run the season-by-season test (writes data/backtest_summary.csv)
pip install xgboost openpyxl
python scripts/backtest.py /path/to/AFL-Data-Analysis/odds_data/odds_data_2009_to_present.xlsx

# tests
pip install pytest && pytest -q
```

## Responsible gambling

18+ only. For fun and learning, not financial advice.
Gambling Help Online: **1800 858 858** · [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au)
