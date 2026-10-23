"""Build data/games_history.csv from the AFL Tables-based dataset by Adam Kareen
(github.com/akareen/AFL-Data-Analysis, MIT License).

This file is only a fallback for when the Squiggle API can't be reached.

Usage:
    python scripts/build_history.py /path/to/AFL-Data-Analysis
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from afl.teams import canonical_team  # noqa: E402

FINALS_STEP = {"Qualifying Final": 1, "Elimination Final": 1, "Semi Final": 2,
               "Preliminary Final": 3, "Grand Final": 4}


def build(src: Path, first_year: int = 2000) -> pd.DataFrame:
    frames = []
    for f in sorted((src / "data" / "matches").glob("matches_*.csv")):
        year = int(f.stem.split("_")[1])
        if year < first_year:
            continue
        d = pd.read_csv(f, dtype={"round_num": str})
        d = d.dropna(subset=["team_1_team_name", "team_2_team_name", "team_1_final_goals"])
        is_final = ~d["round_num"].str.isdigit()
        last_round = d.loc[~is_final, "round_num"].astype(int).max()
        rnd = d["round_num"].where(~is_final, None)
        rnd = rnd.fillna(d["round_num"].map(FINALS_STEP).add(last_round).astype("Int64").astype(str))
        out = pd.DataFrame({
            "date": pd.to_datetime(d["date"]),
            "year": year,
            "round": rnd.astype(int),
            "round_name": d["round_num"].where(is_final, "Round " + d["round_num"]),
            "is_final": is_final.astype(int),
            "venue": d["venue"],
            "home": d["team_1_team_name"].map(canonical_team),
            "away": d["team_2_team_name"].map(canonical_team),
            "hscore": (d["team_1_final_goals"] * 6 + d["team_1_final_behinds"]).astype(int),
            "ascore": (d["team_2_final_goals"] * 6 + d["team_2_final_behinds"]).astype(int),
            "complete": 100,
        })
        frames.append(out)
    df = pd.concat(frames).sort_values("date").reset_index(drop=True)
    df.insert(0, "game_id", [f"h{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    src = Path(sys.argv[1])
    df = build(src)
    dest = Path(__file__).resolve().parent.parent / "data" / "games_history.csv"
    dest.parent.mkdir(exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"wrote {len(df)} games ({df.year.min()}-{df.year.max()}) to {dest}")
