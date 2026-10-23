"""Load AFL match data.

Two sources, same output table:
  * Squiggle API (live results + upcoming fixtures) - used on Hugging Face.
  * Bundled CSV (data/games_history.csv) - used if Squiggle can't be reached.

Output columns:
  game_id, date, year, round, round_name, is_final, venue,
  home, away, hscore, ascore, complete
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .teams import canonical_team

ROOT = Path(__file__).resolve().parent.parent
HISTORY_CSV = ROOT / "data" / "games_history.csv"
CACHE_DIR = Path(os.getenv("AFL_CACHE_DIR", ROOT / ".cache"))

SQUIGGLE_URL = "https://api.squiggle.com.au/"
# Squiggle asks every client to identify itself with a contact in the User-Agent,
# and answers 403 to generic or anonymous ones. Set SQUIGGLE_USER_AGENT (a Space
# secret) to something like "AFL-UI-Predictor (you@example.com)".
USER_AGENT = os.getenv("SQUIGGLE_USER_AGENT",
                       "AFL-UI-Predictor/2.0 (+https://github.com/mizzony/afl-ui)")
FIRST_YEAR = 2000
LIVE_REFRESH_SECONDS = 3 * 60 * 60

COLUMNS = ["game_id", "date", "year", "round", "round_name", "is_final", "venue",
           "home", "away", "hscore", "ascore", "complete"]


# ---------------------------------------------------------------- Squiggle
def _squiggle_year(year: int, current_year: int, timeout: int = 20) -> list[dict]:
    """Games for one season, cached on disk. Past seasons never re-download."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"squiggle_{year}.json"
    live = year >= current_year
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if not live or age < LIVE_REFRESH_SECONDS:
            return json.loads(path.read_text())
    # Build the query by hand: Squiggle's documented format uses a literal ";",
    # which `params=` would encode as %3B.
    resp = requests.get(f"{SQUIGGLE_URL}?q=games;year={year}",
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                        timeout=timeout)
    if resp.status_code != 200:
        raise requests.HTTPError(
            f"Squiggle {resp.status_code} for {year} (User-Agent: {USER_AGENT!r}): "
            f"{resp.text[:200]!r}")
    games = resp.json().get("games", [])
    path.write_text(json.dumps(games))
    return games


def _from_squiggle(games: list[dict]) -> pd.DataFrame:
    rows = []
    for g in games:
        if not g.get("hteam") or not g.get("ateam"):
            continue  # finals placeholders before teams are known
        rows.append({
            "game_id": f"sq{g['id']}",
            "date": g.get("date"),
            "year": int(g["year"]),
            "round": int(g["round"]) if g.get("round") is not None else None,
            "round_name": g.get("roundname") or f"Round {g.get('round')}",
            "is_final": int(bool(g.get("is_final"))),
            "venue": g.get("venue") or "",
            "home": canonical_team(g["hteam"]),
            "away": canonical_team(g["ateam"]),
            "hscore": g.get("hscore"),
            "ascore": g.get("ascore"),
            "complete": int(g.get("complete") or 0),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def load_squiggle(first_year: int = FIRST_YEAR, last_year: int | None = None) -> pd.DataFrame:
    now_year = pd.Timestamp.now(tz="Australia/Melbourne").year
    last_year = last_year or now_year + 1  # next year's fixture appears late in the year
    games = []
    for y in range(first_year, last_year + 1):
        try:
            games.extend(_squiggle_year(y, now_year))
        except requests.RequestException:
            if y <= now_year:
                raise  # history is required; a missing future fixture is fine
    return _finish(_from_squiggle(games))


# ---------------------------------------------------------------- bundled CSV
def load_history_csv(path: Path = HISTORY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _finish(df[COLUMNS])


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("hscore", "ascore"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    return df


def load_games(prefer_live: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (games, source_label). Falls back to the bundled CSV if Squiggle fails."""
    if prefer_live:
        try:
            return load_squiggle(), "Squiggle API (live)"
        except Exception as exc:  # network down, API change, rate limit...
            print(f"[data] Squiggle unavailable ({exc!r}); using bundled history")
    return load_history_csv(), "bundled history file (offline)"


def completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["complete"] >= 100) & df["hscore"].notna() & df["ascore"].notna()]


def upcoming(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["complete"] < 100]
