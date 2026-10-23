"""Ties data, features and model together for the app."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import completed, load_games
from .features import build_features
from .model import MarginModel, score, walk_forward
from .teams import TEAMS, short

MAIN_GROUND = {
    "Adelaide": "Adelaide Oval", "Port Adelaide": "Adelaide Oval",
    "Brisbane Lions": "Gabba", "Gold Coast": "Carrara",
    "Carlton": "M.C.G.", "Collingwood": "M.C.G.", "Hawthorn": "M.C.G.",
    "Melbourne": "M.C.G.", "Richmond": "M.C.G.",
    "Essendon": "Docklands", "North Melbourne": "Docklands",
    "St Kilda": "Docklands", "Western Bulldogs": "Docklands",
    "Fremantle": "Perth Stadium", "West Coast": "Perth Stadium",
    "Geelong": "Kardinia Park", "Greater Western Sydney": "Sydney Showground",
    "Sydney": "S.C.G.",
}

# Plain-English labels for the "why" list.
FACTOR_TEXT = {
    "elo_diff": "Overall team strength (ELO rating)",
    "travel": "Travel / home state advantage",
    "venue_exp_diff": "Experience at this ground",
    "form_diff": "Recent form (last 5 games)",
    "attack_diff": "Recent scoring (last 5 games)",
    "h2h": "Recent head-to-head results",
    "rest_diff": "Days of rest",
    "is_final": "Finals game",
}

REFRESH_SECONDS = 3 * 60 * 60


@dataclass
class Engine:
    games: pd.DataFrame
    feat: pd.DataFrame
    state: object
    model: MarginModel
    source: str
    built_at: float = field(default_factory=time.time)
    season_wf: pd.DataFrame | None = None

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, prefer_live: bool = True) -> "Engine":
        games, source = load_games(prefer_live)
        feat, state = build_features(games)
        model = MarginModel().fit(feat.dropna(subset=["margin"]))
        eng = cls(games, feat, state, model, source)
        eng.season_wf = eng._recent_walk_forward()
        return eng

    def _recent_walk_forward(self) -> pd.DataFrame:
        """Predictions for the last two seasons, each from a model trained on earlier seasons only."""
        done = completed(self.feat)
        if done.empty:
            return done
        last = int(done["year"].max())
        return walk_forward(self.feat, last - 1, last)

    # ------------------------------------------------------------ rounds
    def _predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy()
        out["pred_margin"] = self.model.predict_margin(rows)
        out["prob"] = self.model.predict_proba(rows)
        return out

    def round_options(self) -> list[str]:
        """Rounds that have any games, newest season first: 'YEAR | Round name'."""
        g = (self.games.groupby(["year", "round_name"], as_index=False)["date"].min()
             .sort_values("date", ascending=False))
        return [f"{y} | {n}" for y, n in zip(g["year"], g["round_name"])][:60]

    def default_round(self) -> str:
        up = self.games[self.games["complete"] < 100]
        pick = up.iloc[0] if len(up) else self.games.iloc[-1]
        return f"{pick['year']} | {pick['round_name']}"

    def round_table(self, label: str) -> tuple[pd.DataFrame, str]:
        year, name = label.split(" | ", 1)
        mask = (self.feat["year"] == int(year)) & (self.feat["round_name"] == name)
        rows = self.feat[mask]
        if rows.empty:
            return pd.DataFrame(), "No games found for that round."

        # Completed games use the honest walk-forward prediction when we have it
        # (model trained on earlier seasons). Games still to play use the latest model.
        pred = self._predict_rows(rows)
        if self.season_wf is not None and not self.season_wf.empty:
            wf = self.season_wf.set_index("game_id")[["pred_margin", "prob"]]
            hit = pred["game_id"].isin(wf.index) & pred["margin"].notna()
            pred.loc[hit, ["pred_margin", "prob"]] = wf.loc[pred.loc[hit, "game_id"]].to_numpy()

        table, right, played = [], 0, 0
        for r in pred.itertuples():
            tip = r.home if r.prob >= 0.5 else r.away
            p_tip = max(r.prob, 1 - r.prob)
            row = {
                "Date": r.date.strftime("%a %d %b %H:%M"),
                "Match": f"{short(r.home)} v {short(r.away)}",
                "Venue": r.venue,
                "Tip": short(tip),
                "Win %": f"{p_tip:.0%}",
                "Tip margin": f"{abs(r.pred_margin):.0f} pts",
                "Result": "",
            }
            if pd.notna(r.margin):
                winner = r.home if r.margin > 0 else r.away if r.margin < 0 else None
                played += 1
                if winner is None:
                    row["Result"] = f"Draw {int(r.hscore)}-{int(r.ascore)}"
                    right += 0.5
                else:
                    ok = winner == tip
                    right += ok
                    row["Result"] = (f"{'✅' if ok else '❌'} {short(winner)} by {abs(int(r.margin))}")
            table.append(row)

        if played:
            note = f"**{label}** — tipped {right:g} of {played} played games correctly."
        else:
            note = f"**{label}** — games not played yet. Tips use the latest model."
        return pd.DataFrame(table), note

    # ------------------------------------------------------------ single match
    def predict_match(self, home: str, away: str, venue: str, is_final: bool = False) -> dict:
        now = pd.Timestamp.now(tz="Australia/Melbourne").tz_localize(None)
        last = self.state.last_game_date or now
        date = max(now, last + pd.Timedelta(days=7))
        season, off_season = self.next_game_season(now)
        f = self.state.features(home, away, venue, date, season, int(is_final))
        row = pd.DataFrame([f])
        margin = float(self.model.predict_margin(row)[0])
        prob = float(self.model.predict_proba(row)[0])
        contrib = self.model.contributions(row)
        return {"home": home, "away": away, "venue": venue, "prob_home": prob,
                "margin": margin, "features": f, "contrib": contrib,
                "season": season, "off_season": off_season}

    def next_game_season(self, now: pd.Timestamp | None = None) -> tuple[int, bool]:
        """Season the next game belongs to, and whether we are in the off-season.

        - Games still to play in the data: their season (ratings move back toward
          average automatically if that is a new season).
        - No games left and the last game was over 2 weeks ago: off-season, so use
          next season's start-of-season ratings.
        - Otherwise: the current season.
        """
        now = now or pd.Timestamp.now(tz="Australia/Melbourne").tz_localize(None)
        current = self.state.elo.season or int(now.year)
        up = self.games[self.games["complete"] < 100]
        if len(up):
            nxt = int(up["year"].iloc[0])
            return nxt, nxt > current
        last = self.state.last_game_date
        if last is not None and (now - last).days > 14:
            return current + 1, True
        return current, False

    # ------------------------------------------------------------ tables
    def elo_ladder(self) -> pd.DataFrame:
        season, off = self.next_game_season()
        r = sorted(((t, self.state.elo.rating(t)) for t in TEAMS), key=lambda x: -x[1])
        rows = []
        for i, (t, rating) in enumerate(r, 1):
            row = {"#": i, "Team": t, "ELO (end of last season)" if off else "ELO": round(rating)}
            if off:
                row[f"ELO (start of {season})"] = round(self.state.elo.rating(t, season))
            rows.append({**row,
                         "Last 5 avg margin": round(self.state._form(t), 1),
                         "Last 5 avg score": round(self.state._attack(t), 1)})
        return pd.DataFrame(rows)

    def recent_season_scores(self) -> pd.DataFrame:
        wf = self.season_wf
        if wf is None or wf.empty:
            return pd.DataFrame()
        rows = []
        for year, part in wf.groupby("year"):
            s = score(part)
            rows.append({"Season": int(year), "Games": s["games"],
                         "Tips correct": f"{s['accuracy']:.1%}",
                         "Avg margin error": f"{s['margin_mae']:.1f} pts",
                         "Brier score": round(s["brier"], 3)})
        return pd.DataFrame(rows)

    def venues(self) -> list[str]:
        recent = self.games[self.games["year"] >= self.games["year"].max() - 3]
        return sorted(set(recent["venue"].dropna()) | set(MAIN_GROUND.values()))


# ---------------------------------------------------------------- cached access
_lock = threading.Lock()
_engine: Engine | None = None


def get_engine(prefer_live: bool = True) -> Engine:
    """Build once, then rebuild at most every 3 hours so new results flow in."""
    global _engine
    with _lock:
        if _engine is None or time.time() - _engine.built_at > REFRESH_SECONDS:
            try:
                _engine = Engine.build(prefer_live)
            except Exception:
                if _engine is None:
                    raise
                _engine.built_at = time.time()  # keep serving the old one, retry later
        return _engine


def value_check(prob_home: float, home_odds: float | None, away_odds: float | None,
                min_edge: float = 0.05) -> list[dict]:
    """Compare our probabilities with bookmaker decimal odds.

    Returns one row per side with: bookie %, our %, edge, expected return per $1.
    """
    if not home_odds or not away_odds or home_odds <= 1 or away_odds <= 1:
        return []
    ih, ia = 1 / home_odds, 1 / away_odds
    overround = ih + ia
    rows = []
    for side, ours, odds, implied in (("home", prob_home, home_odds, ih),
                                      ("away", 1 - prob_home, away_odds, ia)):
        fair = implied / overround
        ev = ours * odds - 1
        rows.append({"side": side, "odds": odds, "bookie_pct": fair, "ours_pct": ours,
                     "edge": ours - fair, "ev": ev,
                     "value": (ours - fair) >= min_edge and ev > 0})
    rows.append({"side": "margin", "overround": overround - 1})
    return rows
