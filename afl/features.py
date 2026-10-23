"""Pre-game features, computed in date order so a game never sees its own result
or any later result (no leakage).

Features (all from the HOME team's point of view):
  elo_diff         home ELO - away ELO before the game
  travel           +1 away team travelled interstate, -1 home team did, 0 neither/both
  venue_exp_diff   games at this venue in the last 2 years: home - away
  form_diff        average margin over each team's last 5 games: home - away
  attack_diff      average points scored over last 5 games: home - away
  h2h              recent head-to-head margin (last 5 meetings, 6 years, shrunk to 0)
  rest_diff        days since each team's last game (capped at 21): home - away
  is_final         1 for finals
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .teams import travel_edge

FEATURES = ["elo_diff", "travel", "venue_exp_diff", "form_diff", "attack_diff",
            "h2h", "rest_diff", "is_final"]


@dataclass
class EloParams:
    # Tuned by grid search on 2003-2014 only (later seasons kept clean for testing).
    k: float = 90.0             # update speed
    home_base: float = 20.0     # rating points for being the home team (same state)
    travel_bonus: float = 80.0  # extra rating points per unit of travel edge
    margin_scale: float = 20.0  # how fast a bigger margin counts as a "fuller" win
    carry: float = 0.85         # share of rating kept between seasons (rest goes to mean)
    new_team: float = 1380.0    # starting rating for an expansion team


class Elo:
    MEAN = 1500.0

    def __init__(self, p: EloParams | None = None):
        self.p = p or EloParams()
        self.r: dict[str, float] = {}
        self.season: int | None = None

    def rating(self, team: str, year: int | None = None) -> float:
        """Current rating. For a later season, the start-of-season value is returned
        (moved part way back to average) without changing the stored rating."""
        if team not in self.r:
            self.r[team] = self.p.new_team if self.r else self.MEAN
            return self.r[team]  # a brand-new team starts at its starting rating
        r = self.r[team]
        if year is not None and self.season is not None and year > self.season:
            r = self.MEAN + self.p.carry * (r - self.MEAN)
        return r

    def new_season(self, year: int):
        if self.season is not None and year > self.season:
            for t in self.r:
                self.r[t] = self.MEAN + self.p.carry * (self.r[t] - self.MEAN)
        if self.season is None or year > self.season:
            self.season = year

    def edge(self, home: str, away: str, travel: int, year: int | None = None) -> float:
        return (self.rating(home, year) - self.rating(away, year)
                + self.p.home_base + self.p.travel_bonus * travel)

    def prob(self, home: str, away: str, travel: int, year: int | None = None) -> float:
        return 1.0 / (1.0 + 10 ** (-self.edge(home, away, travel, year) / 400.0))

    def update(self, home: str, away: str, travel: int, margin: float):
        expected = self.prob(home, away, travel)
        actual = 1.0 / (1.0 + math.exp(-margin / self.p.margin_scale))
        delta = self.p.k * (actual - expected)
        self.r[home] = self.rating(home) + delta
        self.r[away] = self.rating(away) - delta


class FeatureState:
    """Everything the features need, updated one completed game at a time."""

    def __init__(self, elo_params: EloParams | None = None):
        self.elo = Elo(elo_params)
        self.last5 = defaultdict(lambda: deque(maxlen=5))      # team -> margins
        self.last5_for = defaultdict(lambda: deque(maxlen=5))  # team -> points scored
        self.last_date: dict[str, pd.Timestamp] = {}
        self.venue_hist = defaultdict(list)                    # team -> [(date, venue)]
        self.h2h_hist = defaultdict(list)                      # sorted pair -> [(date, margin for pair[0])]
        self.last_game_date: pd.Timestamp | None = None

    # --- helpers
    def _venue_games(self, team, venue, d):
        return sum(1 for (dt, v) in self.venue_hist[team] if v == venue and (d - dt).days <= 730)

    def _form(self, team):
        return float(np.mean(self.last5[team])) if self.last5[team] else 0.0

    def _attack(self, team):
        return float(np.mean(self.last5_for[team])) if self.last5_for[team] else 85.0

    def _rest(self, team, d):
        if team not in self.last_date:
            return 21.0
        return float(min(max((d - self.last_date[team]).days, 0), 21))

    # --- public
    def features(self, home: str, away: str, venue: str, date: pd.Timestamp,
                 year: int, is_final: int = 0) -> dict:
        """Features for a game. Read-only: never changes the state."""
        trav = travel_edge(home, away, venue)
        key = tuple(sorted((home, away)))
        recent = [m for (dt, m) in self.h2h_hist[key] if (date - dt).days <= 6 * 365][-5:]
        sign = 1.0 if key[0] == home else -1.0
        eh, ea = self.elo.rating(home, year), self.elo.rating(away, year)
        return {
            "elo_home": eh,
            "elo_away": ea,
            "elo_diff": eh - ea,
            "elo_prob": self.elo.prob(home, away, trav, year),
            "travel": trav,
            "venue_exp_diff": self._venue_games(home, venue, date) - self._venue_games(away, venue, date),
            "form_diff": self._form(home) - self._form(away),
            "attack_diff": self._attack(home) - self._attack(away),
            "h2h": sign * sum(recent) / (len(recent) + 3.0),  # shrink toward 0 when few games
            "rest_diff": self._rest(home, date) - self._rest(away, date),
            "is_final": int(is_final),
        }

    def update(self, home: str, away: str, venue: str, date: pd.Timestamp, year: int,
               hscore: float, ascore: float):
        self.elo.new_season(year)
        trav = travel_edge(home, away, venue)
        margin = float(hscore - ascore)
        self.elo.update(home, away, trav, margin)
        self.last5[home].append(margin)
        self.last5[away].append(-margin)
        self.last5_for[home].append(float(hscore))
        self.last5_for[away].append(float(ascore))
        self.last_date[home] = self.last_date[away] = date
        self.venue_hist[home].append((date, venue))
        self.venue_hist[away].append((date, venue))
        key = tuple(sorted((home, away)))
        self.h2h_hist[key].append((date, margin if key[0] == home else -margin))
        self.last_game_date = date


def build_features(games: pd.DataFrame, elo_params: EloParams | None = None,
                   ) -> tuple[pd.DataFrame, FeatureState]:
    """Return (games + feature columns + margin, state after all completed games).

    `games` must be sorted by date. Incomplete games get features from the latest
    state but do not update it.
    """
    state = FeatureState(elo_params)
    rows = []
    for g in games.itertuples(index=False):
        rows.append(state.features(g.home, g.away, g.venue, g.date, g.year, g.is_final))
        if g.complete >= 100 and pd.notna(g.hscore) and pd.notna(g.ascore):
            state.update(g.home, g.away, g.venue, g.date, g.year, g.hscore, g.ascore)

    feats = pd.DataFrame(rows, index=games.index).drop(columns="is_final")
    out = pd.concat([games, feats], axis=1)
    out["is_final"] = out["is_final"].astype(int)
    out["margin"] = out["hscore"] - out["ascore"]
    return out, state
