import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afl.data import _from_squiggle, _finish, load_history_csv  # noqa: E402
from afl.engine import Engine, value_check  # noqa: E402
from afl.features import build_features  # noqa: E402
from afl.model import MarginModel, score, walk_forward  # noqa: E402
from afl.teams import travel_edge  # noqa: E402


@pytest.fixture(scope="module")
def feat():
    f, _ = build_features(load_history_csv())
    return f


def test_history_is_clean():
    g = load_history_csv()
    assert len(g) > 4500
    assert g["home"].nunique() == 18
    assert g["date"].is_monotonic_increasing
    assert (g["hscore"] >= 0).all()


def test_no_leakage(feat):
    """Changing a game's result must not change that game's own features."""
    g = load_history_csv()
    i = 3000
    g2 = g.copy()
    g2.loc[i, "hscore"] += 100
    f2, _ = build_features(g2)
    cols = ["elo_diff", "form_diff", "attack_diff", "h2h", "venue_exp_diff"]
    assert np.allclose(feat.loc[i, cols].astype(float), f2.loc[i, cols].astype(float))
    # ...but the next games of those teams do change
    later = f2.index > i
    assert not np.allclose(feat.loc[later, "elo_diff"], f2.loc[later, "elo_diff"])


def test_travel_edge():
    assert travel_edge("West Coast", "Carlton", "Perth Stadium") == 1
    assert travel_edge("Carlton", "Collingwood", "M.C.G.") == 0
    assert travel_edge("Hawthorn", "Adelaide", "York Park") == 1  # Hawthorn's second home


def test_walk_forward_is_sensible(feat):
    wf = walk_forward(feat, 2022, 2023)
    s = score(wf)
    assert 0.6 < s["accuracy"] < 0.8
    assert s["brier"] < 0.23
    assert set(wf["year"]) == {2022, 2023}


def test_model_probabilities(feat):
    m = MarginModel().fit(feat.dropna(subset=["margin"]))
    p = m.predict_proba(feat.tail(50))
    assert ((p > 0) & (p < 1)).all()
    # positive predicted margin <=> favourite
    assert ((m.predict_margin(feat.tail(50)) > 0) == (p > 0.5)).all()


def test_value_check():
    rows = value_check(0.85, 1.25, 4.0)  # bookie says ~76%, we say 85%
    home, away, margin = rows
    assert abs(home["bookie_pct"] + away["bookie_pct"] - 1) < 1e-9
    assert abs(margin["overround"] - 0.05) < 1e-9
    assert home["value"] and not away["value"]
    assert value_check(0.5, None, 2.0) == []


def test_squiggle_parsing():
    sample = [
        {"id": 1, "year": 2026, "round": 1, "roundname": "Round 1", "is_final": 0,
         "date": "2026-03-12 19:30:00", "venue": "M.C.G.", "hteam": "Carlton",
         "ateam": "Richmond", "hscore": 90, "ascore": 70, "complete": 100},
        {"id": 2, "year": 2026, "round": 2, "roundname": "Round 2", "is_final": 0,
         "date": "2026-03-19 19:30:00", "venue": "Docklands", "hteam": "Kangaroos",
         "ateam": "GWS Giants", "hscore": None, "ascore": None, "complete": 0},
        {"id": 3, "year": 2026, "round": 25, "roundname": "Grand Final", "is_final": 6,
         "date": "2026-09-26 14:30:00", "venue": "M.C.G.", "hteam": None, "ateam": None,
         "complete": 0},
    ]
    df = _finish(_from_squiggle(sample))
    assert len(df) == 2  # placeholder final without teams dropped
    assert df.loc[1, "home"] == "North Melbourne"
    assert df.loc[1, "away"] == "Greater Western Sydney"


def test_engine_offline():
    e = Engine.build(prefer_live=False)
    table, note = e.round_table("2024 | Round 10")
    assert len(table) == 9
    assert "tipped" in note
    p = e.predict_match("West Coast", "Carlton", "Perth Stadium")
    p_away = e.predict_match("Carlton", "West Coast", "Perth Stadium")
    assert p["contrib"]["travel"] > 0 > p_away["contrib"]["travel"]
    assert len(e.elo_ladder()) == 18


def test_off_season_uses_next_season_ratings():
    e = Engine.build(prefer_live=False)
    last = e.state.last_game_date
    season = e.state.elo.season
    # Mid-season (a few days after the last game): current season, no adjustment.
    assert e.next_game_season(last + pd.Timedelta(days=5)) == (season, False)
    # Long after the last game with nothing left to play: next season.
    assert e.next_game_season(last + pd.Timedelta(days=60)) == (season + 1, True)
    p = e.predict_match("Sydney", "Geelong", "S.C.G.")  # "now" is long after the bundled data
    assert p["off_season"] and p["season"] == season + 1
    raw = e.state.elo.rating("Sydney")
    assert abs(p["features"]["elo_home"] - (1500 + 0.85 * (raw - 1500))) < 1e-9
    assert "ELO (start of" in " ".join(e.elo_ladder().columns)


def test_prediction_does_not_change_state():
    e = Engine.build(prefer_live=False)
    before = e.elo_ladder()
    first = e.predict_match("Sydney", "Geelong", "S.C.G.")
    e.predict_match("Carlton", "Richmond", "M.C.G.")
    assert e.elo_ladder().equals(before)
    assert e.predict_match("Sydney", "Geelong", "S.C.G.")["prob_home"] == first["prob_home"]
