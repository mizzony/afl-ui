"""AFL Predictor & Betting Edge - Gradio app (Hugging Face Space)."""
from pathlib import Path

import gradio as gr
import pandas as pd

from afl.engine import FACTOR_TEXT, MAIN_GROUND, get_engine, value_check
from afl.teams import TEAMS, short

ROOT = Path(__file__).resolve().parent
BACKTEST_CSV = ROOT / "data" / "backtest_summary.csv"

# This app only needs a CPU. If the Space is set to ZeroGPU hardware, it refuses to
# start unless at least one @spaces.GPU function exists, so register a no-op one.
# (Better: set the Space hardware to "CPU basic" in Settings.)
try:
    import spaces

    @spaces.GPU
    def _zero_gpu_placeholder():
        return None
except ImportError:
    pass

engine = get_engine()  # build at start-up so the first click is fast

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#hero h1 {margin-bottom: 0;}
#verdict h2 {margin: 0.2em 0;}
.small-note {font-size: 0.85em; opacity: 0.75;}
"""

RESPONSIBLE = (
    "18+ only. This is a stats project for fun and learning, not financial advice. "
    "No model can see injuries, late changes or weather. "
    "If gambling stops being fun: **Gambling Help Online 1800 858 858** "
    "([gamblinghelponline.org.au](https://www.gamblinghelponline.org.au))."
)


# ------------------------------------------------------------------ This round
def show_round(label):
    eng = get_engine()
    table, note = eng.round_table(label)
    return note, table


def refresh_rounds():
    eng = get_engine()
    label = eng.default_round()
    note, table = show_round(label)
    return gr.update(choices=eng.round_options(), value=label), note, table


# ------------------------------------------------------------------ Match predictor
def default_venue(home):
    return MAIN_GROUND.get(home, "M.C.G.")


def predict(home, away, venue, is_final, home_odds, away_odds):
    if not home or not away:
        return "Pick two teams.", {}, "", ""
    if home == away:
        return "⚠️ Please pick two different teams.", {}, "", ""
    eng = get_engine()
    p = eng.predict_match(home, away, venue or default_venue(home), bool(is_final))
    ph, m = p["prob_home"], p["margin"]
    winner, pw = (home, ph) if ph >= 0.5 else (away, 1 - ph)

    verdict = (f"## 🏆 {winner} to win\n"
               f"**{pw:.0%} chance** · tipped margin **{abs(m):.0f} points** · {p['venue']}")
    if p["off_season"]:
        verdict += (f"\n\n<span class='small-note'>Off-season: using start-of-{p['season']} "
                    "ratings (moved part way back to average, as happens every new season).</span>")
    probs = {home: ph, away: 1 - ph}

    # Why: biggest factors, in points of margin.
    items = sorted(p["contrib"].items(), key=lambda kv: -abs(kv[1]))
    lines = ["### 🧠 Why the model thinks this",
             "Each line shows how many points of margin a factor adds, compared with an average game."]
    for key, pts in items:
        if abs(pts) < 1:
            continue
        helps = home if pts > 0 else away
        lines.append(f"- **{FACTOR_TEXT[key]}**: +{abs(pts):.0f} pts to {short(helps)}")
    f = p["features"]
    lines.append(f"\n<span class='small-note'>ELO: {short(home)} {f['elo_home']:.0f} · "
                 f"{short(away)} {f['elo_away']:.0f}</span>")
    why = "\n".join(lines)

    return verdict, probs, why, value_text(home, away, ph, home_odds, away_odds)


def value_text(home, away, ph, home_odds, away_odds):
    rows = value_check(ph, home_odds, away_odds)
    if not rows:
        return ("### 💰 Value check\nEnter the bookmaker's decimal odds for both teams "
                "(for example 1.65 and 2.30) to compare them with the model.")
    names = {"home": home, "away": away}
    margin = rows[-1]["overround"]
    out = ["### 💰 Value check",
           "| Team | Odds | Bookie's chance | Our chance | Difference |",
           "|---|---|---|---|---|"]
    for r in rows[:2]:
        out.append(f"| {names[r['side']]} | {r['odds']:.2f} | {r['bookie_pct']:.0%} | "
                   f"{r['ours_pct']:.0%} | {r['edge'] * 100:+.0f} pts |")
    out.append(f"\nBookmaker's built-in margin: **{margin:.1%}** "
               "(the 'bookie chance' above has this removed).")
    value = [r for r in rows[:2] if r["value"]]
    if value:
        r = value[0]
        out.append(f"\n✅ **Possible value on {names[r['side']]}**: the model gives them a "
                   f"{r['ours_pct']:.0%} chance, the bookie {r['bookie_pct']:.0%} "
                   f"({r['edge'] * 100:.0f} points higher). "
                   f"Expected return about **{r['ev'] * 100:+.0f}c per $1** if the model is right.")
    else:
        out.append("\n➖ **No value found.** The odds are close to what the model expects "
                   "(we need at least a 5-point gap).")
    out.append("\n<span class='small-note'>Remember: from 2015 to 2023 bookmakers tipped more "
               "winners than this model (69.2% vs 68.0%). Treat a 'value' flag as a reason to look "
               "closer, not a sure thing.</span>")
    return "\n".join(out)


# ------------------------------------------------------------------ Track record
def backtest_frames():
    if not BACKTEST_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    bt = pd.read_csv(BACKTEST_CSV, dtype={"year": str})
    overall = bt[bt["year"] == "All"].copy()
    overall = pd.DataFrame({
        "Model": overall["model"],
        "Games": overall["games"],
        "Tips correct": (overall["accuracy"] * 100).round(1).astype(str) + "%",
        "Log loss (lower = better)": overall["log_loss"].round(3),
        "Brier (lower = better)": overall["brier"].round(3),
    })
    keep = ["Our model (Ridge)", "ELO only", "Bookmakers"]
    by_year = bt[(bt["year"] != "All") & bt["model"].isin(keep)].copy()
    by_year["Tips correct %"] = (by_year["accuracy"] * 100).round(1)
    by_year = by_year.rename(columns={"year": "Season", "model": "Model"})
    return overall, by_year[["Season", "Model", "Tips correct %"]]


# ------------------------------------------------------------------ UI
with gr.Blocks(title="AFL Predictor & Betting Edge") as app:
    gr.Markdown(
        "# 🏉 AFL Predictor & Betting Edge\n"
        "Win chances and margins from ELO ratings, recent form, travel, ground experience "
        "and head-to-head — tested season by season against real results and bookmakers.",
        elem_id="hero",
    )
    gr.Markdown(f"<span class='small-note'>Data: {engine.source}. "
                f"Model trained on {engine.model.n_train:,} games since 2003.</span>")

    with gr.Tabs():
        # ---------------- This round
        with gr.Tab("📅 This Round"):
            with gr.Row():
                round_dd = gr.Dropdown(engine.round_options(), value=engine.default_round(),
                                       label="Round", scale=4)
                refresh_btn = gr.Button("🔄 Latest data", scale=1)
            first_note, first_table = show_round(engine.default_round())
            round_note = gr.Markdown(first_note)
            round_table = gr.Dataframe(first_table, interactive=False, wrap=True)
            gr.Markdown("<span class='small-note'>Played games show the prediction a model "
                        "trained only on earlier seasons would have made, so the ✅/❌ are fair.</span>")
            round_dd.change(show_round, round_dd, [round_note, round_table])
            refresh_btn.click(refresh_rounds, None, [round_dd, round_note, round_table])

        # ---------------- Match predictor
        with gr.Tab("🔮 Match Predictor"):
            with gr.Row():
                home_dd = gr.Dropdown(TEAMS, value="Carlton", label="🏠 Home team")
                away_dd = gr.Dropdown(TEAMS, value="Collingwood", label="✈️ Away team")
                venue_dd = gr.Dropdown(engine.venues(), value=default_venue("Carlton"),
                                       label="🏟️ Ground", allow_custom_value=True)
                final_cb = gr.Checkbox(label="Finals game", value=False)
            with gr.Accordion("💰 Bookmaker odds (optional)", open=False):
                with gr.Row():
                    home_odds = gr.Number(label="Home team odds (decimal)", value=None, minimum=1)
                    away_odds = gr.Number(label="Away team odds (decimal)", value=None, minimum=1)
            go = gr.Button("Run prediction", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    verdict_md = gr.Markdown(elem_id="verdict")
                    why_md = gr.Markdown()
                with gr.Column(scale=2):
                    prob_label = gr.Label(label="Win chance", num_top_classes=2)
            value_md = gr.Markdown()

            home_dd.change(default_venue, home_dd, venue_dd)
            go.click(predict, [home_dd, away_dd, venue_dd, final_cb, home_odds, away_odds],
                     [verdict_md, prob_label, why_md, value_md])

        # ---------------- Ladder
        with gr.Tab("📈 ELO Ladder"):
            gr.Markdown("Team strength after the last game played. 1500 is average; +100 ELO "
                        "is roughly a 64% chance to beat that team at a neutral ground. "
                        "Ratings move part way back to 1500 at the start of each season.")
            ladder = gr.Dataframe(engine.elo_ladder(), interactive=False)

        # ---------------- Track record
        with gr.Tab("🎯 Track Record"):
            overall, by_year = backtest_frames()
            gr.Markdown(
                "### How good is it?\n"
                "Every season from 2015 was predicted by a model trained **only on earlier "
                "seasons** — the same position you are in before a game. "
                "Bookmaker odds were available to 2023."
            )
            gr.Dataframe(overall, interactive=False)
            if not by_year.empty:
                gr.LinePlot(by_year, x="Season", y="Tips correct %", color="Model",
                            title="Tips correct by season", y_lim=[55, 80], height=320)
            gr.Markdown("### Recent seasons (live data)")
            gr.Dataframe(engine.recent_season_scores(), interactive=False)
            gr.Markdown(
                "<span class='small-note'>**Tips correct**: share of winners picked. "
                "**Brier / log loss**: how good the percentages are, not just the pick "
                "(a 90% tip that loses is punished more than a 55% one). "
                "Typical AFL models tip 65–70%.</span>"
            )

        # ---------------- About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                "### How it works\n"
                "1. **ELO ratings**: every team gains or loses points after each game, more for "
                "bigger wins. Ratings move part of the way back to average between seasons.\n"
                "2. **Other factors**: interstate travel, games at the ground in the last 2 "
                "years, last-5-game form and scoring, recent head-to-head, days of rest, finals.\n"
                "3. **Margin model**: a linear model turns these into a predicted margin. "
                "The win chance comes from how spread out real margins are around the "
                "prediction (about ±36 points).\n"
                "4. **No peeking**: every feature uses only games played *before* that match.\n\n"
                "### Data\n"
                "- Live results and fixtures: [Squiggle API](https://api.squiggle.com.au)\n"
                "- Backup history and test odds: AFL Tables data via "
                "[akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) (MIT)\n\n"
                f"### Responsible gambling\n{RESPONSIBLE}"
            )

    gr.Markdown(f"<span class='small-note'>{RESPONSIBLE}</span>")


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS)
