import gradio as gr
import requests

API_URL = "https://afl-api-168340537629.australia-southeast1.run.app/predict_match"

teams = [
    "Collingwood", "Carlton", "Richmond", "Melbourne",
    "Hawthorn", "Geelong", "Sydney", "Brisbane Lions"
]

def predict(home_team, away_team):

    payload = {
        "home_team": home_team,
        "away_team": away_team
    }

    try:
        r = requests.post(API_URL, json=payload)
        result = r.json()

        winner = home_team if result["prediction"] == 1 else away_team
        prob = result["probability"] * 100

        return f"🏆 {winner} likely to WIN ({prob:.1f}%)"

    except Exception as e:
        return f"Error: {e}"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(teams, label="Home Team"),
        gr.Dropdown(teams, label="Away Team"),
    ],
    outputs="text",
    title="🏉 AFL Match Outcome Predictor",
    description="Predict AFL match results using ML model"
).launch()
