import gradio as gr
import requests

API_URL = "https://afl-api-168340537629.australia-southeast1.run.app/predict_match"

#API_URL = os.getenv("API_URL")
teams = [
    "Collingwood", "Carlton", "Richmond", "Melbourne",
    "Hawthorn", "Geelong", "Sydney", "Brisbane Lions",
    "Adelaide", "Port Adelaide", "West Coast", "Fremantle",
    "Essendon", "St Kilda", "Western Bulldogs",
    "North Melbourne", "Gold Coast", "GWS"
]

def predict(home_team, away_team):
    if home_team == away_team:
        return "⚠️ Please select two different teams", "", 0

    payload = {
        "home_team": home_team,
        "away_team": away_team
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            return f"❌ API Error: {response.status_code}", "", 0

        result = response.json()

        winner = result["winner"]
        confidence = result["confidence"]
        reasons = result.get("reasons", [])

        # 🎯 main result (clean + strong)
        result_text = f"""
## 🏆 Prediction Result
### **{winner} expected to win**
**Confidence:** {confidence*100:.1f}%
"""

        # 🧠 explanation
        if reasons:
            reasons_text = "### 📊 Model Insights\n"
            for r in reasons:
                reasons_text += f"- {r}\n"
        else:
            reasons_text = "No additional insights available."

        return result_text, reasons_text, confidence

    except Exception as e:
        return f"❌ Error: {str(e)}", "", 0


with gr.Blocks(theme=gr.themes.Soft()) as app:

    # 🔥 HERO SECTION
    gr.Markdown("""
# 📊 Betting Intelligence Platform
### Find the edge. Bet with probability, not emotion.
This tool uses ELO rating and recent performance trends to estimate match outcomes  
and support smarter, data-driven betting decisions.
""")

    # 🎯 INPUT SECTION (CARD STYLE)
    with gr.Group():
        gr.Markdown("### 🔍 Select Match")

        with gr.Row():
            home_team = gr.Dropdown(teams, label="🏠 Home Team")
            away_team = gr.Dropdown(teams, label="✈️ Away Team")

        predict_btn = gr.Button("Run Prediction", variant="primary")

    # 📈 OUTPUT SECTION
    with gr.Group():
        result_output = gr.Markdown()
        confidence_bar = gr.Slider(
            minimum=0, maximum=1, step=0.01,
            label="Model Confidence",
            interactive=False
        )

    # 🧠 INSIGHTS SECTION
    with gr.Group():
        reason_output = gr.Markdown()

    # 🔗 ACTION
    predict_btn.click(
        fn=predict,
        inputs=[home_team, away_team],
        outputs=[result_output, reason_output, confidence_bar]
    )

app.launch()
