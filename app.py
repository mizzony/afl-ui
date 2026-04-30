import gradio as gr
import requests

API_URL = "https://afl-api-168340537629.australia-southeast1.run.app/predict"

def predict(elo, form, score, rest, home):
    payload = {
        "elo_diff": elo,
        "form_diff": form,
        "score_diff": score,
        "rest_diff": rest,
        "home_ground": home
    }

    try:
        r = requests.post(API_URL, json=payload)
        result = r.json()
        return f"Prediction: {result['prediction']} | Prob: {result['probability']:.2f}"
    except Exception as e:
        return f"Error: {e}"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="ELO Diff"),
        gr.Number(label="Form Diff"),
        gr.Number(label="Score Diff"),
        gr.Number(label="Rest Diff"),
        gr.Radio([0,1], label="Home Ground")
    ],
    outputs="text",
    title="AFL Predictor",
    description="Powered by FastAPI on GCP"
).launch()