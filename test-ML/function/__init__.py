import logging
import pickle
import os
import json
import numpy as np
import azure.functions as func

# Load model once at cold start (not inside handler)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    _artifact = pickle.load(f)

_model        = _artifact["model"]
_target_names = _artifact["target_names"]
_feature_names = _artifact["feature_names"]

# ── HTML helpers ────────────────────────────────────────────────────────────

_HTML_FORM = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Iris Flower Predictor</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',sans-serif;background:#f0f4f8;min-height:100vh;display:flex;align-items:center;justify-content:center}}
    .card{{background:#fff;border-radius:16px;box-shadow:0 4px 24px rgba(0,0,0,.10);padding:40px 48px;max-width:480px;width:100%}}
    h1{{font-size:1.6rem;color:#1a202c;margin-bottom:4px}}
    .subtitle{{color:#718096;font-size:.9rem;margin-bottom:28px}}
    label{{display:block;font-size:.85rem;font-weight:600;color:#4a5568;margin-bottom:6px;margin-top:18px}}
    input[type=number]{{width:100%;padding:10px 14px;border:1.5px solid #e2e8f0;border-radius:8px;font-size:1rem;color:#2d3748;transition:border .2s}}
    input[type=number]:focus{{outline:none;border-color:#667eea}}
    button{{margin-top:28px;width:100%;padding:13px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;transition:opacity .2s}}
    button:hover{{opacity:.9}}
    .tip{{font-size:.78rem;color:#a0aec0;margin-top:20px;text-align:center}}
    {result_style}
  </style>
</head>
<body>
<div class="card">
  <h1>🌸 Iris Predictor</h1>
  <p class="subtitle">Enter measurements to classify the iris flower species</p>
  {result_html}
  <form method="GET" action="/api/predict_iris">
    <label>Sepal Length (cm)</label>
    <input type="number" name="sepal_length" step="0.1" min="0" max="20" placeholder="e.g. 5.1" value="{sl}" required>
    <label>Sepal Width (cm)</label>
    <input type="number" name="sepal_width" step="0.1" min="0" max="20" placeholder="e.g. 3.5" value="{sw}" required>
    <label>Petal Length (cm)</label>
    <input type="number" name="petal_length" step="0.1" min="0" max="20" placeholder="e.g. 1.4" value="{pl}" required>
    <label>Petal Width (cm)</label>
    <input type="number" name="petal_width" step="0.1" min="0" max="20" placeholder="e.g. 0.2" value="{pw}" required>
    <button type="submit">Predict Species →</button>
  </form>
  <p class="tip">Try: 5.1, 3.5, 1.4, 0.2 → setosa &nbsp;|&nbsp; 6.3, 3.3, 6.0, 2.5 → virginica</p>
</div>
</body>
</html>"""

_SPECIES_META = {
    "setosa":     {"emoji": "🌼", "color": "#38a169", "bg": "#f0fff4"},
    "versicolor": {"emoji": "🌺", "color": "#d69e2e", "bg": "#fffff0"},
    "virginica":  {"emoji": "🪷", "color": "#805ad5", "bg": "#faf5ff"},
}

def _render_result(species: str, proba: list[float]) -> tuple[str, str]:
    meta = _SPECIES_META.get(species, {"emoji": "🌸", "color": "#3182ce", "bg": "#ebf8ff"})
    bars = "".join(
        f"<div class='bar-row'><span>{n}</span>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{p*100:.1f}%;background:{_SPECIES_META.get(n,{}).get('color','#3182ce')}'></div></div>"
        f"<span class='pct'>{p*100:.1f}%</span></div>"
        for n, p in zip(_target_names, proba)
    )
    result_html = f"""
    <div class='result' style='background:{meta["bg"]};border-left:4px solid {meta["color"]}'>
      <div class='result-header'>Prediction</div>
      <div class='result-species'>{meta["emoji"]} <em>Iris {species}</em></div>
      <div class='prob-label'>Confidence breakdown</div>
      {bars}
    </div>"""
    result_style = f"""
    .result{{border-radius:10px;padding:18px 20px;margin-bottom:22px}}
    .result-header{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#718096}}
    .result-species{{font-size:1.4rem;font-weight:700;color:{meta["color"]};margin:4px 0 14px}}
    .prob-label{{font-size:.78rem;font-weight:600;color:#718096;margin-bottom:8px}}
    .bar-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.82rem;color:#4a5568}}
    .bar-row span:first-child{{width:80px;text-align:right}}
    .bar-track{{flex:1;height:10px;background:#e2e8f0;border-radius:99px;overflow:hidden}}
    .bar-fill{{height:100%;border-radius:99px;transition:width .4s}}
    .pct{{width:44px;font-weight:600}}"""
    return result_html, result_style


# ── Main handler ─────────────────────────────────────────────────────────────

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("predict_iris triggered")

    # Support both GET (query params) and POST (JSON body)
    params: dict = {}
    if req.method == "POST":
        try:
            params = req.get_json()
        except ValueError:
            pass
    else:
        params = {k: req.params.get(k) for k in
                  ["sepal_length", "sepal_width", "petal_length", "petal_width"]}

    keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    accept_json = "application/json" in req.headers.get("Accept", "")

    # No inputs → show blank form
    if not any(params.get(k) for k in keys):
        html = _HTML_FORM.format(result_html="", result_style="",
                                 sl="", sw="", pl="", pw="")
        return func.HttpResponse(html, mimetype="text/html", status_code=200)

    # Validate inputs
    try:
        features = [float(params[k]) for k in keys]
    except (TypeError, ValueError, KeyError) as e:
        msg = f"Invalid input: {e}. Provide: sepal_length, sepal_width, petal_length, petal_width"
        if accept_json:
            return func.HttpResponse(json.dumps({"error": msg}),
                                     mimetype="application/json", status_code=400)
        return func.HttpResponse(msg, status_code=400)

    # Predict
    X = np.array([features])
    pred_idx = int(_model.predict(X)[0])
    species  = _target_names[pred_idx]
    proba    = _model.predict_proba(X)[0].tolist()

    # JSON response (for API clients)
    if accept_json:
        payload = {
            "prediction": species,
            "probabilities": dict(zip(_target_names, proba)),
            "features": dict(zip(keys, features)),
        }
        return func.HttpResponse(json.dumps(payload, indent=2),
                                 mimetype="application/json", status_code=200)

    # HTML response (for browser)
    result_html, result_style = _render_result(species, proba)
    sl, sw, pl, pw = (str(params.get(k, "")) for k in keys)
    html = _HTML_FORM.format(result_html=result_html, result_style=result_style,
                             sl=sl, sw=sw, pl=pl, pw=pw)
    return func.HttpResponse(html, mimetype="text/html", status_code=200)
