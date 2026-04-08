import azure.functions as func
import pickle
import json
import os

app = func.FunctionApp()

@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:

    try:
        # Load pickle file
        pickle_path = os.path.join(os.path.dirname(__file__), "Pickle file", "model.pkl")
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)

        model = bundle['model']
        vectorizer = bundle['vectorizer']

        # Get input text from request
        req_body = req.get_json()
        input_text = req_body.get('text')

        if not input_text:
            return func.HttpResponse(
                json.dumps({"error": "Please provide 'text' in request body"}),
                mimetype="application/json",
                status_code=400
            )

        # Vectorize and predict
        X = vectorizer.transform([input_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        result = {
            "input_text": input_text,
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": round(float(max(probability)), 2)
        }

        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
