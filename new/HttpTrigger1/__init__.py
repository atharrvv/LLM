import azure.functions as func
import json
from Prediction.model import predict

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()

        result = predict(data)

        return func.HttpResponse(
            json.dumps({"prediction": result}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
