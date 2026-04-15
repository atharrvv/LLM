import os
import pickle

# Load model once (global scope)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'Model_Pickle',
    'model.pkl'   # change if your name is different
)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def preprocess(data: dict):
    """
    Modify this based on your model input
    """
    try:
        features = [
            data.get("feature1"),
            data.get("feature2"),
            # add all required features here
        ]
        return [features]
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")


def predict(data: dict):
    processed = preprocess(data)
    result = model.predict(processed)
    return result.tolist()
