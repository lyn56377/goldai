import joblib
from types import SimpleNamespace

class SklearnClassifier:
    def __init__(self, model_path):
        self.pipeline = joblib.load(model_path)

    def run(self, query):
        prediction = self.pipeline.predict([query])[0]
        proba = max(self.pipeline.predict_proba([query])[0])

        result = {
            "documents": [
                SimpleNamespace(
                    meta={
                        "predicted_label": prediction,
                        "probability": round(proba, 4)
                    }
                )
            ]
        }

        return result, None

