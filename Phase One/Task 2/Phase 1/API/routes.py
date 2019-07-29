from flask import Blueprint, request, jsonify
import pickle
import numpy as np

model = pickle.load(open("Count_vectorized_model", "rb"))
pre_processing = pickle.load(open("count_vectorizer", "rb"))
sentiment = ["negative", "positive", "neutral"]
model_predict = Blueprint("model_predictor", __name__)


@model_predict.route('/predict_sentiment', methods=["POST"])
def predict_value():
    review = [request.get_json()["review"]]
    #rating = request.get_json()["rating"]
    count_vectorized_data = pre_processing.transform(review)
    count_vectorized_data = np.asarray(count_vectorized_data.toarray())
    predictions = model.predict(count_vectorized_data)

    return jsonify(
        {
            "sentiment": sentiment[predictions[0]]
        }
    )
