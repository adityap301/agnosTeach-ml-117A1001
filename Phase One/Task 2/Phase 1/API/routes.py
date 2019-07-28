from flask import Blueprint, request, jsonify
import pickle
import numpy as np

model = pickle.load(open("Count_vectorized_model", "rb"))
pre_processing = pickle.load(open("count_vectorizer", "rb"))
sentiment = ["negative", "positive", "neutral"]
model_predict = Blueprint("model_predictor", __name__)


@model_predict.route('/predict_sentiment', methods=["POST"])
def predict_value():
    review = request.get_json()["review"]
    #rating = request.get_json()["rating"]
    clean_review = pre_processing.transform(np.asarray([review]))
    predictions = model.predict([[clean_review]])

    return jsonify(
        {
            "sentiment": sentiment[predictions[0]]
        }
    )
