from flask import Blueprint, render_template, request, jsonify
import pickle
import pandas as pd

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predict_delay(data)
    return jsonify(prediction)


def predict_delay(data):
    # Load model and label encoders
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Create DataFrame from input data
    df = pd.DataFrame([data])

    # Apply label encoding to categorical columns
    categorical_columns = ['OP_UNIQUE_CARRIER', 'TAIL_NUM', 'DEST', 'Wind', 'Condition']
    for col in categorical_columns:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # Make prediction
    prediction = model.predict(df)
    return {'delay': prediction[0]}
