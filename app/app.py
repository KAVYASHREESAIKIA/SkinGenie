from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Corrected model loading using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'skingenie_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'model', 'label_encoders.pkl')

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# Skincare tips based on skin type
skincare_tips = {
    'Dry': "Use a hydrating moisturizer and avoid hot water. Look for products with hyaluronic acid and glycerin.",
    'Oily': "Use oil-free and non-comedogenic products. Cleanse twice a day and use salicylic acid if needed.",
    'Combination': "Moisturize dry areas and use oil control on oily zones. Balance your skincare routine.",
    'Sensitive': "Use fragrance-free and hypoallergenic products. Avoid harsh scrubs and alcohol-based toners.",
    'Normal': "Maintain your skin with a gentle cleanser and daily SPF. Stay hydrated and eat balanced meals."
}

@app.route('/')
def home():
    return render_template('index.html', prediction=None, tip=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        features = ['Age', 'Gender', 'Hydration_Level', 'Oil_Level', 'Sensitivity', 'Humidity', 'Temperature']
        input_data = []

        for feature in features:
            value = request.form.get(feature)
            if value is None:
                return render_template('index.html', prediction="Error: Missing value.", tip=None)

            encoder = label_encoders.get(feature)
            if encoder:
                if value not in encoder.classes_:
                    return render_template('index.html', prediction=f"Error: Invalid value '{value}' for {feature}. Expected one of: {list(encoder.classes_)}", tip=None)
                value = encoder.transform([value])[0]
            else:
                try:
                    value = float(value)
                except ValueError:
                    return render_template('index.html', prediction=f"Error: Invalid number format for {feature}.", tip=None)

            input_data.append(value)

        input_df = pd.DataFrame([input_data], columns=features)
        prediction_encoded = model.predict(input_df)[0]

        # Decode the predicted skin type
        skin_type = label_encoders['Skin_Type'].inverse_transform([prediction_encoded])[0]
        tip = skincare_tips.get(skin_type, "No tips available for this skin type.")

        return render_template('index.html', prediction=f"Recommended Skin Type: {skin_type}", tip=tip)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", tip=None)

if __name__ == '__main__':
    app.run(debug=True)
