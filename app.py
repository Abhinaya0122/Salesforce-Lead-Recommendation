from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load models and encoders
pipeline = joblib.load('pipeline_model.joblib')
service_label_encoder = joblib.load('service_recommended_label_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)

    # Manually create the DataFrame based on expected columns
    # Ensure to extract all required fields and handle them correctly
    try:
        location = data.get('Location')
        number_of_employees = data.get('NumberOfEmployees')
        industry = data.get('Industry')

        # Construct the DataFrame with the necessary columns
        input_data = {
            'Location': [location],
            'Industry': [industry],
            # Add other fields if necessary, ensure they are properly populated
            'Company_Size': [number_of_employees]  # Adjust as necessary if your model uses this
        }

        input_df = pd.DataFrame(input_data)
        print(input_df)

        # Predict using the pipeline
        prediction = pipeline.predict(input_df)
        predicted_service = service_label_encoder.inverse_transform(prediction)

        return jsonify({"predicted_service": predicted_service.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
