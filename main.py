from flask import Flask, request, render_template, render_template_string
import numpy as np
import joblib

# Load all saved components
model = joblib.load('model.pkl')
brand_encoder = joblib.load('brand_encoder.pkl')
item_encoder = joblib.load('item_encoder.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = float(request.form['age'])
        brand = request.form['brand']
        item = request.form['item']

        # Encode brand and item
        brand_encoded = brand_encoder.transform([brand])[0]
        item_encoded = item_encoder.transform([item])[0]

        # Create feature vector
        features = np.array([[weight, height, age, brand_encoded, item_encoded]])

        # Scale numerical features (assumes first 3 columns are numerical)
        scaled_features = scaler.transform(features[:, :3])
        combined_features = np.hstack((scaled_features, features[:, 3:]))

        # Select top 2 features
        selected_features = selector.transform(combined_features)

        # Make prediction
        prediction = model.predict(selected_features)[0]

        # Map numeric prediction back to size label
        size_labels = {0: 'XS', 1: 'S', 2: 'M', 3: 'L', 4: 'XL', 5: 'XXL', 6: 'XXXL'}
        predicted_size = size_labels[prediction]

        # Render the prediction result using the separate template
        result_html = render_template('prediction_result.html', 
                                    predicted_size=predicted_size,
                                    weight=weight,
                                    height=height,
                                    age=age,
                                    brand=brand,
                                    item=item)

        return render_template('index.html', prediction_html=result_html)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

app.run(debug=True)
