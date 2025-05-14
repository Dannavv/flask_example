from app import app
from flask import render_template, request
from utils.pipeline_utils import recommend_nearby_apartments, load_split_pickle, load_pipeline, make_prediction
import pandas as pd

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load the pipeline and necessary data
        merged_file = load_split_pickle('pickles')
        pipeline = load_pipeline(merged_file)

        # Extract form data and create a DataFrame
        input_data = pd.DataFrame({
            'property_type': [request.form['property_type']],
            'sector': [request.form['sector']],
            'bedRoom': [float(request.form['bedRoom'])],
            'bathroom': [float(request.form['bathroom'])],
            'balcony': [request.form['balcony']],
            'agePossession': [request.form['agePossession']],
            'built_up_area': [float(request.form['built_up_area'])],
            'servant room': [float(request.form['servant_room'])],
            'store room': [float(request.form['store_room'])],
            'furnishing_type': [request.form['furnishing_type']],
            'luxury_category': [request.form['luxury_category']],
            'floor_category': [request.form['floor_category']],
        })

        print(input_data.head())

        try:
            # Make prediction using the pipeline
            lower, upper = make_prediction(pipeline, input_data)
            # Render the result on a template (you can create a new one or reuse)
            if lower and upper:
                print(f"🏷️ Estimated Price Range: ₹ {lower} Cr – ₹ {upper} Cr")
                return render_template('predict_test.html', lower=lower, upper=upper)
            return render_template('predict_test.html', lower="loading", upper="loading")
        except Exception as e:
            error_message = str(e)
            return render_template('predict_test.html', lower=error_message)

    # If GET request, render the form
    return render_template('predict.html')

@app.route('/predict_test')
def predict_test():
    merged_file = load_split_pickle('pickles')
    pipeline = load_pipeline(merged_file)

    input_data = pd.DataFrame({
        'property_type': ['flat'],
        'sector': ['sector 36'],
        'bedRoom': [3.0],
        'bathroom': [2.0],
        'balcony': ['2'],
        'agePossession': ['New Property'],
        'built_up_area': [850.0],
        'servant room': [0.0],
        'store room': [0.0],
        'furnishing_type': ['unfurnished'],
        'luxury_category': ['Low'],
        'floor_category': ['Low Floor'],
    })

    try:
        lower, upper = make_prediction(pipeline, input_data)
        # print(f"🏷️ Estimated Price Range: ₹ {lower} Cr – ₹ {upper} Cr")
    except Exception as e:
        print("❌ Error:", e)
    
    return render_template('predict_test.html', lower=lower, upper=upper)


@app.route('/recommend')
def recommendation():
    recommend = recommend_nearby_apartments('Bajghera Road', radius_km=5.0)
    return render_template('recommend.html', recommend=recommend)