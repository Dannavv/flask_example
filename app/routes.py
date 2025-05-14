from app import app
from flask import render_template
from utils.pipeline_utils import recommend_nearby_apartments, load_split_pickle, load_pipeline, make_prediction
import pandas as pd

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
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
    
    return render_template('predict.html', lower=lower, upper=upper)


@app.route('/recommend')
def recommendation():
    recommend = recommend_nearby_apartments('Bajghera Road', radius_km=5.0)
    return render_template('recommend.html', recommend=recommend)