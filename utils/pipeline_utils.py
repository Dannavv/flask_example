import os
import pickle
import pandas as pd
import numpy as np

def load_split_pickle(split_dir, output_path='pipeline_merged.pkl'):
    part_files = sorted(
        [f for f in os.listdir(split_dir) if f.startswith('pipeline_part')],
        key=lambda x: int(x.split('part')[1].split('.')[0])
    )
    with open(output_path, 'wb') as merged:
        for part in part_files:
            with open(os.path.join(split_dir, part), 'rb') as pf:
                merged.write(pf.read())
    return output_path

def load_pipeline(pickle_path):
    with open(pickle_path, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

def make_prediction(pipeline, input_data):
    prediction = pipeline.predict(input_data)
    base_price = np.expm1(prediction[0])
    lower_bound = round(base_price * 0.78, 2)
    upper_bound = round(base_price * 1.22, 2)
    return lower_bound, upper_bound


# recommendation_utils




def load_file(filename):
    """Load a pickle file safely."""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found.")
    except pickle.UnpicklingError:
        print(f"⚠️ Unpickling error: '{filename}' may be corrupted.")
    except Exception as e:
        print(f"🚨 Error loading '{filename}': {str(e)}")
    return None


def load_csv(filename):
    """Load a CSV file safely."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ CSV file '{filename}' not found.")
    except Exception as e:
        print(f"🚨 Error loading CSV: {str(e)}")
    return None

def recommend_properties_with_scores(property_name, location_df, cosine_sim1, cosine_sim2, cosine_sim3, top_n=5):
    """Generate top-N recommended properties with similarity scores."""
    try:
        # Check if required data files are loaded
        if location_df is None or cosine_sim1 is None or cosine_sim2 is None or cosine_sim3 is None:
            print("❌ Required data files are missing.")
            return pd.DataFrame()

        # Combine similarity matrices with weights
        cosine_sim_matrix = 3 * cosine_sim1 + 5 * cosine_sim2 + 6 * cosine_sim3

        # Get similarity scores for the selected property
        sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

        # Sort the similarity scores in descending order
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N properties and their similarity scores
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()

        # Return the results as a DataFrame
        return pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })

    except Exception as e:
        print(f"🚨 Error generating recommendations: {str(e)}")
        return pd.DataFrame()

def filter_properties_within_radius(location_df, selected_location, radius_km):
    """Filter properties within a certain radius of a location."""
    try:
        # Check if required data is present
        if location_df is None or selected_location not in location_df.columns:
            print(f"❌ Location data is missing or invalid for '{selected_location}'.")
            return []

        # Filter properties within the radius
        filtered_locations = location_df[location_df[selected_location] < (radius_km * 1000)][selected_location].sort_values()

        # Check if any properties were found
        if filtered_locations.empty:
            print("⚠️ No properties found within the given radius.")
            return []

        # Print filtered properties and their distances
        print(f"Properties within {radius_km} km from {selected_location}:")
        for prop, dist in filtered_locations.items():
            print(f"{prop}: {round(dist / 1000, 2)} km")

        # Return the list of properties within the radius
        return filtered_locations.index.to_list()

    except Exception as e:
        print(f"🚨 Error filtering properties: {str(e)}")
        return []


def recommend_nearby_apartments(
    selected_location,
    radius_km=5.0,
    cosine_sim1_path="pickles/cosine_sim1.pkl",
    cosine_sim2_path="pickles/cosine_sim2.pkl",
    cosine_sim3_path="pickles/cosine_sim3.pkl",
    location_df_path="pickles/location_distance.pkl",
    data_csv_path="data/data_viz1.csv",
    top_n=5
):
    """
    Loads data, filters apartments within a radius, and recommends similar properties.
    
    Args:
        selected_location (str): The location to search around.
        radius_km (float): Radius in kilometers.
        cosine_sim1_path (str): Path to cosine_sim1 pickle.
        cosine_sim2_path (str): Path to cosine_sim2 pickle.
        cosine_sim3_path (str): Path to cosine_sim3 pickle.
        location_df_path (str): Path to location distance pickle.
        data_csv_path (str): Path to CSV data.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: DataFrame of recommended properties, or None if not found.
    """
    # Load data files
    cosine_sim1 = load_file(cosine_sim1_path)
    cosine_sim2 = load_file(cosine_sim2_path)
    cosine_sim3 = load_file(cosine_sim3_path)
    location_df = load_file(location_df_path)
    df1 = load_csv(data_csv_path)

    # Show available locations
    if location_df is not None:
        print("Available locations:")
        print(location_df.columns.to_list())
    else:
        print("❌ Location data not loaded.")
        return None

    # Filter apartments within the specified radius
    filtered_apartments = filter_properties_within_radius(location_df, selected_location, radius_km)

    if filtered_apartments:
        selected_apartment = filtered_apartments[0]
        recommendations = recommend_properties_with_scores(
            selected_apartment,
            location_df,
            cosine_sim1,
            cosine_sim2,
            cosine_sim3,
            top_n=top_n
        )

        if not recommendations.empty:
            print("📌 Recommended Apartments:")
            print(recommendations)
            return recommendations
        else:
            print("⚠️ No recommendations found.")
            return None
    else:
        print("⚠️ No apartments found to recommend from.")
        return None
