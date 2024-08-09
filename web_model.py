from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)

# Set directory
os.chdir('/Users/uthscbcm/Documents/Eastern/Project_Files')

# Set API endpoint for home:
@app.route('/')
def home():
    return render_template('index.html')

# Load models and pipeline
gbr_model_path = 'gbr_model.pkl'
gpr_model_path = 'gpr_model.pkl'
pipeline_path = 'preprocessing_pipeline.pkl'

try:
    with open(gbr_model_path, 'rb') as f:
        gbr_model = pickle.load(f)
    print("GBR model loaded successfully")
except Exception as e:
    print(f"Error loading GBR model: {e}")

try:
    with open(gpr_model_path, 'rb') as f:
        gpr_model = pickle.load(f)
    print("GPR model loaded successfully")
except Exception as e:
    print(f"Error loading GPR model: {e}")

try:
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    print("Pipeline loaded successfully")
except Exception as e:
    print(f"Error loading pipeline: {e}")

# Function to make predictions with confidence intervals
def confidence_int_preds(X):
    gbr_preds = gbr_model.predict(X)
    gpr_mean, gpr_std = gpr_model.predict(X, return_std=True)
    final_preds = gbr_preds + gpr_mean
    lower_bound = final_preds - 1.96 * gpr_std
    upper_bound = final_preds + 1.96 * gpr_std
    return final_preds, lower_bound, upper_bound

# Function to generate and save line plot with error bars
def generate_prediction_plot(prediction, lower_bound, upper_bound, plot_filename):
    fig, ax = plt.subplots()

    # Plot central prediction line with customized markers and line properties
    ax.errorbar(['Prediction'], [prediction], yerr=[[prediction - lower_bound], [upper_bound - prediction]],
                fmt='s', markersize=12, markerfacecolor='blue', markeredgecolor='black', markeredgewidth=2, color='blue', label='Prediction',
                capsize=5, linewidth=2)

    # Add reference lines
    ax.axhline(y=15, color='grey', linestyle='--', label='Lower Specification Limit')
    ax.axhline(y=16.3, color='orange', linestyle='--', label='Lower Control Limit')
    ax.axhline(y=18.3, color='orange', linestyle='--', label='Upper Control Limit')
    ax.axhline(y=19, color='grey', linestyle='--', label='Upper Specification Limit')

    ax.set_ylim(10, 20)
    ax.set_ylabel('Value')
    ax.legend()

    # Save the plot to a file
    plt.savefig(plot_filename, format='png')
    plt.close()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['Weight'])
        nv = float(request.form['NV'])
        amine = float(request.form['Amine'])
        max_temp = float(request.form['Max_Temp_Degrees'])
        rain_inches = float(request.form['Rain_Inches'])

        # Fixed batch weight
        batch_weight = 4100

        # Calculate BPA_level
        bpa_level = batch_weight * amine

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[weight, nv, amine, bpa_level, max_temp, rain_inches]],
                                  columns=['Weight', 'NV', 'Amine', 'BPA_level', 'Max_Temp_Degrees', 'Rain_Inches'])

        # Apply the same preprocessing and polynomial transformations used during training
        preprocessed_data = pipeline.transform(input_data)

        # Make predictions with confidence intervals
        predictions, lower_bounds, upper_bounds = confidence_int_preds(preprocessed_data)

        # Round the results to 4 decimal places
        prediction = round(predictions[0], 4)
        lower_bound = round(lower_bounds[0], 4)
        upper_bound = round(upper_bounds[0], 4)

        # Generate and save the plot
        plot_filename = os.path.join(app.root_path, 'static', 'prediction_plot.png')
        generate_prediction_plot(prediction, lower_bound, upper_bound, plot_filename)

        return render_template('predict.html', result={
            'prediction': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'plot_url': '/static/prediction_plot.png'
        })

    return render_template('predict.html')

# Endpoint to render bio page
@app.route('/bio')
def bio():
    return render_template('bio.html')

# Endpoint to render resume page
@app.route('/resume')
def resume():
    return render_template('resume.html')

# Endpoint to render projects page
@app.route('/projects')
def projects():
    return render_template('projects.html')

if __name__ == '__main__':
    app.run(debug=True)
