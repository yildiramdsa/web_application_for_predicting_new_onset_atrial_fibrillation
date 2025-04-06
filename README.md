![Web Application for Predicting New-Onset Atrial Fibrillation Using Routinely Reported 12-Lead ECG Variables and Electronic Health Data](https://github.com/yildiramdsa/web_application_for_predicting_new_onset_atrial_fibrillation/blob/main/title_sm.png)

# Web Application for Predicting New-Onset Atrial Fibrillation Using Routinely Reported 12-Lead ECG Variables and Electronic Health Data

## Overview

This project delivers an interactive web application that uses routinely reported 12-lead ECG data and electronic health records to predict a patient's risk for new-onset atrial fibrillation (AF). Clinicians can enter patient data through an intuitive form and view personalized risk predictions and visual comparisons on a dashboard.

## Problem & Summary

Atrial fibrillation is a common, high-risk condition that greatly increases stroke risk. Traditional risk scores have had limited accuracy. Our application uses a Random Forest classifier—trained on real data—to provide personalized AF risk predictions and estimated time to onset. An interactive dashboard displays a PCA-based visualization, highlighting key predictors like age and ECG parameters to support clinical decision-making.

## Data Engineering Lifecycle

**Data Generation:** A synthetic dataset of ~100,000 patients (without prior AF) was generated from real patient profiles in the CIROC to mirror realistic clinical patterns.  
**Data Ingestion:** Patient data is collected via a Streamlit web form hosted on Amazon EC2, then securely transmitted for processing.  
**Data Transformation:** In EC2, data is imputed, standardized, and reduced using PCA for optimized predictive modeling.  
**Data Storage:** All raw, processed data and the trained model (serialized as a pickle file) are stored in Amazon S3. New patient records are archived as CSV files for future analysis and model retraining.  
**Data Serving:** The app retrieves processed data and the model from S3 to deliver real-time predictions through Streamlit. The dashboard compares patient profiles against a reference population using PCA and feature importance scores.  

## Infrastructure Justification

**Amazon EC2:** Hosts the web app for continuous, real-time data processing and prediction, avoiding the cold start delays of serverless options like Lambda.  
**Amazon S3:** Provides secure, scalable, and cost-effective storage for large volumes of data and model artifacts.  

## Key Files

**aws_app.py**: Runs on EC2 for real-time processing.  
**custom_transformers.py**: Contains custom data transformation classes.  
**model.pkl**: Serialized ML model trained on real data.  
**model_development.ipynb**: Notebook for EDA and model comparisons.  
**model_training.py**: Pipeline and model training script.  
**requirements.txt**: Lists dependencies for the Streamlit app.  
**streamlit_app.py**: Streamlit version (S3 saving disabled).  
**synthetic_data.csv**: Synthetic data for dashboard visualizations.  

**Live Application:** https://predicting-new-onset-af.streamlit.app/
