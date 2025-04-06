import os
import boto3
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from custom_transformers import PreprocessDataTransformer

def download_dataset_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)

def upload_file_to_s3(local_file, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, key)

DATA_BUCKET = "prediction-of-atrial-fibrillation"
DATASET_KEY = "synthetic_data_stats_competition_2025.xlsx"
LOCAL_DATASET_PATH = "synthetic_data_stats_competition_2025.xlsx"

if not os.path.exists(LOCAL_DATASET_PATH):
    download_dataset_from_s3(DATA_BUCKET, DATASET_KEY, LOCAL_DATASET_PATH)

data = pd.read_excel(LOCAL_DATASET_PATH)

leakage_cols = [
    "time_to_outcome_afib_aflutter_new_post",
    "outcome_all_cause_death",
    "time_to_outcome_all_cause_death",
    "follow_up_duration",
    "ecg_resting_afib",
    "ecg_resting_aflutter"
]
data_processed = data.drop(columns=leakage_cols)

y = data_processed["outcome_afib_aflutter_new_post"]
X = data_processed.drop(columns=["outcome_afib_aflutter_new_post"])

def drop_missing_ECG(df):
    missing_idx = df.index[
        (df['ecg_resting_hr'].isnull()) &
        (df['ecg_resting_pr'].isnull()) &
        (df['ecg_resting_qrs'].isnull()) &
        (df['ecg_resting_qtc'].isnull())
    ]
    return df.drop(missing_idx)
X = drop_missing_ECG(X)
y = y.loc[X.index]

normal_val = {
    'hgb_peri': 145,
    'hct_peri': 0.42,
    'rdw_peri': 13.5,
    'wbc_peri': 7.0,
    'plt_peri': 250,
    'inr_peri': 1.0,
    'ptt_peri': 30,
    'esr_peri': 15,
    'crp_high_sensitive_peri': 1.0,
    'albumin_peri': 40,
    'alkaline_phophatase_peri': 90,
    'alanine_transaminase_peri': 25,
    'aspartate_transaminase_peri': 35,
    'bilirubin_total_peri': 7,
    'bilirubin_direct_peri': 5,
    'urea_peri': 5,
    'creatinine_peri': 84,
    'urine_alb_cr_ratio_peri': 15,
    'sodium_peri': 140,
    'potassium_peri': 4.0,
    'chloride_peri': 105,
    'ck_peri': 150,
    'troponin_t_hs_peri_highest': 44,
    'NTproBNP_peri_highest': 250,
    'glucose_fasting_peri_highest': 5.5,
    'glucose_random_peri_highest': 6.9,
    'hga1c_peri_highest': 5.6,
    'tchol_peri_highest': 5,
    'ldl_peri_highest': 2.6,
    'hdl_peri_lowest': 1.4,
    'tg_peri_highest': 1.7,
    'iron_peri': 15,  
    'tibc_peri': 58, 
    'ferritin_peri': 100,   
    'tsh_peri': 2.0
}
X = X.fillna(value=normal_val)

pipeline = Pipeline(steps=[
    ('preprocess_data', PreprocessDataTransformer()),
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('rf', RandomForestClassifier(random_state=26, class_weight='balanced', n_estimators=500))
])
pipeline.fit(X, y)
joblib.dump(pipeline, 'model.pkl')

MODEL_UPLOAD_KEY = "model.pkl"
upload_file_to_s3('model.pkl', DATA_BUCKET, MODEL_UPLOAD_KEY)

print("Model training complete and model uploaded to S3.")