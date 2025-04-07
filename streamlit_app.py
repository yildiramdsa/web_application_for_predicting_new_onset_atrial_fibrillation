import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from custom_transformers import PreprocessDataTransformer

st.set_page_config(page_title="AFib Risk Prediction", layout="wide")

LOCAL_MODEL_PATH = "model.pkl"
LOCAL_DATA_PATH = "synthetic_data.csv"

model = joblib.load(LOCAL_MODEL_PATH)

@st.cache_data(show_spinner=True)
def load_dataset():
    return pd.read_csv(LOCAL_DATA_PATH)

data = load_dataset()

def create_pca_for_plotting(df, input_keys):
    d = df.copy()
    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [col for col in numeric_cols if col in input_keys]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(d[common_cols].fillna(0))
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    d["PC1"] = x_pca[:, 0]
    d["PC2"] = x_pca[:, 1]
    return d, scaler, pca, common_cols

def transform_new_input_for_plotting(new_input, common_cols, scaler, pca):
    d = pd.DataFrame([new_input]).copy()
    x_scaled = scaler.transform(d[common_cols].fillna(0))
    x_pca = pca.transform(x_scaled)
    return x_pca[0, 0], x_pca[0, 1]

def plot_pca_with_af_colors(df_plot, x_new, y_new):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_yes = df_plot[df_plot["outcome_afib_aflutter_new_post"] == 1]
    df_no  = df_plot[df_plot["outcome_afib_aflutter_new_post"] == 0]
    ax.scatter(df_yes["PC1"], df_yes["PC2"], c="#db6459", alpha=0.25, label="AF: Yes")
    ax.scatter(df_no["PC1"],  df_no["PC2"],  c="#989898", alpha=0.25, label="AF: No")
    ax.scatter(x_new, y_new, c="#db6459", marker=".", s=750, label="New Patient")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Plot, Highlighting New Patient")
    ax.legend()
    st.pyplot(fig)

def make_prediction(form_values):
    input_data = pd.DataFrame([form_values])
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(input_data)[:, 1][0]
    else:
        risk_score = model.predict(input_data)[0]
    if risk_score <= 0.33:
        prediction = "🟢 Low Risk"
    elif risk_score <= 0.66:
        prediction = "🟡 Medium Risk"
    else:
        prediction = "🔴 High Risk"
    estimated_life_years = (1 - risk_score) * 10
    return prediction, risk_score, estimated_life_years

def display_results(pid, prediction, risk_score, estimated_life_years):
    st.subheader(f"Prediction Summary for Patient: `{pid}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("AFib Risk Level:", prediction)
    c2.metric("Risk Probability:", f"{risk_score:.2f}")
    c3.metric("Expected AFib-Free Years", f"{estimated_life_years:.1f} yrs")

def plot_distribution_with_afib_hue(df, form_values, feature_name, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    custom_palette = {0: "#989898", 1: "#db6459"}
    sns.histplot(
        data=df,
        x=feature_name,
        hue="outcome_afib_aflutter_new_post",
        palette=custom_palette,
        bins=50,
        ax=ax,
        kde=False,
        multiple="stack",
        alpha=0.6
    )
    ax.axvline(
        form_values[feature_name],
        color="#db6459",
        linestyle="--",
        linewidth=2,
        label=f"Patient Value: {form_values[feature_name]}"
    )
    afib_absent = mpatches.Patch(color=custom_palette[0], label="AFib Absent")
    afib_present = mpatches.Patch(color=custom_palette[1], label="AFib Present")
    patient_line = mlines.Line2D([], [], color='#db6459', linestyle='--', linewidth=2, label=f"Patient Value: {form_values[feature_name]}")
    ax.legend(handles=[afib_absent, afib_present, patient_line])
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

default_values = {
    "patient_id": None,
    "demographics_age_index_ecg": None,
    "demographics_birth_sex": None,
    "hypertension_icd10": None,
    "diabetes_combined": None,
    "dyslipidemia_combined": None,
    "dcm_icd10": 0,
    "hcm_icd10": 0,
    "myocarditis_icd10_prior": 0,
    "pericarditis_icd10_prior": 0,
    "aortic_aneurysm_icd10": 0,
    "aortic_dissection_icd10_prior": 0,
    "pulmonary_htn_icd10": 0,
    "amyloid_icd10": 0,
    "copd_icd10": 0,
    "obstructive _sleep_apnea_icd10": 0,
    "hyperthyroid_icd10": 0,
    "hypothyroid_icd10": 0,
    "rheumatoid_arthritis_icd10": 0,
    "sle_icd10": 0,
    "sarcoid_icd10": 0,
    "cancer_any_icd10": 0,
    "event_cv_hf_admission_icd10_prior": 0,
    "event_cv_cad_acs_acute_mi_icd10_prior": 0,
    "event_cv_cad_acs_unstable_angina_icd10_prior": 0,
    "event_cv_cad_acs_other_icd10_prior": 0,
    "event_cv_ep_vt_any_icd10_prior": 0,
    "event_cv_ep_sca_survived_icd10_cci_prior": 0,
    "event_cv_cns_stroke_ischemic_icd10_prior": 0,
    "event_cv_cns_stroke_hemorrh_icd10_prior": 0,
    "event_cv_cns_tia_icd10_prior": 0,
    "pci_prior": 0,
    "cabg_prior": 0,
    "transplant_heart_cci_prior": 0,
    "lvad_cci_prior": 0,
    "pacemaker_permanent_cci_prior": 0,
    "crt_cci_prior": 0,
    "icd_cci_prior": 0,
    "ecg_resting_hr": None,
    "ecg_resting_pr": None,
    "ecg_resting_qrs": None,
    "ecg_resting_qtc": None,
    "ecg_resting_paced": 0,
    "ecg_resting_bigeminy": 0,
    "ecg_resting_LBBB": 0,
    "ecg_resting_RBBB": 0,
    "ecg_resting_incomplete_LBBB": 0,
    "ecg_resting_incomplete_RBBB": 0,
    "ecg_resting_LAFB": 0,
    "ecg_resting_LPFB": 0,
    "ecg_resting_bifascicular_block": 0,
    "ecg_resting_trifascicular_block": 0,
    "ecg_resting_intraventricular_conduction_delay": 0,
    "hgb_peri": 145,
    "hct_peri": 0.42,
    "rdw_peri": 13.5,
    "wbc_peri": 7.0,
    "plt_peri": 250,
    "inr_peri": 1.0,
    "ptt_peri": 30,
    "esr_peri": 15,
    "crp_high_sensitive_peri": 1.0,
    "albumin_peri": 40,
    "alkaline_phophatase_peri": 90,
    "alanine_transaminase_peri": 25,
    "aspartate_transaminase_peri": 35,
    "bilirubin_total_peri": 7,
    "bilirubin_direct_peri": 5,
    "urea_peri": 5,
    "creatinine_peri": 84,
    "urine_alb_cr_ratio_peri": 15,
    "sodium_peri": 140,
    "potassium_peri": 4.0,
    "chloride_peri": 105,
    "ck_peri": 150,
    "troponin_t_hs_peri_highest": 44,
    "NTproBNP_peri_highest": 250,
    "glucose_fasting_peri_highest": 5.5,
    "glucose_random_peri_highest": 6.9,
    "hga1c_peri_highest": 5.6,
    "tchol_peri_highest": 5,
    "ldl_peri_highest": 2.6,
    "hdl_peri_lowest": 1.4,
    "tg_peri_highest": 1.7,
    "iron_peri": 15,
    "tibc_peri": 58,
    "ferritin_peri": 100,
    "tsh_peri": 2.0,
    "anti_platelet_oral_non_asa_any_peri": 0,
    "anti_coagulant_oral_any_peri": 0,
    "nitrates_any_peri": 0,
    "ranolazine_peri": 0,
    "acei_peri": 0,
    "arb_peri": 0,
    "arni_entresto_peri": 0,
    "beta_blocker_any_peri": 0,
    "ivabradine_peri": 0,
    "ccb_dihydro_peri": 0,
    "ccb_non_dihydro_peri": 0,
    "diuretic_loop_peri": 0,
    "diuretic_thiazide_peri": 0,
    "diuretic_low_ceiling_non_thiazide_peri": 0,
    "diuretic_metolazone_peri": 0,
    "diuretic_indapamide_peri": 0,
    "diuretic_mra_peri": 0,
    "diuretic_vasopressin_antagonist_peri": 0,
    "anti_arrhythmic_any_peri": 0,
    "anti_arrhythmic_amiodarone_peri": 0,
    "anti_arrhythmic_disopyramide_peri": 0,
    "digoxin_peri": 0,
    "amyloid_therapeutics_tafamidis_peri": 0,
    "amyloid_therapeutics_diflunisal_peri": 0,
    "amyloid_therapeutics_patisiran_peri": 0,
    "amyloid_therapeutics_inotersen_peri": 0,
    "lipid_statin_peri": 0,
    "lipid_fibrate_peri": 0,
    "lipid_ezetimibe_peri": 0,
    "lipid_PCKSK9_peri": 0,
    "lipid_other_peri": 0,
    "glucose_insulin_peri": 0,
    "glucose_glp_1_agonsist_peri": 0,
    "glucose_ohg_biguanide_peri": 0,
    "glucose_ohg_alphagluc_peri": 0,
    "glucose_ohg_dpp_4_peri": 0,
    "glucose_ohg_sglt_2_peri": 0,
    "glucose_ohg_thiazolidine_peri": 0,
    "glucose_ohg_repaglinide_peri": 0,
    "glucose_ohg_sulfonylurea_peri": 0,
    "glucose_ohg_other_peri": 0,
    "smoking_cessation_oral_peri": 0,
    "smoking_cessation_nicotine_replacement_peri": 0
}

form_values = default_values.copy()

def is_valid(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True

mandatory_fields = [
    "patient_id",
    "demographics_age_index_ecg",
    "ecg_resting_hr",
    "ecg_resting_pr",
    "ecg_resting_qrs",
    "ecg_resting_qtc"
]

img_c1, img_c2, img_c3 = st.columns(3)
with img_c2:
    st.image("title.png", width=300)
st.title("Risk Prediction for Atrial Fibrillation")
st.write("All fields marked with ⚠️ are required. Please fill them out before submitting.")

if "form_key" not in st.session_state:
    st.session_state["form_key"] = str(uuid.uuid4())

def unique_key(base):
    return f"{base}_{st.session_state['form_key']}"

def render_form():
    with st.form(key=st.session_state["form_key"], clear_on_submit=False):
        st.subheader("Patient Information")
        pi_c1, pi_c2, pi_c3 = st.columns(3)
        form_values["patient_id"] = pi_c1.text_input("Enter the Patient ID ⚠️", key=unique_key("patient_id"))
        g_options = {"Male": 1, "Female": 2}
        sel_gender = pi_c2.selectbox("Select the gender ⚠️", list(g_options.keys()))
        form_values["demographics_birth_sex"] = g_options[sel_gender]
        form_values["demographics_age_index_ecg"] = pi_c3.number_input("Enter the age ⚠️", min_value=0, max_value=120, value=0)
        st.divider()

        st.subheader("Cardiac Risk Information")
        form_values["hypertension_icd10"] = 1 if st.checkbox("Hypertension") else 0
        form_values["diabetes_combined"] = 1 if st.checkbox("Diabetes") else 0
        form_values["dyslipidemia_combined"] = 1 if st.checkbox("Dislipidemia") else 0
        st.divider()

        st.subheader("History of Cardiovascular Diseases")
        st.caption("Observations recorded at any time prior to or within 6 months after the index ECG.")
        c1, c2, c3 = st.columns(3)
        form_values["dcm_icd10"] = 1 if c1.checkbox("Dilated cardiomyopathy") else 0
        form_values["hcm_icd10"] = 1 if c2.checkbox("Hypertrophic cardiomyopathy") else 0
        form_values["myocarditis_icd10_prior"] = 1 if c2.checkbox("Myocarditis - acute") else 0
        form_values["pericarditis_icd10_prior"] = 1 if c3.checkbox("Pericarditis - acute") else 0
        form_values["aortic_aneurysm_icd10"] = 1 if c3.checkbox("Aortic aneurysm") else 0
        form_values["aortic_dissection_icd10_prior"] = 1 if c1.checkbox("Aortic dissection") else 0
        st.divider()

        st.subheader("History of Non-Cardiovascular diseases")
        st.caption("Observations recorded at any time prior to or within 6 months after the index ECG.")
        c4, c5, c6 = st.columns(3)
        form_values["pulmonary_htn_icd10"] = 1 if c4.checkbox("Pulmonary hypertension") else 0
        form_values["amyloid_icd10"] = 1 if c4.checkbox("Amyloidosis") else 0
        form_values["copd_icd10"] = 1 if c6.checkbox("COPD") else 0
        form_values["obstructive _sleep_apnea_icd10"] = 1 if c5.checkbox("Obstructive Sleep Apnea") else 0
        form_values["hyperthyroid_icd10"] = 1 if c4.checkbox("Hyperthyroidism") else 0
        form_values["hypothyroid_icd10"] = 1 if c4.checkbox("Hypothyroidism") else 0
        form_values["rheumatoid_arthritis_icd10"] = 1 if c5.checkbox("Rheumatoid arthritis") else 0
        form_values["sle_icd10"] = 1 if c5.checkbox("Systemic Lupus Erythematosus") else 0
        form_values["sarcoid_icd10"] = 1 if c6.checkbox("Sarcoidosis") else 0
        form_values["cancer_any_icd10"] = 1 if c6.checkbox("Cancer") else 0
        st.divider()

        st.subheader("Prior Cardiovascular Events and Procedures")
        st.caption("Observations recorded at any time prior to the index ECG.")
        c7, c8, c9 = st.columns(3)
        form_values["event_cv_hf_admission_icd10_prior"] = 1 if c7.checkbox("Heart failure admission") else 0
        form_values["event_cv_cad_acs_acute_mi_icd10_prior"] = 1 if c7.checkbox("Acute myocardial infarction") else 0
        form_values["event_cv_cad_acs_unstable_angina_icd10_prior"] = 1 if c7.checkbox("Unstable angina") else 0
        form_values["event_cv_cad_acs_other_icd10_prior"] = 1 if c8.checkbox("Other acute coronary syndrome") else 0
        form_values["event_cv_ep_vt_any_icd10_prior"] = 1 if c8.checkbox("Ventricular tachycardia") else 0
        form_values["event_cv_ep_sca_survived_icd10_cci_prior"] = 1 if c8.checkbox("Survived sudden cardiac arrest") else 0
        form_values["event_cv_cns_stroke_ischemic_icd10_prior"] = 1 if c9.checkbox("Acute ischemic stroke") else 0
        form_values["event_cv_cns_stroke_hemorrh_icd10_prior"] = 1 if c9.checkbox("Acute hemorrhagic stroke") else 0
        form_values["event_cv_cns_tia_icd10_prior"] = 1 if c9.checkbox("Transient ischemic attack (TIA)") else 0
        form_values["pci_prior"] = 1 if c9.checkbox("Percutaneous coronary intervention (PCI)") else 0
        form_values["cabg_prior"] = 1 if c7.checkbox("Coronary artery bypass grafting (CABG)") else 0
        form_values["transplant_heart_cci_prior"] = 1 if c8.checkbox("Heart transplantation") else 0
        form_values["lvad_cci_prior"] = 1 if c7.checkbox("LVAD implantation") else 0
        st.divider()

        st.subheader("Cardiovascular Devices")
        st.caption("Observations recorded at any time prior to the index ECG.")
        form_values["pacemaker_permanent_cci_prior"] = 1 if st.checkbox("Prior permanent pacemaker implantation") else 0
        form_values["crt_cci_prior"] = 1 if st.checkbox("Prior cardiac resynchronization therapy (CRT) implantation") else 0
        form_values["icd_cci_prior"] = 1 if st.checkbox("Prior internal cardioverter defibrillator (ICD) implantation") else 0
        st.divider()

        st.subheader("12 Lead ECG Information")
        c10, c11 = st.columns(2)
        form_values["ecg_resting_hr"] = c10.number_input("Enter the Heart rate ⚠️", step=1, value=None)
        form_values["ecg_resting_pr"] = c10.number_input("Enter the PR interval duration ⚠️", min_value=0, value=None)
        form_values["ecg_resting_qrs"] = c11.number_input("Enter the QRS complex duration ⚠️", min_value=0, value=None)
        form_values["ecg_resting_qtc"] = c11.number_input("Enter the Corrected QT interval ⚠️", min_value=0, value=None)
        
        st.subheader("Heart Rhythm and QRS Morphology")
        c12, c13, c14 = st.columns(3)
        form_values["ecg_resting_paced"] = 1 if c12.checkbox("Paced") else 0
        form_values["ecg_resting_bigeminy"] = 1 if c12.checkbox("Bigeminy") else 0
        form_values["ecg_resting_LBBB"] = 1 if c13.checkbox("LBBB") else 0
        form_values["ecg_resting_RBBB"] = 1 if c13.checkbox("RBBB") else 0
        form_values["ecg_resting_incomplete_LBBB"] = 1 if c13.checkbox("Incomplete LBBB") else 0
        form_values["ecg_resting_incomplete_RBBB"] = 1 if c13.checkbox("Incomplete RBBB") else 0
        form_values["ecg_resting_LAFB"] = 1 if c14.checkbox("LAFB") else 0
        form_values["ecg_resting_LPFB"] = 1 if c14.checkbox("LPFB") else 0
        form_values["ecg_resting_bifascicular_block"] = 1 if c14.checkbox("Bifascicular Block") else 0
        form_values["ecg_resting_trifascicular_block"] = 1 if c14.checkbox("Trifascicular Block") else 0
        form_values["ecg_resting_intraventricular_conduction_delay"] = 1 if c12.checkbox("Intraventricular Conduction Delay") else 0
        
        st.subheader("Laboratory Results")
        st.caption("Measurements taken within 3 years prior to or 1 year after the index 12-lead ECG.")
        lab_c1, lab_c2, lab_c3 = st.columns(3)
        lab_hgb = lab_c1.text_input("Enter the Hemoglobin", key=unique_key("hgb_peri"))
        lab_hct = lab_c2.text_input("Enter the Hematocrit", key=unique_key("hct_peri"))
        lab_rdw = lab_c3.text_input("Enter the Red Cell Distribution Width (RDW)", key=unique_key("rdw_peri"))
        lab_wbc = lab_c1.text_input("Enter the White Blood Cell Count (WBC)", key=unique_key("wbc_peri"))
        lab_plt = lab_c2.text_input("Enter the Platelet Count", key=unique_key("plt_peri"))
        lab_inr = lab_c3.text_input("Enter the International Normalized Ratio (INR)", key=unique_key("inr_peri"))
        lab_ptt = lab_c1.text_input("Enter the Partial Thromboplastin Time (PTT)", key=unique_key("ptt_peri"))
        lab_esr = lab_c2.text_input("Enter the Erythrocyte Sedimentation Rate (ESR)", key=unique_key("esr_peri"))
        lab_crp = lab_c3.text_input("Enter the High Sensitivity C-Reactive Protein (CRP)", key=unique_key("crp_high_sensitive_peri"))
        lab_albumin = lab_c1.text_input("Enter the Albumin level", key=unique_key("albumin_peri"))
        lab_tibc = lab_c2.text_input("Enter the Total Iron Binding Capacity (TIBC)", key=unique_key("tibc_peri"))
        lab_alka = lab_c3.text_input("Enter the Alkaline Phosphatase", key=unique_key("alkaline_phophatase_peri"))
        lab_alt = lab_c1.text_input("Enter the Alanine Transaminase (ALT)", key=unique_key("alanine_transaminase_peri"))
        lab_ast = lab_c2.text_input("Enter the aspartate Transaminase (AST)", key=unique_key("aspartate_transaminase_peri"))
        lab_bilirubin_total = lab_c3.text_input("Enter Total Bilirubin", key=unique_key("bilirubin_total_peri"))
        lab_bilirubin_direct = lab_c1.text_input("Enter Direct Bilirubin", key=unique_key("bilirubin_direct_peri"))
        lab_urea = lab_c2.text_input("Enter Urea", key=unique_key("urea_peri"))
        lab_creatinine = lab_c3.text_input("Enter Creatinine", key=unique_key("creatinine_peri"))
        lab_urine = lab_c1.text_input("Enter Albumin/Creatinine Ratio", key=unique_key("urine_alb_cr_ratio_peri"))
        lab_sodium = lab_c2.text_input("Enter Sodium", key=unique_key("sodium_peri"))
        lab_potassium = lab_c3.text_input("Enter Potassium", key=unique_key("potassium_peri"))
        lab_chloride = lab_c1.text_input("Enter Chloride", key=unique_key("chloride_peri"))
        lab_ferritin = lab_c2.text_input("Enter Closest Serum Ferritin", key=unique_key("ferritin_peri"))
        lab_ck = lab_c3.text_input("Enter Creatine Kinase", key=unique_key("ck_peri"))
        lab_troponin = lab_c1.text_input("Enter Highest Troponin", key=unique_key("troponin_t_hs_peri_highest"))
        lab_ntprobnp = lab_c2.text_input("Enter Highest NT-proBNP", key=unique_key("NTproBNP_peri_highest"))
        
        st.caption("Measurements taken within 3 years prior to or 1 year after the index 12-lead ECG.")
        lab_c4, lab_c5, lab_c6 = st.columns(3)
        lab_glucose_fasting = lab_c4.text_input("Enter Highest Fasting Glucose", key=unique_key("glucose_fasting_peri_highest"))
        lab_glucose_random = lab_c5.text_input("Enter Highest Random Glucose", key=unique_key("glucose_random_peri_highest"))
        lab_hga1c = lab_c6.text_input("Enter Highest HbA1C", key=unique_key("hga1c_peri_highest"))
        lab_tchol = lab_c4.text_input("Enter Highest Total Cholesterol", key=unique_key("tchol_peri_highest"))
        lab_ldl = lab_c5.text_input("Enter Highest LDL Cholesterol", key=unique_key("ldl_peri_highest"))
        lab_hdl = lab_c6.text_input("Enter Lowest HDL Cholesterol ", key=unique_key("hdl_peri_lowest"))
        lab_tg = lab_c4.text_input("Enter Highest Serum Triglycerides", key=unique_key("tg_peri_highest"))
        
        def process_lab(input_str, default):
            if input_str.strip() == "":
                return default
            else:
                try:
                    return float(input_str)
                except:
                    return default

        form_values["hgb_peri"] = process_lab(lab_hgb, default_values["hgb_peri"])
        form_values["hct_peri"] = process_lab(lab_hct, default_values["hct_peri"])
        form_values["rdw_peri"] = process_lab(lab_rdw, default_values["rdw_peri"])
        form_values["wbc_peri"] = process_lab(lab_wbc, default_values["wbc_peri"])
        form_values["plt_peri"] = process_lab(lab_plt, default_values["plt_peri"])
        form_values["inr_peri"] = process_lab(lab_inr, default_values["inr_peri"])
        form_values["ptt_peri"] = process_lab(lab_ptt, default_values["ptt_peri"])
        form_values["esr_peri"] = process_lab(lab_esr, default_values["esr_peri"])
        form_values["crp_high_sensitive_peri"] = process_lab(lab_crp, default_values["crp_high_sensitive_peri"])
        form_values["albumin_peri"] = process_lab(lab_albumin, default_values["albumin_peri"])
        form_values["tibc_peri"] = process_lab(lab_tibc, default_values["tibc_peri"])
        form_values["alkaline_phophatase_peri"] = process_lab(lab_alka, default_values["alkaline_phophatase_peri"])
        form_values["alanine_transaminase_peri"] = process_lab(lab_alt, default_values["alanine_transaminase_peri"])
        form_values["aspartate_transaminase_peri"] = process_lab(lab_ast, default_values["aspartate_transaminase_peri"])
        form_values["bilirubin_total_peri"] = process_lab(lab_bilirubin_total, default_values["bilirubin_total_peri"])
        form_values["bilirubin_direct_peri"] = process_lab(lab_bilirubin_direct, default_values["bilirubin_direct_peri"])
        form_values["urea_peri"] = process_lab(lab_urea, default_values["urea_peri"])
        form_values["creatinine_peri"] = process_lab(lab_creatinine, default_values["creatinine_peri"])
        form_values["urine_alb_cr_ratio_peri"] = process_lab(lab_urine, default_values["urine_alb_cr_ratio_peri"])
        form_values["sodium_peri"] = process_lab(lab_sodium, default_values["sodium_peri"])
        form_values["potassium_peri"] = process_lab(lab_potassium, default_values["potassium_peri"])
        form_values["chloride_peri"] = process_lab(lab_chloride, default_values["chloride_peri"])
        form_values["ferritin_peri"] = process_lab(lab_ferritin, default_values["ferritin_peri"])
        form_values["ck_peri"] = process_lab(lab_ck, default_values["ck_peri"])
        form_values["troponin_t_hs_peri_highest"] = process_lab(lab_troponin, default_values["troponin_t_hs_peri_highest"])
        form_values["NTproBNP_peri_highest"] = process_lab(lab_ntprobnp, default_values["NTproBNP_peri_highest"])
        form_values["glucose_fasting_peri_highest"] = process_lab(lab_glucose_fasting, default_values["glucose_fasting_peri_highest"])
        form_values["glucose_random_peri_highest"] = process_lab(lab_glucose_random, default_values["glucose_random_peri_highest"])
        form_values["hga1c_peri_highest"] = process_lab(lab_hga1c, default_values["hga1c_peri_highest"])
        form_values["tchol_peri_highest"] = process_lab(lab_tchol, default_values["tchol_peri_highest"])
        form_values["ldl_peri_highest"] = process_lab(lab_ldl, default_values["ldl_peri_highest"])
        form_values["hdl_peri_lowest"] = process_lab(lab_hdl, default_values["hdl_peri_lowest"])
        form_values["tg_peri_highest"] = process_lab(lab_tg, default_values["tg_peri_highest"])
        st.divider()

        st.subheader("Medications")
        st.caption("Usage recorded within 90 days before or after the index 12-lead ECG.")  
        c15, c16, c17 = st.columns(3)
        form_values["anti_platelet_oral_non_asa_any_peri"] = 1 if c15.checkbox("Non-Aspirin anti-platelet") else 0
        form_values["anti_coagulant_oral_any_peri"] = 1 if c15.checkbox("Oral Anticoagulants") else 0
        form_values["nitrates_any_peri"] = 1 if c15.checkbox("Any nitrate medication use (mononitrate, dinitrate, trinitrate)") else 0
        form_values["ranolazine_peri"] = 1 if c16.checkbox("Ranolazine") else 0
        form_values["acei_peri"] = 1 if c16.checkbox("Angiotensin converting enzyme inhibitor (ACEi)") else 0
        form_values["arb_peri"] = 1 if c16.checkbox("Angiotensin Receptor Blocker (ARB)") else 0
        form_values["arni_entresto_peri"] = 1 if c17.checkbox("Entresto") else 0
        form_values["beta_blocker_any_peri"] = 1 if c17.checkbox("Beta-blocker") else 0
        form_values["ivabradine_peri"] = 1 if c17.checkbox("Ivabradine") else 0
        form_values["ccb_dihydro_peri"] = 1 if c17.checkbox("Dihydropyridine calcium channel blocker") else 0
        form_values["ccb_non_dihydro_peri"] = 1 if c15.checkbox("Non-dihydropyridine calcium channel blocker") else 0
        form_values["diuretic_loop_peri"] = 1 if c16.checkbox("Loop diuretic medication") else 0
        form_values["diuretic_thiazide_peri"] = 1 if c15.checkbox("Thiazide diuretic") else 0
        form_values["diuretic_low_ceiling_non_thiazide_peri"] = 1 if c17.checkbox("Low-ceiling non-thiazide diuretic") else 0
        form_values["diuretic_metolazone_peri"] = 1 if c17.checkbox("Metolazone") else 0
        form_values["diuretic_indapamide_peri"] = 1 if c17.checkbox("Indapamide") else 0
        form_values["diuretic_mra_peri"] = 1 if c15.checkbox("Potassium sparing diuretic") else 0
        form_values["diuretic_vasopressin_antagonist_peri"] = 1 if c16.checkbox("Vasopressin antagonist diuretic") else 0
        form_values["anti_arrhythmic_any_peri"] = 1 if c15.checkbox("Any anti-arrhythmic medication") else 0
        form_values["anti_arrhythmic_amiodarone_peri"] = 1 if c16.checkbox("Amiodarone (anti-arrhythmic) medication") else 0
        form_values["anti_arrhythmic_disopyramide_peri"] = 1 if c17.checkbox("Disopyramide (anti-arrhythmic) medication") else 0
        form_values["digoxin_peri"] = 1 if c17.checkbox("Digoxin") else 0
        form_values["amyloid_therapeutics_tafamidis_peri"] = 1 if c17.checkbox("Tafamidis (amyloid therapeutic)") else 0
        form_values["amyloid_therapeutics_diflunisal_peri"] = 1 if c17.checkbox("Diflunisal (amyloid therapeutic)") else 0
        form_values["amyloid_therapeutics_patisiran_peri"] = 1 if c15.checkbox("Patisiran (amyloid therapeutic)") else 0
        form_values["amyloid_therapeutics_inotersen_peri"] = 1 if c16.checkbox("Inotersen (amyloid therapeutic)") else 0
        form_values["lipid_statin_peri"] = 1 if c15.checkbox("Statin (lipid lowering)") else 0
        form_values["lipid_fibrate_peri"] = 1 if c17.checkbox("Fibrate (lipid lowering)") else 0
        form_values["lipid_ezetimibe_peri"] = 1 if c17.checkbox("Ezetimibe (lipid lowering)") else 0
        form_values["lipid_PCKSK9_peri"] = 1 if c17.checkbox("PCSK9-inhibitor (lipid lowering)") else 0
        form_values["lipid_other_peri"] = 1 if c15.checkbox("Other lipid lowering") else 0
        form_values["glucose_insulin_peri"] = 1 if c16.checkbox("Insulin (glucose lowering)") else 0
        form_values["glucose_glp_1_agonsist_peri"] = 1 if c15.checkbox("GLP-1 agonist (glucose lowering)") else 0
        form_values["glucose_ohg_biguanide_peri"] = 1 if c15.checkbox("Biguanide (oral hypoglycemic)") else 0
        form_values["glucose_ohg_alphagluc_peri"] = 1 if c15.checkbox("Alpha-glucosidase inhibitor (oral hypoglycemic)") else 0
        form_values["glucose_ohg_dpp_4_peri"] = 1 if c15.checkbox("DPP-4 inhibitor (oral hypoglycemic)") else 0
        form_values["glucose_ohg_sglt_2_peri"] = 1 if c16.checkbox("SGLT2 inhibitor (oral hypoglycemic)") else 0
        form_values["glucose_ohg_thiazolidine_peri"] = 1 if c16.checkbox("Thiazolidine (oral hypoglycemic)") else 0
        form_values["glucose_ohg_repaglinide_peri"] = 1 if c16.checkbox("Repaglinide (oral hypoglycemic)") else 0
        form_values["glucose_ohg_sulfonylurea_peri"] = 1 if c15.checkbox("Sulfonylurea (oral hypoglycemic)") else 0
        form_values["glucose_ohg_other_peri"] = 1 if c16.checkbox("Other oral hypoglycemic") else 0
        form_values["smoking_cessation_oral_peri"] = 1 if c16.checkbox("Oral smoking cessation agent") else 0
        form_values["smoking_cessation_nicotine_replacement_peri"] = 1 if c16.checkbox("Nicotine-replacement therapy") else 0
       
        submit_flag = st.form_submit_button("Submit for Risk Prediction 🚀")
        save_flag = st.form_submit_button("Save Patient Record to S3 ☁️")
        return submit_flag, save_flag

form_container = st.empty()

with form_container:
    submit_flag, save_flag = render_form()

if submit_flag:
    if any(not is_valid(form_values[f]) for f in mandatory_fields):
        st.error("Please complete all mandatory fields with valid values.")
    else:
        df_input = pd.DataFrame([form_values])
        try:
            pred, score, life_yrs = make_prediction(form_values)
            display_results(form_values["patient_id"], pred, score, life_yrs)
            c1, c2 = st.columns(2)
            with c1:
                df_plot, scaler, pca, common_cols = create_pca_for_plotting(data, form_values.keys())
                x_new, y_new = transform_new_input_for_plotting(form_values, common_cols, scaler, pca)
                plot_pca_with_af_colors(df_plot, x_new, y_new)
            with c2:
                plot_distribution_with_afib_hue(data, form_values, "demographics_age_index_ecg", "Age Distribution")
            st.subheader("ECG Feature Distributions Compared to AFib Population")
            c1, c2 = st.columns(2)
            with c1:
                plot_distribution_with_afib_hue(data, form_values, "ecg_resting_hr", "Heart Rate (HR) Distribution")
                plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qrs", "QRS Duration Distribution")
            with c2:
                plot_distribution_with_afib_hue(data, form_values, "ecg_resting_pr", "PR Interval Distribution")
                plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qtc", "QTc Interval Distribution")
            # with st.expander("Review Your Input Data"):
            #     st.table(df_input.T)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if save_flag:
    st.info("Saving to S3 is currently disabled in this environment.")

if st.button("Clear Form for New Entry 🗑️", key="clear_form_btn"):
    st.session_state["form_key"] = str(uuid.uuid4())
    form_container.empty()
    with form_container:
        render_form()
