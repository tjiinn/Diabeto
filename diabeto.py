import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64
from scipy.stats import spearmanr
import networkx as nx

# Page config
st.set_page_config(page_title="Diabeto", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color: #1f77b4; padding-bottom: 1rem;}
h2 {color: #2c3e50; padding-top: 1rem;}
.stAlert {margin-top: 1rem;}
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.stat-label {
    font-size: 1rem;
    opacity: 0.95;
}

.intro-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 2rem;
    border-radius: 15px;
    border-left: 5px solid #dc3545;
    margin: 1rem 0;
    color: #1a1a1a !important;
}

.intro-box p,
.intro-box li,
.intro-box b,
.intro-box h3,
.intro-box h4 {
    color: #1a1a1a !important;
}

.warning-box {
    background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #e53e3e;
    margin: 1rem 0;
    color: #1a1a1a !important;
}

.warning-box p,
.warning-box h2,
.warning-box h3 {
    color: #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)
    
def safe_load_model(path):
    if not Path(path).exists():
        st.warning(f"⚠️ Model file not found: {path}")
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Could not load model {path}: {e}")
        return None

# Load models
@st.cache_resource
def load_models():
    models = {}

    models['stage1_clinical'] = safe_load_model('xgb_stage1_clinical.pkl')
    models['stage1_non_clinical'] = safe_load_model('xgb_stage1_non_clinical.pkl')
    models['stage2_clinical'] = safe_load_model('xgb_stage2_clinical.pkl')
    models['stage2_non_clinical'] = safe_load_model('xgb_stage2_non_clinical.pkl')

    # Clustering & scalers
    models['cluster_clinical'] = safe_load_model('clustering_model_clinical.pkl')
    models['cluster_non_clinical'] = safe_load_model('clustering_model_non_clinical.pkl')
    models['scaler_clinical'] = safe_load_model('scaler_clinical.pkl')
    models['scaler_non_clinical'] = safe_load_model('scaler_non_clinical.pkl')

    return models

# Load dataset for EDA
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
        return df
    except:
        return None

# load clustering profiles
@st.cache_data
def load_cluster_profiles():
    try:
        profiles_clinical = pd.read_csv('cluster_profiles_clinical.csv')
        profiles_non_clinical = pd.read_csv('cluster_profiles_non_clinical.csv')
        return profiles_clinical, profiles_non_clinical
    except:
        return None, None

# load cluster recommendations
def get_cluster_recommendations(cluster_row, cluster_type):
    recs = []
    row = cluster_row

    bmi = row.get('BMI', 0)
    age = row.get('Age', 0)
    gen_health = row.get('GenHlth', 0)
    phys_health = row.get('PhysHlth', 0)
    phys_activity = row.get('PhysActivity', 0)
    smoker = row.get('Smoker', 0)
    fruits = row.get('Fruits', 0)
    veggies = row.get('Veggies', 0)
    alcohol = row.get('HvyAlcoholConsump', 0)
    high_bp = row.get('HighBP', 0)
    high_chol = row.get('HighChol', 0)
    diabetes_pct = row.get('Diabetes_Pct', 0)
    prediabetes_pct = row.get('Prediabetes_Pct', 0)

    # Risk level (insert at front after determining)
    if diabetes_pct > 30:
        recs.append(("error", "🚨 High Diabetes Risk", "Your profile group has high diabetes prevalence. Get an HbA1c test, see an endocrinologist, and implement lifestyle changes immediately."))
    elif prediabetes_pct > 30:
        recs.append(("warning", "⚠️ Prediabetes Risk", "Your profile group has elevated prediabetes risk. Prevention is possible: lose 5–7% body weight, exercise 150+ min/week, and monitor blood sugar."))
    else:
        recs.append(("success", "✅ Lower Risk Group", "Your profile group has lower diabetes prevalence. Maintain your healthy habits and keep up with annual health screenings."))

    if bmi > 30:
        recs.append(("error", "🏋️ Weight Management", "Your BMI indicates obesity. Aim for 5–7% weight loss through a calorie deficit (~500 cal/day) and regular exercise."))
    elif bmi > 25:
        recs.append(("warning", "⚖️ Healthy Weight", "Your BMI is in the overweight range. Focus on portion control and 30+ minutes of daily activity."))

    if phys_activity < 0.5:
        recs.append(("warning", "🏃 Get Moving", "Low physical activity detected. Start with 30-minute daily walks and progress to 150 min/week of moderate exercise."))

    if fruits < 0.5 or veggies < 0.5:
        recs.append(("warning", "🥗 Improve Diet", "Increase fruit and vegetable intake to 5+ servings daily. Focus on fiber-rich foods and whole grains."))

    if smoker > 0.3:
        recs.append(("error", "🚭 Quit Smoking", "Smoking significantly increases diabetes risk. Seek cessation programmes, nicotine replacement therapy, or counselling."))

    if alcohol > 0.3:
        recs.append(("warning", "🍺 Limit Alcohol", "Heavy alcohol consumption detected. Reduce to moderate levels (<7 drinks/week for women, <14 for men)."))

    if high_bp > 0.3:
        recs.append(("warning", "💊 Blood Pressure Control", "High BP is prevalent in your profile group. Monitor regularly, reduce sodium intake, manage stress, and consult a doctor about medication."))

    if high_chol > 0.3:
        recs.append(("warning", "🩺 Cholesterol Management", "High cholesterol is common in your profile group. Reduce saturated fats, increase omega-3s, and consider statin therapy if prescribed."))

    if gen_health > 3.5:
        recs.append(("warning", "❤️ Overall Health", "Poor general health is reported in your group. Schedule a comprehensive medical check-up and focus on preventive care."))

    if phys_health > 10:
        recs.append(("warning", "🤕 Physical Health", "Your group reports many poor physical health days. Consult a healthcare provider; physical therapy may help manage chronic pain."))

    if age > 9:
        recs.append(("info", "👴 Senior Health", "Age increases diabetes risk. Prioritise regular screenings, maintain muscle mass through strength training, and ensure adequate vitamin D."))

    return recs

# Feature engineering function
def create_engineered_features(X):
    """
    Create interaction and polynomial features that may help distinguish
    between diabetes stages - MATCHES NOTEBOOK EXACTLY
    """
    X_eng = X.copy()
    
    # BMI-based interactions
    X_eng['BMI_Age'] = X_eng['BMI'] * X_eng['Age']
    X_eng['BMI_GenHlth'] = X_eng['BMI'] * X_eng['GenHlth']
    X_eng['BMI_squared'] = X_eng['BMI'] ** 2
    
    # Health status combinations
    X_eng['Poor_Health_Score'] = (
        X_eng['GenHlth'] * 2 + 
        X_eng['PhysHlth'] / 10
    )
    
    # Lifestyle score
    X_eng['HealthyLifestyle'] = (
        X_eng['PhysActivity'] + 
        X_eng['Fruits'] + 
        X_eng['Veggies'] - 
        X_eng['Smoker'] - 
        X_eng['HvyAlcoholConsump']
    )
    
    # Age-based interactions
    X_eng['Age_GenHlth'] = X_eng['Age'] * X_eng['GenHlth']
    X_eng['Age_PhysHlth'] = X_eng['Age'] * (X_eng['PhysHlth'] / 30)
    
    # Mobility score
    X_eng['Mobility_Score'] = X_eng['DiffWalk'] * X_eng['PhysHlth']
    
    # Risk score combinations (clinical)
    if 'HighBP' in X_eng.columns:
        X_eng['Total_Risk_Score'] = (
            X_eng['HighBP'] + X_eng['HighChol'] + 
            X_eng['Stroke'] + X_eng['HeartDiseaseorAttack']
        )
        
        # Metabolic syndrome proxy
        X_eng['MetabolicSyndrome_Proxy'] = (
            (X_eng['BMI'] > 30).astype(int) + 
            X_eng['HighBP'] + 
            X_eng['HighChol']
        )
    
    # Health deterioration indicators (transition markers)
    X_eng['Health_Decline'] = X_eng['GenHlth'] * (1 + X_eng['PhysHlth']/30)
    
    # Age-risk interactions (prediabetes often occurs in younger ages)
    X_eng['Age_BMI_Risk'] = X_eng['Age'] * X_eng['BMI'] / 100
    X_eng['Young_High_BMI'] = ((X_eng['Age'] < 7) & (X_eng['BMI'] > 30)).astype(int)
    
    # Lifestyle factors (prediabetes more responsive to lifestyle)
    X_eng['Lifestyle_Risk'] = (
        (1 - X_eng['PhysActivity']) + 
        (1 - X_eng['Fruits']) + 
        (1 - X_eng['Veggies']) + 
        X_eng['Smoker'] + 
        X_eng['HvyAlcoholConsump']
    )
    return X_eng

# Hierarchical prediction
def hierarchical_predict(model_s1, model_s2, X_test, threshold_s1=0.3, threshold_s2=0.7):
    stage1_proba = model_s1.predict_proba(X_test)[:, 1]
    stage1_pred = (stage1_proba >= threshold_s1).astype(int)
    final_pred = np.zeros(len(X_test), dtype=int)
    diabetes_mask = stage1_pred == 1
    if diabetes_mask.sum() > 0:
        X_diabetes = X_test[diabetes_mask]
        stage2_proba = model_s2.predict_proba(X_diabetes)[:, 1]
        stage2_pred = (stage2_proba >= threshold_s2).astype(int)
        stage2_pred_remapped = stage2_pred + 1
        final_pred[diabetes_mask] = stage2_pred_remapped
    return final_pred, stage1_proba

# Helper functions
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
img1 = get_base64_image("normal.png")
img2 = get_base64_image("diabetes.jpeg")

def map_age_to_category(age):
    if age <= 24: return 1
    elif age <= 29: return 2
    elif age <= 34: return 3
    elif age <= 39: return 4
    elif age <= 44: return 5
    elif age <= 49: return 6
    elif age <= 54: return 7
    elif age <= 59: return 8
    elif age <= 64: return 9
    elif age <= 69: return 10
    elif age <= 74: return 11
    elif age <= 79: return 12
    else: return 13

def bp_to_high_bp(systolic):
    return 1 if systolic >= 140 else 0

def chol_to_high_chol(total_chol):
    return 1 if total_chol >= 240 else 0

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

# Display results
def display_results(prediction, probability, module_type, input_eng, models):
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    results = {
        0: {"label": "No Diabetes", "color": "#28a745", "icon": "✅", "msg": "No significant diabetes risk indicators."},
        1: {"label": "Prediabetes", "color": "#ffc107", "icon": "⚠️", "msg": "Warning sign - lifestyle changes needed."},
        2: {"label": "Diabetes", "color": "#dc3545", "icon": "🚨", "msg": "Consult healthcare provider immediately."}
    }
    result = results[prediction]

    st.markdown(f"""
    <div style='background-color: {result['color']}; padding: 2rem; border-radius: 10px; color: white; text-align: center;'>
        <h1>{result['icon']} {result['label']}</h1>
        <p style='font-size: 1.2rem;'>{result['msg']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Score (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': result['color']},
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Summary")
        st.metric("Prediction", result['label'])
        st.metric("Risk Score", f"{probability*100:.1f}%")
        st.metric("Module", module_type.replace(" Assessment", ""))

    # --- CLUSTER-BASED RECOMMENDATIONS ---
    st.markdown("---")
    st.markdown("### 💡 Personalised Recommendations")

    profiles_clinical, profiles_non_clinical = load_cluster_profiles()
    is_clinical = "Clinical" in module_type and "Non" not in module_type

    cluster_model = models['cluster_clinical'] if is_clinical else models['cluster_non_clinical']
    scaler = models['scaler_clinical'] if is_clinical else models['scaler_non_clinical']
    profiles = profiles_clinical if is_clinical else profiles_non_clinical

    CLUSTER_NAMES = {
        'clinical': {
            0: ("Young Obese", "🧑‍⚕️", 
                "Younger individuals with high BMI and elevated lifestyle risk. This group often lacks chronic clinical conditions but is on a trajectory toward metabolic complications if habits are not addressed early.",
                ["Weight reduction", "Physical activity", "Dietary improvement"],
                "Early intervention is critical — lifestyle changes at this stage are highly effective at reversing risk trajectory."),
            1: ("Poor Health", "🤒",
                "Individuals with consistently poor self-reported health, high physical health burden, and multiple comorbidities. This group experiences the most days of poor physical health and significant difficulty with mobility.",
                ["Medical management", "Pain & mobility support", "Mental and physical wellbeing"],
                "Focus on working closely with healthcare providers to manage existing conditions while gradually improving lifestyle factors."),
            2: ("Healthy", "💪",
                "Generally healthy individuals with good lifestyle habits, lower BMI, and fewer clinical risk factors. This group has the lowest diabetes prevalence and serves as the baseline for healthy behaviours.",
                ["Maintenance", "Preventive screening", "Sustaining healthy habits"],
                "Keep up the great work — your priority is maintaining current habits and staying on top of routine health screenings."),
            3: ("Elderly High-Risk", "👴",
                "Older individuals with a high prevalence of clinical risk factors including high blood pressure, high cholesterol, heart disease history, and elevated BMI. This group carries the highest diabetes burden.",
                ["Blood pressure & cholesterol control", "Cardiovascular risk management", "Senior-specific health monitoring"],
                "Urgent attention to clinical risk factors is needed. Regular specialist consultations and medication adherence are essential at this stage."),
        },
        'non_clinical': {
            0: ("Poor Health & Lifestyle", "⚠️",
                "Individuals with unhealthy lifestyle patterns — low physical activity, poor diet, higher rates of smoking and alcohol use, and poor self-reported health. This group has significantly higher diabetes and prediabetes prevalence.",
                ["Lifestyle overhaul", "Physical activity", "Dietary habits", "Substance use reduction"],
                "Comprehensive lifestyle change is the most impactful intervention available to this group — small consistent steps can significantly shift risk."),
            1: ("Healthy Lifestyle", "✅",
                "Individuals with generally positive lifestyle behaviours — physically active, eating fruits and vegetables regularly, and reporting better overall health. This group has lower diabetes prevalence.",
                ["Habit maintenance", "Preventive awareness", "Routine check-ups"],
                "You are already on the right track. Focus on consistency and don't neglect annual health screenings even when feeling well."),
        }
    }

    if profiles is not None:
        try:
            input_scaled = scaler.transform(input_eng)
            cluster_id = int(cluster_model.predict(input_scaled)[0])

            cluster_row = profiles[profiles['Cluster_ID'] == cluster_id].iloc[0].to_dict()
            n_clusters = len(profiles)

            profile_key = 'clinical' if is_clinical else 'non_clinical'
            cluster_info = CLUSTER_NAMES[profile_key].get(cluster_id, (f"Group {cluster_id}", "🔵", "", [], ""))
            cluster_name, cluster_icon, cluster_profile, advice_focus, cluster_summary = cluster_info

            # ── Cluster badge ──
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                padding: 1.4rem 1.8rem;
                border-radius: 12px;
                margin-bottom: 1rem;
            '>
                <div style='color: #a8d8f0; font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.3rem;'>Your Profile Group</div>
                <div style='display: flex; align-items: center; gap: 0.8rem;'>
                    <span style='font-size: 2.2rem;'>{cluster_icon}</span>
                    <span style='color: white; font-size: 1.6rem; font-weight: 700;'>{cluster_name}</span>
                </div>
                <div style='color: #cce4f7; font-size: 0.88rem; margin-top: 0.5rem;'>
                    Matched to Cluster {cluster_id} of {n_clusters} · Recommendations tailored to people similar to you
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Cluster Insights Expander ──
            with st.expander("🔍 View Cluster Insights", expanded=False):

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    st.markdown("##### 👤 Profile")
                    st.markdown(f"""
                    <div style='
                        background: #f8fafc;
                        border-left: 4px solid #3498db;
                        border-radius: 8px;
                        padding: 0.9rem 1.1rem;
                        color: #2d3748;
                        font-size: 0.95rem;
                        line-height: 1.7;
                    '>{cluster_profile}</div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### 🎯 Advice Focus Areas")
                    focus_html = "".join([
                        f"<span style='display:inline-block; background:#ebf8ff; color:#2b6cb0; border-radius:20px; padding:0.25rem 0.75rem; margin:0.2rem; font-size:0.85rem; font-weight:600;'>• {f}</span>"
                        for f in advice_focus
                    ])
                    st.markdown(f"<div style='margin-top:0.3rem;'>{focus_html}</div>", unsafe_allow_html=True)

                with col_b:
                    st.markdown("##### 📊 Cluster Statistics")

                    diabetes_pct = cluster_row.get('Diabetes_Pct', 0)
                    prediabetes_pct = cluster_row.get('Prediabetes_Pct', 0)
                    no_diabetes_pct = cluster_row.get('No_Diabetes_Pct', 0)
                    cluster_size = int(cluster_row.get('Size', 0))

                    # Mini donut chart
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=['No Diabetes', 'Prediabetes', 'Diabetes'],
                        values=[no_diabetes_pct, prediabetes_pct, diabetes_pct],
                        hole=0.55,
                        marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
                        textinfo='percent',
                        textfont_size=11,
                        showlegend=True,
                    )])
                    fig_donut.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10)),
                        annotations=[dict(text=f"{cluster_size:,}<br>people", x=0.5, y=0.5, font_size=11, showarrow=False, font_color='#2d3748')]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                    # Key feature stats
                    stat_rows = [
                        ("BMI", f"{cluster_row.get('BMI', 0):.1f}"),
                        ("Avg Age Group", f"{cluster_row.get('Age', 0):.1f}"),
                        ("Physical Activity", f"{cluster_row.get('PhysActivity', 0)*100:.0f}% active"),
                        ("Smokers", f"{cluster_row.get('Smoker', 0)*100:.0f}%"),
                    ]
                    if is_clinical:
                        stat_rows += [
                            ("High BP", f"{cluster_row.get('HighBP', 0)*100:.0f}%"),
                            ("High Cholesterol", f"{cluster_row.get('HighChol', 0)*100:.0f}%"),
                        ]

                    for label, val in stat_rows:
                        st.markdown(f"""
                        <div style='display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid #e2e8f0; font-size:0.88rem;'>
                            <span style='color:#718096;'>{label}</span>
                            <span style='color:#2d3748; font-weight:600;'>{val}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Summary banner across full width
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
                    border-left: 4px solid #3182ce;
                    border-radius: 8px;
                    padding: 0.85rem 1.1rem;
                    margin-top: 1rem;
                    color: #2c5282;
                    font-size: 0.93rem;
                    line-height: 1.6;
                '>
                    <b>📝 Summary: </b>{cluster_summary}
                </div>
                """, unsafe_allow_html=True)

            # ── Recommendation cards ──
            st.markdown("#### 📋 Your Recommendations")

            recs = get_cluster_recommendations(cluster_row, module_type)

            level_styles = {
                "error":   {"bg": "#fff0f0", "border": "#e53e3e", "icon": "🚨", "label_color": "#c53030"},
                "warning": {"bg": "#fffbeb", "border": "#d69e2e", "icon": "⚠️", "label_color": "#b7791f"},
                "success": {"bg": "#f0fff4", "border": "#38a169", "icon": "✅", "label_color": "#276749"},
                "info":    {"bg": "#ebf8ff", "border": "#3182ce", "icon": "ℹ️", "label_color": "#2b6cb0"},
            }

            for level, title, body in recs:
                s = level_styles.get(level, level_styles["info"])
                st.markdown(f"""
                <div style='
                    background: {s["bg"]};
                    border-left: 4px solid {s["border"]};
                    border-radius: 8px;
                    padding: 0.9rem 1.2rem;
                    margin-bottom: 0.7rem;
                '>
                    <div style='font-weight: 700; color: {s["label_color"]}; margin-bottom: 0.2rem;'>
                        {s["icon"]} {title}
                    </div>
                    <div style='color: #2d3748; font-size: 0.95rem; line-height: 1.6;'>{body}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not generate cluster-based recommendations: {e}")

    else:
        st.info("Cluster profile data not found. Showing general recommendations.")
        if prediction == 0:
            st.success("**Maintain healthy lifestyle:** Continue exercise, balanced diet, regular check-ups")
        elif prediction == 1:
            st.warning("**Take action:** Consult doctor, increase activity (150+ min/week), lose 5–7% weight, healthy diet, monitor glucose")
        else:
            st.error("**URGENT:** See healthcare provider immediately, discuss medication, monitor glucose, dietitian consultation, supervised exercise")

    st.info("⚠️ **Disclaimer:** Educational purposes only. Not medical advice. Consult healthcare professionals for proper diagnosis.")

# Main app
def main():
    st.title("🏥 Diabeto")

    with st.sidebar:
        st.title("Navigation")
        
        # ✅ EDA FIRST (Landing Page)
        page = st.radio(
            "Go to",
            ["📊 Dashboard", "🔍 Diabetes Prediction"],
            index=0  # ensures EDA loads by default
        )
        
        st.markdown("---")
        st.info("""
        ⚠️ *For educational use only*
        """)

    if page == "📊 Dashboard":
        eda_page()

    elif page == "🔍 Diabetes Prediction":
        tab1, tab2 = st.tabs(["Clinical Assessment", "Non-Clinical Assessment"])
        with tab1:
            clinical_form()
        with tab2:
            non_clinical_form()

def clinical_form():
    st.markdown("### Clinical Assessment")
    st.info("💡 Includes medical test results and health history")
    
    models = load_models()
    if not models: return
    
    with st.form("clinical_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            age_input = st.number_input(
                "Enter Age",
                min_value=18,
                max_value=100,
                value=45,
                step=1
            )
            age = map_age_to_category(age_input)

            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            sex_val = 1 if sex == "Male" else 0
            
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 300, 70)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}", bmi_category(bmi))
            
            st.markdown("#### Clinical Measurements")
            systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 250, 120, help="Normal: <120, High: >130")
            high_bp = bp_to_high_bp(systolic_bp)
            if high_bp: st.warning("⚠️ High BP detected")
            
            total_chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200, help="Normal: <200, High:>200")
            high_chol = chol_to_high_chol(total_chol)
            if high_chol: st.warning("⚠️ High cholesterol")
            
            chol_check = st.radio("Cholesterol checked (last 5 years)?", ["No", "Yes"], horizontal=True)
            chol_check_val = 1 if chol_check == "Yes" else 0
        
        with col2:
            st.markdown("#### Medical History")
            stroke = st.radio("History of Stroke?", ["No", "Yes"], horizontal=True)
            stroke_val = 1 if stroke == "Yes" else 0
            
            heart = st.radio("Heart Disease/Attack?", ["No", "Yes"], horizontal=True)
            heart_val = 1 if heart == "Yes" else 0

            st.markdown("#### Diet & Lifestyle")
            fruits = st.radio("Fruit 1+ times/day?", ["No", "Yes"], horizontal=True)
            fruits_val = 1 if fruits == "Yes" else 0
            
            veggies = st.radio("Vegetables 1+ times/day?", ["No", "Yes"], horizontal=True)
            veggies_val = 1 if veggies == "Yes" else 0
            
            smoker = st.radio("Smoked 100+ cigarettes lifetime?", ["No", "Yes"], horizontal=True)
            smoker_val = 1 if smoker == "Yes" else 0
            
            alcohol = st.radio("Heavy alcohol? (M:>14, F:>7 drinks/week)", ["No", "Yes"], horizontal=True)
            alcohol_val = 1 if alcohol == "Yes" else 0

            st.markdown("#### Health")
            phys_act = st.radio("Physical activity (past 30 days)?", ["No", "Yes"], horizontal=True)
            phys_act_val = 1 if phys_act == "Yes" else 0
            
            gen_health = st.select_slider("General Health", [1,2,3,4,5], 3, 
                format_func=lambda x: {1:"Poor",2:"Fair",3:"Good",4:"Very Good",5:"Excellent" }[x]) # inverse mapping for easier readability
            
            physical_health = st.slider("Poor physical health days (past 30)", 0, 30, 0)
            
            diff_walk = st.radio("Difficulty walking/climbing stairs?", ["No", "Yes"], horizontal=True)
            diff_walk_val = 1 if diff_walk == "Yes" else 0
        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)
        
        if submitted:
            input_data = pd.DataFrame({
                'BMI': [bmi],
                'Smoker': [smoker_val],
                'PhysActivity': [phys_act_val],
                'Fruits': [fruits_val],
                'Veggies': [veggies_val],
                'HvyAlcoholConsump': [alcohol_val],
                'GenHlth': [6 - gen_health],  # inverse mapping for easier readability
                'PhysHlth': [physical_health],
                'DiffWalk': [diff_walk_val],
                'Sex': [sex_val],
                'Age': [age],
                'HighBP': [high_bp],
                'HighChol': [high_chol],
                'CholCheck': [chol_check_val],
                'Stroke': [stroke_val],
                'HeartDiseaseorAttack': [heart_val]
            })
            input_eng = create_engineered_features(input_data)
            pred, proba = hierarchical_predict(models['stage1_clinical'], models['stage2_clinical'], input_eng)
            display_results(pred[0], proba[0], "Clinical Assessment", input_eng, models)

def non_clinical_form():
    st.markdown("### Non-Clinical Assessment")
    st.info("💡 Lifestyle and demographic factors only")
    
    models = load_models()
    if not models: return
    
    with st.form("non_clinical_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            age_input = st.number_input(
                "Enter Age",
                min_value=18,
                max_value=100,
                value=45,
                step=1
            )
            age = map_age_to_category(age_input)
            
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            sex_val = 1 if sex == "Male" else 0
            
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 300, 70)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}", bmi_category(bmi))
            
            st.markdown("#### Physical Activity")
            phys_act = st.radio("Physical activity (past 30 days)?", ["No", "Yes"], horizontal=True)
            phys_act_val = 1 if phys_act == "Yes" else 0
            
            diff_walk = st.radio("Difficulty walking/stairs?", ["No", "Yes"], horizontal=True)
            diff_walk_val = 1 if diff_walk == "Yes" else 0
        
        with col2:
            st.markdown("#### Diet & Lifestyle")
            fruits = st.radio("Fruit 1+ times/day?", ["No", "Yes"], horizontal=True)
            fruits_val = 1 if fruits == "Yes" else 0
            
            veggies = st.radio("Vegetables 1+ times/day?", ["No", "Yes"], horizontal=True)
            veggies_val = 1 if veggies == "Yes" else 0
            
            smoker = st.radio("Smoked 100+ cigarettes?", ["No", "Yes"], horizontal=True)
            smoker_val = 1 if smoker == "Yes" else 0
            
            alcohol = st.radio("Heavy alcohol?", ["No", "Yes"], horizontal=True)
            alcohol_val = 1 if alcohol == "Yes" else 0
            
            st.markdown("#### Health Status")
            gen_health = st.select_slider("General Health", [1,2,3,4,5], 3, 
                format_func=lambda x: {1:"Poor",2:"Fair",3:"Good",4:"Very Good",5:"Excellent" }[x]) # inverse mapping for easier readability
            
            physical_health = st.slider("Poor physical health days (past 30)", 0, 30, 0)
        
        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)
        
        if submitted:
            input_data = pd.DataFrame({
                'BMI': [bmi],
                'Smoker': [smoker_val],
                'PhysActivity': [phys_act_val],
                'Fruits': [fruits_val],
                'Veggies': [veggies_val],
                'HvyAlcoholConsump': [alcohol_val],
                'GenHlth': [6 - gen_health],  # inverse mapping for easier readability
                'PhysHlth': [physical_health],
                'DiffWalk': [diff_walk_val],
                'Sex': [sex_val],
                'Age': [age]
            })
            input_eng = create_engineered_features(input_data)
            pred, proba = hierarchical_predict(models['stage1_non_clinical'], models['stage2_non_clinical'], input_eng)
            display_results(pred[0], proba[0], "Non-Clinical Assessment", input_eng, models)

def eda_page():
    """Interactive EDA Page with Creative Visualizations"""
    
    st.markdown("## 📊 Understanding Diabetes")
    
    df = load_dataset()
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Importance", 
        "📈 Overview", 
        "🔍 Risk Factors",
        "💪 Lifestyle Impact"
    ])
    
    # ==================== IMPORTANCE TAB ====================
    with tab1:
        st.markdown("### 🚨 The Silent Epidemic: Understanding Diabetes")
        
        # Alarming statistics header
        st.markdown("""
        <div class="warning-box">
            <h2 style='color: #c53030; margin-top: 0;'>⚠️ Critical Statistics You Should Know</h2>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
            Diabetes is not just a health concern—it's a <b>global crisis</b> affecting millions of lives.
            Yet many people with diabetes or prediabetes don't even know they have it.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key statistics in visual cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
                <div class="stat-label">Adults with Diabetes (USA)</div>
                <div class="stat-number">38M</div>
                <div class="stat-label">📊 20% are UNDIAGNOSED</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
                <div class="stat-label">Prediabetic Americans</div>
                <div class="stat-number">1 in 3</div>
                <div class="stat-label">⚠️ 80% DON'T KNOW IT</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #c94b4b 0%, #4b134f 100%);">
                <div class="stat-label">Risk Multiplier</div>
                <div class="stat-number">5-15x</div>
                <div class="stat-label">🚨 Higher stroke risk with prediabetes</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # What is Diabetes section
        st.markdown("### 🔬 What is Diabetes?")
        st.markdown("""
        Diabetes is a chronic metabolic condition that affects how your body converts food into energy (CDC, 2024).
                    """)
        st.markdown(f"""

        <p style='font-size:1.05rem; line-height:1.8; margin-top:15px;'>
        <b>🔄 Normal Process:</b><br>
        Food consumed →  Body breaks down food into glucose (sugar) →  Glucose enters bloodstream →  Pancreas releases insulin → <br>
        Insulin helps cells absorb glucose for energy →  Blood sugar regulated ✅
        </p>
        <img src="data:image/png;base64,{img1}" style="width:100%; max-width:500px; display:block; margin:15px auto; border-radius:10px;">

        <p style='font-size:1.05rem; line-height:1.8; margin-top:15px;'>
        <b>⚠️ With Diabetes:</b><br>
        Food consumed →  Body breaks down food into glucose (sugar) →  Glucose enters bloodstream →  Body doesn't produce enough insulin OR <br> 
        cannot effectively use insulin produced →  Glucose stays in bloodstream → High blood sugar levels ❌
        </p>
        <img src="data:image/png;base64,{img2}" style="width:100%; max-width:500px; display:block; margin:15px auto; border-radius:10px;">
""", unsafe_allow_html=True)
        
        # Severity and complications
        st.markdown("### ⚡ Why You Should Act NOW")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.error("""
            **🚨 Severe Complications of Untreated Diabetes:**
            
            - 💔 **Kidney Failure** - Leading cause of dialysis
            - 🦵 **Lower-Limb Amputations** - Nerve damage & poor circulation
            - 👁️ **Adult Blindness** - Diabetes is the #1 cause
            - 🧠 **Stroke & Heart Disease** - 2-4x higher risk
            - 💀 **Premature Death** - Reduces life expectancy by 5-10 years
            
            *Source: CDC, 2024*
            """)
        
        with col2:
            st.success("""
            **✅ Good News: Early Detection Changes Everything!**
            
            - 🎯 **Prediabetes is REVERSIBLE** with lifestyle changes
            - 📉 **5-7% weight loss** can significantly reduce diabetes risk
            - 🏃 **150 min/week exercise** significantly lowers risk
            - 🥗 **Dietary changes** can prevent progression
            - 💊 **Early treatment** prevents complications
            
            *Know your risk before it's too late.*
            """)
        
        st.markdown("---")
        
        # The hidden danger
        st.markdown("### 🕵️ The Hidden Danger: Undiagnosed Cases")
        
        # Create visualization showing diagnosed vs undiagnosed
        fig = go.Figure()
        
        categories = ['Diabetes<br>(38M Americans)', 'Prediabetes<br>(98M Americans)']
        diagnosed = [80, 20]  # 80% diagnosed for diabetes, 20% for prediabetes
        undiagnosed = [20, 80]  # 20% undiagnosed for diabetes, 80% for prediabetes
        
        fig.add_trace(go.Bar(
            name='Diagnosed & Aware',
            y=categories,
            x=diagnosed,
            orientation='h',
            marker=dict(color='#48bb78'),
            text=[f'{val}%' for val in diagnosed],
            textposition='inside',
            textfont=dict(size=16, color='white'),
            hovertemplate='<b>%{y}</b><br>Diagnosed: %{x}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Undiagnosed & Unaware',
            y=categories,
            x=undiagnosed,
            orientation='h',
            marker=dict(color='#f56565'),
            text=[f'{val}%' for val in undiagnosed],
            textposition='inside',
            textfont=dict(size=16, color='white'),
            hovertemplate='<b>%{y}</b><br>Undiagnosed: %{x}%<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "The Shocking Reality: Most People Don't Know They're At Risk",
                'font': {'size': 20}
            },
            barmode='stack',
            height=300,
            xaxis={'title': 'Percentage', 'range': [0, 100]},
            yaxis={'title': ''},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.3, 'xanchor': 'center', 'x': 0.5}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        **💡 What this means:**
        - **7.6 million** Americans have diabetes but don't know it
        - **78 million** Americans have prediabetes but don't know it
        - Without screening, these individuals miss the critical window for prevention
        """)
        
        st.markdown("---")
        
        # Take control
        st.markdown("""
        <div class="intro-box" style="background: linear-gradient(135deg, #b8d7ff 0%, #6aaafc 100%); color: white; border-left: 5px solid #529eff;">
            <h3 style='color: white; margin-top: 0;'>🎯 Take Control of Your Health Today</h3>
            <p style='font-size: 1.1rem; line-height: 1.7;'>
            Knowledge is power. Understanding your diabetes risk is the <b>first step</b> toward prevention.
            </p>
            <p style='font-size: 1.1rem; line-height: 1.7; margin-bottom: 0;'>
            <b>Don't be part of the statistic. Get screened. Take action. Save your life.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== OVERVIEW TAB ====================
    with tab2:
        st.markdown("### Dataset Overview")
        st.markdown("""
            This dataset consists of survey responses to the Center of Disease Control and Prevention's (CDC) survey regarding diabetes from 2015.
            """)    
        
        # Key statistics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_samples = len(df)
        diabetes_count = (df['Diabetes_012'] == 2).sum()
        prediabetes_count = (df['Diabetes_012'] == 1).sum()
        no_diabetes_count = (df['Diabetes_012'] == 0).sum()
        
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="stat-label">Total Samples</div>
                <div class="stat-number">{total_samples:,}</div>
                <div class="stat-label">‎</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-label">Diabetes Cases</div>
                <div class="stat-number">{diabetes_count:,}</div>
                <div class="stat-label">{diabetes_count/total_samples*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #ffa751 0%, #ffe259 100%);">
                <div class="stat-label">Prediabetes Cases</div>
                <div class="stat-number">{prediabetes_count:,}</div>
                <div class="stat-label">{prediabetes_count/total_samples*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);">
                <div class="stat-label">Healthy</div>
                <div class="stat-number">{no_diabetes_count:,}</div>
                <div class="stat-label">{no_diabetes_count/total_samples*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Class Distribution - Interactive Pie and Bar
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Class Distribution (Pie Chart)")
            class_counts = df['Diabetes_012'].value_counts().sort_index()
            fig = go.Figure(data=[go.Pie(
                labels=['No Diabetes', 'Prediabetes', 'Diabetes'],
                values=class_counts.values,
                hole=0.4,
                marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
                textinfo='label+percent',
                textfont_size=14
            )])
            fig.update_layout(
                title="Diabetes Status Distribution",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Class Distribution (Bar Chart)")
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Diabetes', 'Prediabetes', 'Diabetes'],
                    y=class_counts.values,
                    marker=dict(color=['#28a745', '#ffc107', '#dc3545']),
                    text=class_counts.values,
                    textposition='auto',
                    texttemplate='%{text:,}',
                )
            ])
            fig.update_layout(
                title="Count by Diabetes Status",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📖 Understanding the Distribution"):
            st.markdown("""
            **What does this tell us?**
            
            - The majority of individuals in this dataset do not have diabetes (healthy group)
            - A significant portion has prediabetes, representing an "at-risk" population
            - The relatively smaller diabetes group represents those with confirmed diagnosis of diabetes
            
            **Why this matters:**
            - This distribution reflects real-world diabetes prevalence patterns
            - The prediabetic group highlights the importance of early intervention
            """)
        
        # BMI Distribution by Diabetes Status
        st.markdown("#### BMI Distribution by Diabetes Status")
        fig = go.Figure()
        
        for status, name, color in [(0, 'No Diabetes', '#28a745'), 
                                     (1, 'Prediabetes', '#ffc107'), 
                                     (2, 'Diabetes', '#dc3545')]:
            data = df[df['Diabetes_012'] == status]['BMI']
            fig.add_trace(go.Box(
                y=data,
                name=name,
                marker_color=color,
                boxmean='sd'
            ))
        
        fig.update_layout(
            yaxis_title="BMI",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📖 Interpreting BMI Patterns"):
            st.markdown("""
            **Key Observations:**
            - **Higher median BMI** in diabetes and prediabetes groups compared to healthy individuals
            - **Greater variability** in BMI among diabetic individuals (wider box)
            
            **Clinical Significance:**
            - BMI is a **strong predictor** of diabetes risk; every 1-unit increase in BMI above 25 increases diabetes risk
            - Weight management is a **modifiable risk factor**
            
            **BMI Categories:**
            - Underweight: <18.5
            - Normal: 18.5-24.9
            - Overweight: 25-29.9
            - Obese: ≥30
            """)
    
    # ==================== RISK FACTORS TAB ====================
    with tab3:
        st.markdown("### Key Risk Factors Analysis")
        
        # Load models to get feature importance
        models = load_models()
        
        if models:
            st.markdown("#### 🎯 Feature Importance: What Matters Most?")
            
            # Features used in model
            feature_names = ['BMI', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies',
                            'HvyAlcoholConsump', 'GenHlth', 'PhysHlth',
                            'DiffWalk', 'Sex', 'Age', 'HighBP', 'HighChol', 'CholCheck',
                            'Stroke', 'HeartDiseaseorAttack']

            # Compute correlations
            correlations = []
            for feature in feature_names:
                corr, _ = spearmanr(df[feature], df['Diabetes_012'])
                correlations.append(abs(corr))

            # Create DataFrame
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': correlations   # keep same column name to preserve chart code
            })

            # Round values
            feat_imp_df['Importance'] = feat_imp_df['Importance'].round(2)

            # Friendly names + descriptions
            feature_info = {
                'BMI': ('Body Mass Index', 'A measure of body fat based on height and weight.'),
                'Smoker': ('Smoking Status', 'Indicates whether the individual is a smoker.'),
                'PhysActivity': ('Physical Activity', 'Whether the individual engages in regular physical activity.'),
                'Fruits': ('Fruit Consumption', 'Frequency of fruit intake.'),
                'Veggies': ('Vegetable Consumption', 'Frequency of vegetable intake.'),
                'HvyAlcoholConsump': ('Heavy Alcohol Use', 'Indicates high levels of alcohol consumption.'),
                'GenHlth': ('General Health', 'Self-reported overall health status.'),
                'PhysHlth': ('Physical Health Days', 'Number of days physical health was not good.'),
                'DiffWalk': ('Difficulty Walking', 'Indicates difficulty in walking or climbing stairs.'),
                'Sex': ('Gender', 'Biological sex of the individual.'),
                'Age': ('Age', 'Age of the individual.'),
                'HighBP': ('High Blood Pressure', 'Indicates presence of hypertension.'),
                'HighChol': ('High Cholesterol', 'Indicates high cholesterol levels.'),
                'CholCheck': ('Cholesterol Check', 'Whether cholesterol has been checked recently.'),
                'Stroke': ('Stroke History', 'Indicates whether the individual has had history of a stroke.'),
                'HeartDiseaseorAttack': ('Heart Disease History', 'indicates whether the individual has had history of heart disease or attack.')
            }

            # Map readable names + descriptions
            feat_imp_df['Readable'] = feat_imp_df['Feature'].apply(lambda x: feature_info[x][0])
            feat_imp_df['Description'] = feat_imp_df['Feature'].apply(lambda x: feature_info[x][1])

            # Sort and take top 10
            feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True).tail(10)

            # Plot (UNCHANGED)
            fig = go.Figure(go.Bar(
                x=feat_imp_df['Importance'],
                y=feat_imp_df['Readable'],
                orientation='h',
                marker=dict(
                    color=feat_imp_df['Importance'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=feat_imp_df['Importance'].apply(lambda x: f"{x:.2f}"),
                textposition='auto',
                customdata=feat_imp_df['Description'],
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Correlation: %{x:.2f}<br>'
                    '%{customdata}<extra></extra>'
                )
            ))

            fig.update_layout(
                title="Top 10 Features which Affect Diabetes Risk",
                xaxis_title="Correlation Strength",
                yaxis_title="Feature",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📖 Understanding Feature Importance"):
                st.markdown("""
                Feature importance measures how much each health factor contributes to predicting diabetes risk;
                The higher the score, the more influential the feature is to contributing to diabetes risk.
                
                **How to interpret:**
                - **High importance** (>0.20): Critical predictors that strongly influence diabetes risk
                - **Medium importance** (0.10-0.20): Significant factors that contribute to predictions
                - **Low importance** (<0.15): Minor factors with limited predictive power
                
                **Key Takeaway:**
                Focus on improving the high-importance factors like **BMI, High Blood Pressure & High Cholesterol** to reduce diabetes risk.
                """)
        
        st.markdown("---")
        
        # Risk Factor Prevalence
        st.markdown("#### 📊 Risk Factor Prevalence by Diabetes Status")
        
        risk_factors = {
            'HighBP': 'High Blood Pressure',
            'HighChol': 'High Cholesterol',
            'Smoker': 'Smoking',
            'Stroke': 'History of Stroke',
            'HeartDiseaseorAttack': 'Heart Disease/Attack',
            'PhysActivity': 'Physical Activity',
            'Fruits': 'Daily Fruit Consumption',
            'Veggies': 'Daily Vegetable Consumption',
            'HvyAlcoholConsump': 'Heavy Alcohol Use',
            'DiffWalk': 'Difficulty Walking'
        }
        
        selected_factor = st.selectbox("Select Risk Factor to Explore", 
                                       list(risk_factors.keys()),
                                       format_func=lambda x: risk_factors[x])
        
        # Calculate prevalence
        prevalence_data = []
        for status in [0, 1, 2]:
            status_df = df[df['Diabetes_012'] == status]
            prevalence = (status_df[selected_factor] == 1).sum() / len(status_df) * 100
            prevalence_data.append(prevalence)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['No Diabetes', 'Prediabetes', 'Diabetes'],
            y=prevalence_data,
            marker=dict(color=['#28a745', '#ffc107', '#dc3545']),
            text=[f'{val:.1f}%' for val in prevalence_data],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>Prevalence: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Prevalence of {risk_factors[selected_factor]} by Diabetes Status",
            yaxis_title="Percentage (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic insights based on selected factor
        diff_diabetes_vs_healthy = prevalence_data[2] - prevalence_data[0]
        diff_prediabetes_vs_healthy = prevalence_data[1] - prevalence_data[0]
        
        if diff_diabetes_vs_healthy > 10:
            st.error(f"""
            **⚠️ Significant Risk Factor Alert!**
            
            People with diabetes are **{diff_diabetes_vs_healthy:.1f}% more likely** to have {risk_factors[selected_factor].lower()} compared to healthy individuals.
            
            {risk_factors[selected_factor]} is a **major risk factor** that requires attention.
            """)
        elif diff_diabetes_vs_healthy > 5:
            st.warning(f"""
            **📊 Moderate Association**
            
            {risk_factors[selected_factor]} shows a {diff_diabetes_vs_healthy:.1f} percentage point 
            difference between diabetic and healthy individuals.
            """)
        else:
            st.info(f"""
            **ℹ️ Weak Association**
            
            {risk_factors[selected_factor]} shows minimal difference ({diff_diabetes_vs_healthy:.1f} percentage points) 
            across diabetes status groups.
            """)
        
        st.markdown("---")
        
        # risk factor network
        st.markdown("#### 🕸️ How Risk Factors Connect")

        corr_features = ['BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
                        'Veggies', 'HvyAlcoholConsump', 'GenHlth']

        corr_matrix = df[corr_features].corr()

        # Build graph
        G = nx.Graph()

        # Add edges for strong correlations only
        threshold = 0.2
        for i in range(len(corr_features)):
            for j in range(i+1, len(corr_features)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(corr_features[i], corr_features[j], weight=abs(corr))

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Extract edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # Extract nodes
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        fig = go.Figure()

        # Edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

        # Nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(size=20, color='#e63946'),
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))

        fig.update_layout(
            title="How Risk Factors Are Connected",
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 What This Means"):
            st.markdown("""
            - Each circle is a **risk factor**
            - Lines show **factors that often occur together**
            
            **Key Insight:**
            Diabetes risk is not caused by one factor alone — 
            it's a **network of connected health issues**.
            
            Example:
            - High BP + High Cholesterol + BMI often cluster together
            - Lifestyle factors (diet, activity) influence multiple risks at once
            
            🎯 **Takeaway:** Improving one habit can positively affect multiple risks.
            """)
    
    # ==================== LIFESTYLE IMPACT TAB ====================
    with tab4:
        st.markdown("### 💪 The Power of Lifestyle Choices")        
        # Create comprehensive lifestyle score
        df_temp = df.copy()
        df_temp['Lifestyle_Score'] = (
            df_temp['PhysActivity'] + 
            df_temp['Fruits'] + 
            df_temp['Veggies'] - 
            df_temp['Smoker'] - 
            df_temp['HvyAlcoholConsump']
        )

        # Combined Impact Visualization - More Dynamic
        st.markdown("#### 🚀 The Compound Effect: Multiple Healthy Habits")

        # Calculate diabetes risk by number of healthy behaviors
        df_temp['Healthy_Behaviors'] = (
            (df_temp['PhysActivity'] == 1).astype(int) +
            (df_temp['Fruits'] == 1).astype(int) +
            (df_temp['Veggies'] == 1).astype(int) +
            (df_temp['Smoker'] == 0).astype(int) +
            (df_temp['HvyAlcoholConsump'] == 0).astype(int)
        )

        behavior_range = list(range(1, 6))
        behavior_risk = []

        for behaviors in behavior_range:
            subset = df_temp[df_temp['Healthy_Behaviors'] == behaviors]
            if len(subset) > 0:
                diabetes_pct = (subset['Diabetes_012'] == 2).sum() / len(subset) * 100
                behavior_risk.append(diabetes_pct)
            else:
                behavior_risk.append(None)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=behavior_range,
            y=behavior_risk,
            mode='lines+markers',
            line=dict(color='#dc3545', width=4),
            marker=dict(
                size=12,
                color=behavior_risk,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk %")
            ),
            text=[f'{val:.1f}%' if val is not None else "" for val in behavior_risk],
            textposition='top center',
            hovertemplate='<b>%{x} Healthy Behaviors</b><br>Diabetes Risk: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Diabetes Risk Drops with Each Additional Healthy Behavior!",
            xaxis_title="Number of Healthy Behaviors (1–5)",
            yaxis_title="Diabetes Risk (%)",
            height=400,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            annotations=[
                dict(
                    x=5, y=behavior_risk[-1],
                    text="Lowest<br>Risk",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#28a745",
                    ax=-40, ay=40,
                    font=dict(color="#28a745", size=12)
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        if behavior_risk[0] is not None and behavior_risk[-1] is not None:
            risk_reduction = behavior_risk[0] - behavior_risk[-1]

            st.success(f"""
            **🎉 POWERFUL IMPACT!**
            
            Increasing from **1 to 5 healthy behaviors** can reduce diabetes risk by approximately **{risk_reduction:.1f} percentage points**!
            
            **The 5 Healthy Behaviors:**
            1. 🏃 Regular physical activity (30+ min/day)
            2. 🍎 Eat fruit daily
            3. 🥦 Eat vegetables daily
            4. 🚭 Don't smoke
            5. 🍺 Limit alcohol consumption
            """)
        
        # Action
        st.markdown("### 🎯 Your Action Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="intro-box" style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); border-left: 5px solid #32a852;">
                <h4 style='margin-top: 0;'>✅ Start These Today</h4>
                <ul style='font-size: 1.05rem; line-height: 1.8;'>
                    <li><b>Walk 30 minutes</b> after dinner</li>
                    <li><b>Add one serving</b> of vegetables to lunch</li>
                    <li><b>Replace one soda</b> with water</li>
                    <li><b>Take stairs</b> instead of elevator</li>
                    <li><b>Get 7-8 hours</b> of sleep</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="intro-box" style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);">
                <h4 style='margin-top: 0;'>🎯 Build Up To These</h4>
                <ul style='font-size: 1.05rem; line-height: 1.8;'>
                    <li><b>150 minutes/week</b> of moderate exercise</li>
                    <li><b>5+ servings</b> of fruits/vegetables daily</li>
                    <li><b>Maintain healthy BMI</b> (18.5-24.9)</li>
                    <li><b>Quit smoking</b> (seek support if needed)</li>
                    <li><b>Annual health screenings</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("""
        **💡 Remember: You don't need to do everything at once!**
        
        Pick **ONE** change from the "Start Today" list and commit to it for 2 weeks. 
        Once it becomes a habit, add another. Small, consistent changes lead to lasting transformation.
        
        **Use our prediction tool** to track how these changes might impact your risk profile over time!
        """)

if __name__ == "__main__":
    main()