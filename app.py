import pickle
import pandas as pd
import numpy as np
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime

st.set_page_config(
    page_title="Stroke Riskometer",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #fff5f5; }
    </style>
    """,
    unsafe_allow_html=True
)


#Reset session state on refresh
if "reset_done" not in st.session_state:
    st.session_state.clear()
    st.session_state["reset_done"] = True

def generate_pdf(stroke_risk_percentage, risk_level, user_inputs, file_path="stroke_report.pdf"):
    """
    Generate a PDF report for stroke prediction results.
    """
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Stroke Risk Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Date & Time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report Generated: {current_time}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Risk results (omit percentage from PDF)
    story.append(Paragraph(f"<b>Risk Category:</b> {risk_level}", styles['Normal']))
    story.append(Spacer(1, 12))

    # User Inputs
    story.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))
    for key, value in user_inputs.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))


    doc.build(story)
    return file_path

st.markdown(
    """
    <div style="text-align:center; margin-bottom:56px;">
        <h1 style="color:#000000; margin-bottom:0.25rem;">🧠 Stroke Riskometer</h1>
        <p style="margin-top:0.25rem; color:#000000;">1 in 4 of us will have a stroke in our lifetime, but almost all strokes can be prevented. If you want to avoid a stroke the first step is to understand your individual risk factors.</p>
        
        
    </div>
    """,
    unsafe_allow_html=True
)


# Simple in-app router
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def go_to(page_name: str):
    st.session_state["page"] = page_name
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()


# Load the fixed model
@st.cache_resource
def load_model():
    try:
        with open("models/random_forest.pkl", "rb") as f:
            model_package = pickle.load(f)
            return model_package
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def render_home_page():
    # Increase button size on home
    st.markdown(
        """
        <style>
        div.stButton > button {
            font-size: 18px; /* larger text */
            padding: 14px 22px; /* larger padding */
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col_spacer, col2 = st.columns([1, 0.2, 1])
    with col1:
        st.markdown("<h3 style='color:#dc2626; margin-bottom:0.25rem; font-size:2rem;'>📊 Assess</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#dc2626; margin-top:0; font-size:1.125rem;'>Estimate your current stroke risk.</p>", unsafe_allow_html=True)
        if st.button("Go to Assessment", type="primary", use_container_width=True):
            go_to("assess")
    with col2:
        st.markdown("<h3 style='color:#000000; margin-bottom:0.25rem; font-size:2rem;'>⏱️ F.A.S.T</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#000000; margin-top:0; font-size:1.125rem;'>Learn about key stroke warning signs and risk factors.</p>", unsafe_allow_html=True)
        if st.button("Learn FAST & Risk Factors", use_container_width=True):
            go_to("fast")

    

def render_fast_info_page():
    if st.button("← Back to Home"):
        go_to("home")

    st.markdown(
        """
<div style="color:#000000;">
<h3>FAST: Recognize Stroke Symptoms Quickly</h3>
<ul>
<li><strong>F — Face drooping</strong>: One side of the face droops or is numb.</li>
<li><strong>A — Arm weakness</strong>: Weakness or numbness in one arm; can't raise both arms.</li>
<li><strong>S — Speech difficulty</strong>: Slurred speech or difficulty speaking/understanding.</li>
<li><strong>T — Time to call emergency services</strong>: If you observe any of these signs, call immediately.</li>
</ul>

<hr />
<h3>Common Risk Factors</h3>
<ul>
<li><strong>High blood pressure</strong></li>
<li><strong>Diabetes</strong></li>
<li><strong>High cholesterol</strong></li>
<li><strong>Smoking</strong></li>
<li><strong>Obesity and physical inactivity</strong></li>
<li><strong>Heart disease (e.g., atrial fibrillation)</strong></li>
<li><strong>Family history of stroke</strong></li>
<li><strong>Excessive alcohol use and stress</strong></li>
</ul>

<p><em>If you suspect a stroke, seek emergency care immediately. Early treatment saves brain function.</em></p>
</div>
        """,
        unsafe_allow_html=True
    )

def render_assessment_page():
    if st.button("← Back to Home"):
        go_to("home")

    model_package = load_model()
    if model_package is None:
        st.error("Failed to load model. Please check the model file.")
        st.stop()

    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']

    # Input form
    st.markdown("<h3 style='color:#000000; margin:0;'>📝 Enter Details</h3>", unsafe_allow_html=True)
    # Make form control labels red
    st.markdown(
        """
        <style>
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stTextInput"] label {
            color: #dc2626 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p style='color:#dc2626; font-weight:600;'>👤 Demographics & Medical History</p>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=0, max_value=120)
        sex = st.selectbox("Sex", ["Select","Female", "Male"])
        hypertension = st.selectbox("Hypertension", ["Select","No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["Select","No", "Yes"])
        family_history = st.selectbox("Family History of Stroke", ["Select","No", "Yes"])

        st.markdown("<p style='color:#dc2626; font-weight:600;'>🏃 Lifestyle</p>", unsafe_allow_html=True)
        work_type = st.selectbox("Work Type",["Select","Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Select","Rural", "Urban"])
        smoking_status = st.selectbox("Smoking Status",["Select","Never", "Formerly", "Currently", "Unknown"])

    with col2:
        st.markdown("<p style='color:#dc2626; font-weight:600;'>🧪 Clinical Measurements</p>", unsafe_allow_html=True)
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",min_value=0.0, max_value=300.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0)
        blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)",min_value=0, max_value=200)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)",min_value=0, max_value=400)

        st.markdown("<p style='color:#dc2626; font-weight:600;'>🧘 Activity & Stress</p>", unsafe_allow_html=True)
        physical_activity = st.number_input("Physical Activity (hours/week)",min_value=0.0, max_value=50.0, step=0.5)
        alcohol_intake = st.number_input("Alcohol Consumption (drinks/week)",min_value=0, max_value=30)
        stress_level = st.slider("Stress Level", min_value=0, max_value=10, value=5)
        mri_result = st.number_input("MRI Score (if available)",min_value=0.0, max_value=100.0)

    st.markdown("---")

    # Add risk indicators
    st.markdown("<h3 style='color:#000000; margin:0;'>⚠️ Risk Indicators</h3>", unsafe_allow_html=True)
    risk_factors = []
    if age > 65: risk_factors.append("Age > 65")
    if hypertension == "Yes": risk_factors.append("Hypertension")
    if heart_disease == "Yes": risk_factors.append("Heart Disease")
    if smoking_status == "Currently": risk_factors.append("Current Smoker")
    if bmi > 30: risk_factors.append("BMI > 30")
    if avg_glucose_level > 140: risk_factors.append("High Glucose")
    if blood_pressure > 140: risk_factors.append("High Blood Pressure")
    if cholesterol > 240: risk_factors.append("High Cholesterol")
    if family_history == "Yes": risk_factors.append("Family History")

    if risk_factors:
        st.markdown(
            f"<p style='color:#000000; margin:0;'><strong>Identified Risk Factors:</strong> {', '.join(risk_factors)}</p><br>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<p style='color:#000000; margin:0;'>No major risk factors identified</p>", unsafe_allow_html=True)

    # Prediction button
    if st.button("Predict Stroke Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            # Encode categorical variables to match training
            sex_val = 1 if sex == "Male" else 0
            hypertension_val = 1 if hypertension == "Yes" else 0
            heart_disease_val = 1 if heart_disease == "Yes" else 0

            # Encode work type
            work_type_map = {
                "Select": None,
                "Private": 0,
                "Self-employed": 1,
                "Govt_job": 2,
                "Children": 3,
                "Never_worked": 4
            }
            work_type_val = work_type_map[work_type]

            residence_type_val = 1 if residence_type == "Urban" else 0

            # Encode smoking status
            smoking_map = {
                "Select": None,
                "Never": 0,
                "Formerly": 1,
                "Currently": 2,
                "Unknown": 3
            }
            smoking_status_val = smoking_map[smoking_status]

            family_history_val = 1 if family_history == "Yes" else 0

            # Create input dataframe with exact feature order
            input_data = pd.DataFrame([[
                age,
                sex_val,
                hypertension_val,
                heart_disease_val,
                work_type_val,
                residence_type_val,
                avg_glucose_level,
                bmi,
                smoking_status_val,
                physical_activity,
                alcohol_intake,
                stress_level,
                blood_pressure,
                cholesterol,
                family_history_val,
                mri_result
            ]], columns=feature_names)

            # Scale the input
            input_scaled = scaler.transform(input_data)

            # Get prediction and probabilities
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

            # Extract probabilities
            stroke_prob = probabilities[1]

            # Convert to percentage
            stroke_risk_percentage = stroke_prob * 100

            # Calculate risk level for report generation
            if stroke_risk_percentage > 70:
                risk_level = "Very High"
            elif stroke_risk_percentage > 50:
                risk_level = "High"
            elif stroke_risk_percentage > 30:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

            # Save results to session state for report generation
            st.session_state["stroke_risk_percentage"] = stroke_risk_percentage
            st.session_state["risk_level"] = risk_level
            st.session_state["user_inputs"] = {
                "Age": age,
                "Sex": sex,
                "Hypertension": hypertension,
                "Heart Disease": heart_disease,
                "Family History": family_history,
                "Work Type": work_type,
                "Residence Type": residence_type,
                "Smoking Status": smoking_status,
                "Average Glucose Level": avg_glucose_level,
                "BMI": bmi,
                "Blood Pressure": blood_pressure,
                "Cholesterol": cholesterol,
                "Physical Activity (hrs/week)": physical_activity,
                "Alcohol Intake (drinks/week)": alcohol_intake,
                "Stress Level": stress_level,
                "MRI Score": mri_result
            }

            # Display results (reduced spacing)
            st.markdown("<div style='height:0.5px'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#000000; margin:0;'>Prediction Results</h3>", unsafe_allow_html=True)

            # Final result
            if stroke_risk_percentage > 50:
                st.markdown("<h3 style='margin:0 0 0.5rem 0; color:#dc2626;'>HIGH RISK OF STROKE</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="color:#000000;">
                    <h4>Immediate Recommendations:</h4>
                    <ul>
                        <li><strong>Consult a healthcare professional</strong></li>
                        <li>Consider comprehensive cardiovascular evaluation</li>
                        <li>Review and optimize current medications</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif stroke_risk_percentage > 30:
                st.markdown("<h3 style='margin:0 0 0.5rem 0; color:#f59e0b;'>MODERATE RISK OF STROKE</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style=\"color:#000000;\">\n                    <h4>Recommendations:</h4>
                    <ul>
                        <li>Schedule a check-up with your healthcare provider</li>
                        <li>Monitor blood pressure regularly</li>
                        <li>Consider lifestyle modifications</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<h3 style='margin:0 0 0.5rem 0; color:#16a34a;'>LOW RISK OF STROKE</h3>", unsafe_allow_html=True)



    # PDF download button
    if st.button(" Generate Report"):
        if "stroke_risk_percentage" not in st.session_state:
            st.error(" Please run the prediction first before generating a report.")
        else:
            pdf_path = generate_pdf(
                st.session_state["stroke_risk_percentage"],
                st.session_state["risk_level"],
                st.session_state["user_inputs"]
                )
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(" Download PDF", pdf_file, file_name="stroke_report.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown(
        """
        <p style="color:#000000; margin-top:8px;">
        ℹ️ This system is intended for health awareness and preliminary risk assessment. It does not provide a medical diagnosis. Always consult a qualified healthcare professional for medical advice and treatment decisions.
        </p>
        """,
        unsafe_allow_html=True
    )


# Router: decide which page to render
if st.session_state["page"] == "home":
    render_home_page()
elif st.session_state["page"] == "assess":
    render_assessment_page()
elif st.session_state["page"] == "fast":
    render_fast_info_page()
else:
    render_home_page()