import os
import pickle

import pandas as pd
import streamlit as st

# Provide a lightweight fallback for `option_menu` so the app
# still works on deployments where `streamlit-option-menu` may
# not have been installed for any reason. The fallback simply
# returns a `st.selectbox` value and matches the original
# function signature used in this app.
try:
    from streamlit_option_menu import option_menu
except Exception:
    def option_menu(title, options, icons=None, menu_icon=None, default_index=0):
        return st.selectbox(title, options, index=default_index)


st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🧑‍⚕️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(9, 94, 84, 0.12), rgba(240, 245, 241, 0.92));
        border: 1px solid rgba(9, 94, 84, 0.12);
        box-shadow: 0 18px 50px rgba(8, 35, 33, 0.08);
    }

    .hero h1 {
        margin-bottom: 0.35rem;
        color: #0c3b36;
    }

    .hero p {
        margin-top: 0;
        color: #35524f;
        font-size: 1.04rem;
        line-height: 1.7;
    }

    .feature-card {
        padding: 1.15rem 1.1rem;
        border-radius: 20px;
        background: white;
        border: 1px solid rgba(12, 59, 54, 0.08);
        box-shadow: 0 10px 30px rgba(8, 35, 33, 0.05);
        min-height: 155px;
    }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0c3b36;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: #4c6864;
        line-height: 1.6;
        margin: 0;
    }

    .section-label {
        color: #0f6b63;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKING_DIR, "saved_models")


@st.cache_resource
def load_models():
    diabetes_model = heart_disease_model = parkinsons_model = None

    def _safe_load(path):
        if not os.path.exists(path):
            return None, f"missing file: {path}"
        try:
            with open(path, "rb") as file:
                return pickle.load(file), None
        except Exception as e:
            return None, str(e)

    diabetes_path = os.path.join(MODEL_DIR, "diabetes_model.sav")
    heart_path = os.path.join(MODEL_DIR, "heart_disease_model.sav")
    parkinsons_path = os.path.join(MODEL_DIR, "parkinsons_model.sav")

    diabetes_model, err1 = _safe_load(diabetes_path)
    heart_disease_model, err2 = _safe_load(heart_path)
    parkinsons_model, err3 = _safe_load(parkinsons_path)

    errors = [e for e in (err1, err2, err3) if e]
    if errors:
        # Surface a clear message in the app logs/UI rather than letting Streamlit crash with a red traceback.
        st.error("Model load issues:\n" + "\n".join(errors))

    return diabetes_model, heart_disease_model, parkinsons_model


def predict_label(model, values):
    if model is None:
        return None
    try:
        if hasattr(model, "feature_names_in_"):
            input_data = pd.DataFrame([values], columns=model.feature_names_in_)
        else:
            input_data = [values]
        return model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


diabetes_model, heart_disease_model, parkinsons_model = load_models()


with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        [
            "Home",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinson's Prediction",
        ],
        menu_icon="hospital-fill",
        icons=["house", "activity", "heart", "person"],
        default_index=0,
    )

    st.caption(
        "These models use structured medical measurements from the datasets, not free-text symptom chat."
    )


if selected == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">AI-assisted health screening</div>
            <h1>Care begins with early awareness.</h1>
            <p>
                This app brings three trained disease prediction models together in one calm,
                easy-to-use healthcare dashboard. Enter the required medical measurements,
                review the result instantly, and use it as a quick screening aid.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Diabetes screening</div>
                <p class="feature-text">Check structured clinical measurements such as glucose, BMI, insulin, age, and pedigree function to get a fast prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Heart health insight</div>
                <p class="feature-text">Review key heart indicators including blood pressure, cholesterol, chest pain type, and ECG-based measurements in one place.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Parkinson's assessment</div>
                <p class="feature-text">Use voice-derived biomedical features to estimate the likelihood of Parkinson's disease from the trained model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        st.info(
            "Choose a disease from the sidebar to begin. The models are already trained and loaded from the saved_models folder."
        )

    with right:
        st.success(
            "Designed for quick screening, not medical diagnosis. If a result concerns you, speak with a qualified clinician."
        )


if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    st.caption("Enter the same structured measurements used during model training.")

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Number of Pregnancies", value=0.0, step=1.0)

        with col2:
            glucose = st.number_input("Glucose Level", value=0.0)

        with col3:
            blood_pressure = st.number_input("Blood Pressure value", value=0.0)

        with col1:
            skin_thickness = st.number_input("Skin Thickness value", value=0.0)

        with col2:
            insulin = st.number_input("Insulin Level", value=0.0)

        with col3:
            bmi = st.number_input("BMI value", value=0.0)

        with col1:
            dpf = st.number_input("Diabetes Pedigree Function value", value=0.0)

        with col2:
            age = st.number_input("Age of the Person", value=0.0, step=1.0)

        submitted = st.form_submit_button("Diabetes Test Result")

    if submitted:
        user_input = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
        ]
        prediction = predict_label(diabetes_model, user_input)

        if prediction is None:
            st.error("Diabetes model unavailable or prediction failed. Check logs.")
        else:
            if prediction == 1:
                st.success("The person is diabetic.")
            else:
                st.success("The person is not diabetic.")


if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    st.caption("Use the encoded feature values from the heart dataset.")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", value=0.0, step=1.0)

        with col2:
            sex = st.number_input("Sex", value=0.0, step=1.0)

        with col3:
            cp = st.number_input("Chest Pain types", value=0.0, step=1.0)

        with col1:
            trestbps = st.number_input("Resting Blood Pressure", value=0.0)

        with col2:
            chol = st.number_input("Serum Cholestoral in mg/dl", value=0.0)

        with col3:
            fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl", value=0.0, step=1.0)

        with col1:
            restecg = st.number_input("Resting Electrocardiographic results", value=0.0, step=1.0)

        with col2:
            thalach = st.number_input("Maximum Heart Rate achieved", value=0.0)

        with col3:
            exang = st.number_input("Exercise Induced Angina", value=0.0, step=1.0)

        with col1:
            oldpeak = st.number_input("ST depression induced by exercise", value=0.0)

        with col2:
            slope = st.number_input("Slope of the peak exercise ST segment", value=0.0, step=1.0)

        with col3:
            ca = st.number_input("Major vessels colored by fluoroscopy", value=0.0, step=1.0)

        with col1:
            thal = st.number_input(
                "thal: 0 = normal; 1 = fixed defect; 2 = reversible defect",
                value=0.0,
                step=1.0,
            )

        submitted = st.form_submit_button("Heart Disease Test Result")

    if submitted:
        user_input = [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]
        prediction = predict_label(heart_disease_model, user_input)

        if prediction is None:
            st.error("Heart disease model unavailable or prediction failed. Check logs.")
        else:
            if prediction == 1:
                st.success("The person is having heart disease.")
            else:
                st.success("The person does not have any heart disease.")


if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    st.caption("Use the numeric voice-measurement features from the Parkinson's dataset.")

    with st.form("parkinsons_form"):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", value=0.0)

        with col2:
            fhi = st.number_input("MDVP:Fhi(Hz)", value=0.0)

        with col3:
            flo = st.number_input("MDVP:Flo(Hz)", value=0.0)

        with col4:
            jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.0)

        with col5:
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.0)

        with col1:
            rap = st.number_input("MDVP:RAP", value=0.0)

        with col2:
            ppq = st.number_input("MDVP:PPQ", value=0.0)

        with col3:
            ddp = st.number_input("Jitter:DDP", value=0.0)

        with col4:
            shimmer = st.number_input("MDVP:Shimmer", value=0.0)

        with col5:
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.0)

        with col1:
            apq3 = st.number_input("Shimmer:APQ3", value=0.0)

        with col2:
            apq5 = st.number_input("Shimmer:APQ5", value=0.0)

        with col3:
            apq = st.number_input("MDVP:APQ", value=0.0)

        with col4:
            dda = st.number_input("Shimmer:DDA", value=0.0)

        with col5:
            nhr = st.number_input("NHR", value=0.0)

        with col1:
            hnr = st.number_input("HNR", value=0.0)

        with col2:
            rpde = st.number_input("RPDE", value=0.0)

        with col3:
            dfa = st.number_input("DFA", value=0.0)

        with col4:
            spread1 = st.number_input("spread1", value=0.0)

        with col5:
            spread2 = st.number_input("spread2", value=0.0)

        with col1:
            d2 = st.number_input("D2", value=0.0)

        with col2:
            ppe = st.number_input("PPE", value=0.0)

        submitted = st.form_submit_button("Parkinson's Test Result")

    if submitted:
        user_input = [
            fo,
            fhi,
            flo,
            jitter_percent,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer,
            shimmer_db,
            apq3,
            apq5,
            apq,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe,
        ]
        prediction = predict_label(parkinsons_model, user_input)

        if prediction is None:
            st.error("Parkinson's model unavailable or prediction failed. Check logs.")
        else:
            if prediction == 1:
                st.success("The person has Parkinson's disease.")
            else:
                st.success("The person does not have Parkinson's disease.")
