import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Rating Predictor",
    page_icon="🎬",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9ff, #eef1ff);
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Input section */
    .input-card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }

    /* Prediction result */
    .result-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    margin-top: 25px;
    color: white;
}

.result-card h2 {
    color: white;
    margin-bottom: 10px;
}

.rating {
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

    /* Button */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 14px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("model/movie_rating_ann.keras")
scaler = joblib.load("model/scaler.pkl")
mlb = joblib.load("model/genre_encoder.pkl")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🎬 Movie Rating Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict movie ratings using an Artificial Neural Network</div>',
    unsafe_allow_html=True
)


# ==========================================
# INPUT SECTION
# ==========================================

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.subheader("🎥 Enter Movie Details")

genre = st.multiselect(
    "Genre",
    mlb.classes_,
    placeholder="Select one or more genres"
)

year = st.number_input(
    "Release Year",
    min_value=1900,
    max_value=2026,
    value=2020
)

runtime = st.number_input(
    "Runtime (Minutes)",
    min_value=1,
    max_value=300,
    value=120
)

votes = st.number_input(
    "Votes",
    min_value=0,
    max_value=2000000,
    value=100000
)

revenue = st.number_input(
    "Revenue (Millions)",
    min_value=0.0,
    max_value=2000.0,
    value=50.0
)

metascore = st.number_input(
    "Metascore",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🎯 Predict Movie Rating"):

    if len(genre) == 0:

        st.warning("⚠️ Please select at least one genre.")

    else:

        # Encode genre
        genre_encoded = mlb.transform([genre])

        # Numerical features
        input_data = pd.DataFrame(
            [[
                year,
                runtime,
                votes,
                revenue,
                metascore
            ]],
            columns=[
                "Year",
                "Runtime (Minutes)",
                "Votes",
                "Revenue (Millions)",
                "Metascore"
            ]
        )

        # Genre features
        genre_data = pd.DataFrame(
            genre_encoded,
            columns=mlb.classes_
        )

        # Combine features
        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                genre_data.reset_index(drop=True)
            ],
            axis=1
        )

        # Scale
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(
            input_scaled,
            verbose=0
        )[0][0]

        # Result
        st.markdown(
            f"""
            <div class="result-card">
                <h2>🎬 Predicted Movie Rating</h2>
                <div class="rating">⭐ {prediction:.2f} / 10</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# MODEL INFORMATION
# ==========================================

st.markdown("---")

st.subheader("🤖 About the Model")

st.write(
    """
    This application uses an Artificial Neural Network (ANN)
    trained on movie-related features such as genre, release year,
    runtime, votes, revenue and metascore to predict movie ratings.
    """
)

st.info(
    "Model Type: Artificial Neural Network (ANN) | "
    "Task: Regression | Target: Movie Rating"
)


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">Built with Python, TensorFlow & Streamlit 🎬</div>',
    unsafe_allow_html=True
)