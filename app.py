import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Scholarship Eligibility Predictor Pro",
    page_icon="🎓",
    layout="wide",
)


# 2. MODEL & DATASET SETUP
@st.cache_resource
def load_trained_model():
    f_names = [
        "Parent Income",
        "Entry Test Rank",
        "College",
        "10th Marks",
        "12th Marks",
        "12th Division",
        "Province",
        "Nationality",
    ]

    np.random.seed(42)
    sample_size = 500

    income = np.random.randint(20000, 150000, sample_size)
    rank = np.random.randint(1, 500, sample_size)
    college = np.random.randint(0, 10, sample_size)
    m10 = np.random.randint(50, 100, sample_size)
    m12 = np.random.randint(50, 100, sample_size)
    div = np.random.randint(0, 3, sample_size)
    prov = np.random.randint(0, 5, sample_size)
    nat = np.random.randint(0, 2, sample_size)

    eligible = []
    for i in range(sample_size):
        score = 0
        if income[i] <= 70000:
            score += 35
        if m12[i] >= 75:
            score += 30
        if m10[i] >= 75:
            score += 15
        if rank[i] <= 150:
            score += 20
        eligible.append(1 if score >= 50 else 0)

    df_sample = pd.DataFrame({
        "Parent Income": income,
        "Entry Test Rank": rank,
        "College": college,
        "10th Marks": m10,
        "12th Marks": m12,
        "12th Division": div,
        "Province": prov,
        "Nationality": nat,
        "Eligible": eligible,
    })

    X = df_sample[f_names]
    y = df_sample["Eligible"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, f_names


model, scaler, feature_names = load_trained_model()

# 3. SIDEBAR INPUTS
st.sidebar.header("User Inputs")
division_opt = st.sidebar.selectbox("12th Division", ["1st", "2nd", "3rd"])
entry_rank = st.sidebar.number_input(
    "Entry Test Rank/Score", min_value=1, max_value=1000, value=50
)

universities_list = [
    "The Islamia University of Bahawalpur (IUB)",
    "University of Agriculture Faisalabad (UAF)",
    "University of the Punjab (PU)",
    "National University of Sciences & Technology (NUST)",
    "COMSATS University Islamabad",
    "Quaid-i-Azam University (QAU)",
    "UET Lahore",
    "Bahauddin Zakariya University (BZU)",
    "GCU Lahore",
    "Other",
]
college_opt = st.sidebar.selectbox("College / University", universities_list)
nationality_opt = st.sidebar.selectbox(
    "Nationality", ["Pakistani", "Foreigner"]
)
province_opt = st.sidebar.selectbox(
    "Province",
    ["Punjab", "Sindh", "KPK", "Balochistan", "Gilgit-Baltistan / AJK"],
)
income_opt = st.sidebar.number_input(
    "Parent Monthly Income (Rs)", min_value=0, value=55000, step=5000
)
marks_10th = st.sidebar.number_input(
    "10th Marks (%)", min_value=0, max_value=100, value=93
)
marks_12th = st.sidebar.number_input(
    "12th Marks (%)", min_value=0, max_value=100, value=91
)

predict_btn = st.sidebar.button("Predict Eligibility", type="primary")

# MAPPINGS
division_map = {"1st": 0, "2nd": 1, "3rd": 2}
college_map = {uni: i for i, uni in enumerate(universities_list)}
nationality_map = {"Pakistani": 0, "Foreigner": 1}
province_map = {
    "Punjab": 0,
    "Sindh": 1,
    "KPK": 2,
    "Balochistan": 3,
    "Gilgit-Baltistan / AJK": 4,
}

# 4. MAIN DASHBOARD DISPLAY
st.title("🎓 Scholarship Eligibility Predictor Pro")

input_values = [
    income_opt,
    entry_rank,
    college_map[college_opt],
    marks_10th,
    marks_12th,
    division_map[division_opt],
    province_map[province_opt],
    nationality_map[nationality_opt],
]
input_df = pd.DataFrame([input_values], columns=feature_names)

scaled_input = scaler.transform(input_df)
prob = float(model.predict_proba(scaled_input)[0][1] * 100)

status = (
    "Eligible (Full/Partial Aid)" if prob >= 50 else "Not Eligible / Low Chance"
)
bg_color = "#1b4332" if prob >= 50 else "#4a0e17"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Prediction Results")
    st.markdown(
        f"<div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 15px;'><h2 style='color: white; margin: 0;'>Match: {prob:.2f}% Probability</h2></div>",
        unsafe_allow_html=True,
    )
    st.caption("Status Category")
    st.markdown(f"### **{status}**")

with col2:
    st.subheader("📈 Decision Drivers (Feature Importances)")
    importances = model.feature_importances_ * 100
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=importance_df["Importance"].apply(lambda x: f"{x:.1f}%"),
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        margin=dict(l=0, r=25, t=10, b=0),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

st.divider()

col_act, col_exp = st.columns([1, 1])

with col_act:
    st.subheader("💡 Actionable Recommendations")
    st.info(
        "• Maintain high academic scores in 12th Marks.\n• Parent Monthly Income and Entry Test score are key evaluation criteria."
    )

with col_exp:
    st.subheader("📄 Export Assessment")

    report_df = pd.DataFrame([{
        "University": college_opt,
        "Nationality": nationality_opt,
        "Province": province_opt,
        "Parent Income (Rs)": income_opt,
        "10th Marks (%)": marks_10th,
        "12th Marks (%)": marks_12th,
        "12th Division": division_opt,
        "Entry Test Rank": entry_rank,
        "Match Probability (%)": round(prob, 2),
        "Status": status,
    }])

    csv_data = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv_data,
        file_name="scholarship_report.csv",
        mime="text/csv",
        type="primary",
    )
