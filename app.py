import joblib  # Agar model save karna ho, ya direct train karke use kar sakte hain
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(
    page_title="Scholarship Eligibility Predictor", page_icon="🎓", layout="centered"
)

st.title("🎓 Student Scholarship Eligibility Predictor")
st.write("Student ki details enter karein taake pata chal sake ke wo eligible hai ya nahi.")


# Data Load aur Model Train karne ka function (Caching ke sath)
@st.cache_resource
def load_and_train_model():
  df = pd.read_csv("updated_scholarship_dataset.csv")

  label_encoders = {}
  categorical_cols = ["College", "Nationality", "Province", "12th Division"]

  for col in categorical_cols:
    if col in df.columns:
      le = LabelEncoder()
      df[col] = le.fit_transform(df[col].astype(str))
      label_encoders[col] = le

  X = df.drop(columns=["Scholarship_Status", "Year"], errors="ignore")
  y = df["Scholarship_Status"]

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X, y)

  return model, label_encoders, X.columns


model, label_encoders, feature_cols = load_and_train_model()

# User Inputs UI
st.subheader("📝 Enter Student Details:")

col1, col2 = st.columns(2)

with col1:
  marks_10th = st.number_input("10th Marks", min_value=0, max_value=100, value=85)
  marks_12th = st.number_input("12th Marks", min_value=0, max_value=100, value=80)
  div_12th = st.selectbox(
    "12th Division", [1, 2, 3, 4, 5, 6]
  )  # Jaise dataset mein values thin
  aiee_rank = st.number_input("Entry Test Rank/Score", min_value=1, max_value=5000, value=150)

with col2:
  # Dummy options for dropdowns based on your dataset
  college = st.selectbox(
    "College",
    [
      "IBA Karachi",
      "NUST Islamabad",
      "NED University Karachi",
      "UET Peshawar",
      "Islamia University Bahawalpur",
      "COMSATS Islamabad",
    ],
  )
  nationality = st.selectbox("Nationality", ["Pakistani", "Other"])
  province = st.selectbox(
    "Province",
    ["Punjab", "Sindh", "KPK", "Balochistan", "Gilgit-Baltistan", "Islamabad"],
  )
  parent_income = st.number_input(
    "Parent Monthly Income (Rs)", min_value=10000, max_value=500000, value=45000
  )

# Prediction Button
if st.button("Predict Scholarship Eligibility 🚀"):
  # Input Data ko DataFrame mein convert karna
  input_data = pd.DataFrame({
      "10th Marks": [marks_10th],
      "12th Marks": [marks_12th],
      "12th Division": [div_12th],
      "AIEEE Rank": [aiee_rank],
      "College": [
          label_encoders["College"].transform([college])[0]
          if college in label_encoders["College"].classes_
          else 0
      ],
      "Nationality": [
          label_encoders["Nationality"].transform([nationality])[0]
          if nationality in label_encoders["Nationality"].classes_
          else 0
      ],
      "Province": [
          label_encoders["Province"].transform([province])[0]
          if province in label_encoders["Province"].classes_
          else 0
      ],
      "Parent_Income": [parent_income],
  })

  # Columns order match karna
  input_data = input_data[feature_cols]

  # Prediction
  prediction = model.predict(input_data)

  # Result Display
  st.markdown("---")
  if prediction[0] == 1:
    st.success(
        "🎉 **Mubarak ho! Student Scholarship ke liye ELIGIBLE hai.**"
    )
  else:
    st.error(
        "❌ **Afsoos! Student Scholarship ke liye Eligible nahi hai.** (Income"
        " zyada ya marks kam hain)"
    )