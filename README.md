# 🎓 ML-Based Scholarship Eligibility Predictor

A Machine Learning-powered web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict student scholarship eligibility based on academic marks, entrance test performance, and family income.

---

## 🚀 Live Demo
You can access the live application here:  
👉 **[Scholarship Eligibility Predictor App](https://scholarship-eligibility-predictor.streamlit.app/)** *(Replace with your actual Streamlit link)*

---

## 📌 Project Overview
Scholarship allocation often involves evaluating multiple criteria simultaneously. This application leverages a **Random Forest Classifier** to assess candidates based on localized Pakistani educational metrics and financial parameters.

### Key Features Evaluated:
* **Academic Marks:** 10th Marks (%) & 12th Marks (%)
* **Academic Tier:** 12th Division (BISE Grading)
* **Standardized Testing:** Entry Test Rank/Score (ECAT, MDCAT, NAT, NTS, etc.)
* **Financial Background:** Parent Monthly Income (PKR)
* **Demographics:** College, Province, and Nationality

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, LabelEncoder)
* **Frontend / UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Version Control & Hosting:** GitHub, Streamlit Community Cloud

---

## 📂 Project Structure
```text
├── app.py                   # Main Streamlit application script & Model pipeline
├── updated_scholarship_dataset.csv # Dataset used for model training
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
