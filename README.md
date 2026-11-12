# 🎓 Student360: Academic & Wellbeing Insight

### Data-Driven Research & Intelligent Decision Making

Student360 is a data-driven student analytics dashboard designed to explore relationships between academic performance, lifestyle habits, and student wellbeing.

The project demonstrates how data analytics and machine learning can support educational insights and informed decision-making.

---

## 🎯 Project Objectives

- Analyze student academic performance and wellbeing indicators.
- Explore relationships between study habits, sleep, stress, and attendance.
- Predict academic scores using Linear Regression.
- Classify students into illustrative risk categories.
- Generate personalized educational recommendations.
- Present insights through an interactive dashboard.

---

## ✨ Key Features

### 📊 Student Analytics Dashboard
- Academic score overview
- Sleep and stress indicators
- Interactive visualizations
- Student dataset exploration

### 🤖 Academic Score Prediction
- Linear Regression model
- Predict academic scores from lifestyle and attendance indicators

### 📈 Model Evaluation
- Mean Absolute Error (MAE)
- R² Score
- Actual vs predicted score visualization

### 🚦 Student Risk Classification
- Low Risk
- Medium Risk
- High Risk

### 💡 Personalized Recommendations
- Study habit suggestions
- Sleep improvement suggestions
- Stress management suggestions
- Attendance improvement suggestions

### 🔎 Research Insights
- Correlation analysis
- Academic and wellbeing relationships
- Correlation heatmap
- Data-driven findings

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Git & GitHub

---

## 📁 Project Structure

```text
Student360-Academic-and-Wellbeing-Insight/
├── assets/
├── dashboard/
│   └── dashboard.py
├── data/
│   └── sample_data.csv
├── models/
│   ├── academic_model.py
│   └── evaluate_model.py
├── app.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project

```bash
cd Student360-Academic-and-Wellbeing-Insight
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
streamlit run dashboard/dashboard.py
```

---

## 🧠 Machine Learning Methodology

### Algorithm
Linear Regression.

### Input Features
- Study Hours
- Sleep Hours
- Stress Level
- Attendance Percentage

### Target Variable
Academic Score.

### Evaluation
The model uses a 70/30 train-test split with:
- Mean Absolute Error
- R² Score

---

## 📊 Dataset

This project uses a small synthetic student dataset created for demonstration and educational purposes.

The dataset includes:
- Study hours
- Sleep hours
- Stress level
- Attendance percentage
- Academic score
- Wellbeing score

No real student records are used.

---

## ⚠️ Limitations

- Dataset contains only 10 synthetic student records.
- Model performance is not representative of real-world student populations.
- Risk classifications use predefined illustrative rules.
- Recommendations are rule-based educational suggestions.
- Correlation does not establish causation.
- Predictions should not be used for actual student assessment or intervention decisions.

---

## 🔮 Future Improvements

- Use a larger real-world dataset with appropriate consent and privacy safeguards.
- Compare multiple machine learning algorithms.
- Improve model evaluation using cross-validation.
- Develop a more advanced student recommendation system.
- Deploy the dashboard online.
- Add secure data management and privacy protections.

---

## 👩‍💻 Project Purpose

Student360 demonstrates how data analytics and machine learning can be applied to educational research and intelligent decision-making.

**Disclaimer:** This project is an educational prototype and does not replace professional academic or wellbeing assessment.