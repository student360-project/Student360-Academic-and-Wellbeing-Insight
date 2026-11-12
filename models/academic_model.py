
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "data" / "sample_data.csv")

# Features used for prediction
features = [
    "Study_Hours",
    "Sleep_Hours",
    "Stress_Level",
    "Attendance_Percentage"
]

target = "Academic_Score"

# Prepare training data
X = df[features]
y = df[target]

# Train machine learning model
model = LinearRegression()
model.fit(X, y)


def predict_academic_score(
    study_hours,
    sleep_hours,
    stress_level,
    attendance
):
    new_student = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Sleep_Hours": sleep_hours,
        "Stress_Level": stress_level,
        "Attendance_Percentage": attendance
    }])

    prediction = model.predict(new_student)[0]

    # Keep predicted score within 0–100
    prediction = max(0, min(100, prediction))

    return round(prediction, 2)
