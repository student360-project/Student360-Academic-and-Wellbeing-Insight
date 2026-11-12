
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.academic_model import predict_academic_score


# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Student360 Dashboard",
    page_icon="🎓",
    layout="wide"
)

# =========================
# CUSTOM DESIGN
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff, #f8faff);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #142957, #081a3b);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #17336b;
}

[data-testid="stMetric"] {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #dce7ff;
    box-shadow: 0 4px 15px rgba(50,80,150,0.08);
}

[data-testid="stMetricLabel"] {
    color: #53698f;
}

[data-testid="stMetricValue"] {
    color: #1749a5;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "sample_data.csv")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🎓 Student360")
    st.caption("Academic & Wellbeing Insight")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 Dashboard")
    st.markdown("📊 Insights")
    st.markdown("🤖 Prediction")
    st.markdown("ℹ️ About")

    st.divider()
    st.caption("Smarter Insights. Brighter Future.")

# =========================
# HEADER
# =========================
st.title("📊 Student360 Dashboard")

st.markdown(
    "Explore academic performance and wellbeing insights "
    "to support intelligent decision-making."
)

st.divider()

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric(
    "🎓 Average Academic Score",
    f"{df['Academic_Score'].mean():.1f}/100"
)

col2.metric(
    "🛌 Average Sleep Hours",
    f"{df['Sleep_Hours'].mean():.1f} hrs"
)

col3.metric(
    "💚 Average Stress Level",
    f"{df['Stress_Level'].mean():.1f}/10"
)

st.divider()

# =========================
# CHART
# =========================
st.subheader("🌙 Sleep, Stress and Academic Performance")

fig = px.scatter(
    df,
    x="Sleep_Hours",
    y="Stress_Level",
    size="Study_Hours",
    color="Academic_Score",
    hover_name="Student_ID",
    color_continuous_scale="Blues",
    title="Relationship Between Sleep, Stress and Academic Scores"
)

fig.update_layout(
    template="plotly_white",
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DATASET
# =========================
st.subheader("📚 Student Dataset")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =========================
# FOOTER
# =========================
st.success(
    "Student360 Dashboard Successfully Running! "
    "Data-driven insights for a better student experience."
)

# =========================
# INTELLIGENT PREDICTION
# =========================

st.divider()

st.header("🤖 Academic Performance Prediction")

st.write(
    "Estimate academic performance using student "
    "study habits and wellbeing indicators."
)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Study Hours per Day",
        0.0, 12.0, 5.0, 0.5
    )

    sleep_hours = st.slider(
        "Sleep Hours per Day",
        0.0, 12.0, 7.0, 0.5
    )

with col2:
    stress_level = st.slider(
        "Stress Level",
        1, 10, 5
    )

    attendance = st.slider(
        "Attendance Percentage",
        0, 100, 85
    )

if st.button("Predict Academic Score", type="primary"):

    predicted_score = predict_academic_score(
        study_hours,
        sleep_hours,
        stress_level,
        attendance
    )

    st.metric(
        "Predicted Academic Score",
        f"{predicted_score}/100"
    )

    if predicted_score >= 80:
        st.success("High predicted academic performance.")
    elif predicted_score >= 60:
        st.info("Moderate predicted academic performance.")
    else:
        st.warning(
            "Lower predicted score. Consider reviewing "
            "study habits and student support needs."
        )

# =========================
# MODEL EVALUATION
# =========================

st.divider()

st.header("📈 Model Performance Evaluation")

st.write(
    "Compare actual academic scores with predicted scores "
    "using a test dataset."
)

# Prepare data
features = [
    "Study_Hours",
    "Sleep_Hours",
    "Stress_Level",
    "Attendance_Percentage"
]

X = df[features]
y = df["Academic_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Train evaluation model
evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

# Predict test scores
y_pred = evaluation_model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# Comparison dataframe
comparison = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred
})

# Actual vs predicted chart
fig_eval = px.scatter(
    comparison,
    x="Actual Score",
    y="Predicted Score",
    title="Actual vs predicted academic scores",
    color_discrete_sequence=["#2563eb"]
)

# Reference line
fig_eval.add_shape(
    type="line",
    x0=0, y0=0,
    x1=100, y1=100,
    line=dict(color="red", dash="dash")
)

fig_eval.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Actual Academic Score",
    yaxis_title="Predicted Academic Score"
)

st.plotly_chart(fig_eval, use_container_width=True)

st.caption(
    "Evaluation uses a 70/30 train-test split. "
    "Results are illustrative because the dataset is synthetic and small."
)


# =========================
# STUDENT RISK CLASSIFICATION
# =========================

st.divider()
st.header("🚦 Student Risk Classification")

st.write(
    "Identify students who may need additional academic "
    "or wellbeing support."
)

# Create risk classification rules
def classify_student_risk(row):

    if (
        row["Academic_Score"] < 60
        or row["Wellbeing_Score"] < 45
        or row["Stress_Level"] >= 8
    ):
        return "High Risk"

    elif (
        row["Academic_Score"] < 75
        or row["Wellbeing_Score"] < 65
        or row["Stress_Level"] >= 6
    ):
        return "Medium Risk"

    else:
        return "Low Risk"

# Apply classification to dataset
risk_df = df.copy()

risk_df["Risk_Level"] = risk_df.apply(
    classify_student_risk,
    axis=1
)

# Display risk summary
risk_counts = risk_df["Risk_Level"].value_counts()

col1, col2, col3 = st.columns(3)

col1.metric(
    "🔴 High Risk",
    risk_counts.get("High Risk", 0)
)

col2.metric(
    "🟠 Medium Risk",
    risk_counts.get("Medium Risk", 0)
)

col3.metric(
    "🟢 Low Risk",
    risk_counts.get("Low Risk", 0)
)

# Display student risk table
st.subheader("Student Risk Overview")

st.dataframe(
    risk_df[
        [
            "Student_ID",
            "Academic_Score",
            "Wellbeing_Score",
            "Stress_Level",
            "Risk_Level"
        ]
    ],
    use_container_width=True
)

# Risk distribution chart
risk_chart = px.pie(
    risk_df,
    names="Risk_Level",
    title="Student Risk Distribution",
    color="Risk_Level",
    color_discrete_map={
        "High Risk": "#ef4444",
        "Medium Risk": "#f59e0b",
        "Low Risk": "#22c55e"
    }
)

st.plotly_chart(
    risk_chart,
    use_container_width=True
)


# =========================
# PERSONALIZED RECOMMENDATIONS
# =========================

st.divider()
st.header("💡 Personalized Student Recommendations")

st.write(
    "Generate practical recommendations based on student "
    "academic performance and wellbeing indicators."
)

# Select a student
selected_student = st.selectbox(
    "Select Student ID",
    df["Student_ID"].tolist()
)

# Get selected student information
student = df[df["Student_ID"] == selected_student].iloc[0]

# Generate recommendations
recommendations = []

if student["Study_Hours"] < 4:
    recommendations.append(
        "📚 Consider increasing study time gradually "
        "and following a consistent study schedule."
    )

if student["Sleep_Hours"] < 7:
    recommendations.append(
        "😴 Consider maintaining a regular sleep schedule "
        "and aiming for sufficient rest."
    )

if student["Stress_Level"] >= 7:
    recommendations.append(
        "🧘 Consider relaxation activities, regular breaks, "
        "and seeking support when needed."
    )

if student["Attendance_Percentage"] < 80:
    recommendations.append(
        "🏫 Improving class attendance may help "
        "strengthen learning consistency."
    )

if student["Academic_Score"] < 60:
    recommendations.append(
        "🎯 Consider reviewing difficult topics and "
        "seeking academic support."
    )

if student["Wellbeing_Score"] < 60:
    recommendations.append(
        "💙 Consider prioritizing wellbeing and "
        "reaching out to a trusted support person."
    )

if not recommendations:
    recommendations.append(
        "🌟 Your indicators look encouraging. "
        "Continue maintaining balanced study and wellbeing habits."
    )

# Display selected student information
st.subheader(f"Student Profile: {selected_student}")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Academic Score",
    student["Academic_Score"]
)

col2.metric(
    "Wellbeing Score",
    student["Wellbeing_Score"]
)

col3.metric(
    "Stress Level",
    student["Stress_Level"]
)

# Display recommendations
st.subheader("Recommended Actions")

for recommendation in recommendations:
    st.info(recommendation)

st.caption(
    "Recommendations are educational suggestions based on "
    "synthetic data and simple predefined rules."
)


# =========================
# RESEARCH INSIGHTS & FINDINGS
# =========================

st.divider()
st.header("🔎 Research Insights & Key Findings")

st.write(
    "Explore relationships between academic performance "
    "and student lifestyle indicators."
)

# Calculate correlations
study_corr = df["Study_Hours"].corr(df["Academic_Score"])
sleep_corr = df["Sleep_Hours"].corr(df["Academic_Score"])
stress_corr = df["Stress_Level"].corr(df["Academic_Score"])
attendance_corr = df["Attendance_Percentage"].corr(
    df["Academic_Score"]
)

# Display correlation metrics
col1, col2 = st.columns(2)

col1.metric(
    "Study Hours Correlation",
    f"{study_corr:.2f}"
)

col2.metric(
    "Sleep Hours Correlation",
    f"{sleep_corr:.2f}"
)

col3, col4 = st.columns(2)

col3.metric(
    "Stress Correlation",
    f"{stress_corr:.2f}"
)

col4.metric(
    "Attendance Correlation",
    f"{attendance_corr:.2f}"
)

# Correlation interpretation
st.subheader("Key Findings")

if study_corr > 0:
    st.write(
        "📚 Study hours show a positive relationship "
        "with academic scores in this dataset."
    )
else:
    st.write(
        "📚 Study hours show a negative relationship "
        "with academic scores in this dataset."
    )

if sleep_corr > 0:
    st.write(
        "😴 Sleep hours show a positive relationship "
        "with academic scores in this dataset."
    )
else:
    st.write(
        "😴 Sleep hours show a negative relationship "
        "with academic scores in this dataset."
    )

if stress_corr < 0:
    st.write(
        "🧠 Higher stress levels are associated with "
        "lower academic scores in this dataset."
    )
else:
    st.write(
        "🧠 Stress levels show a positive relationship "
        "with academic scores in this dataset."
    )

if attendance_corr > 0:
    st.write(
        "🏫 Attendance shows a positive relationship "
        "with academic scores in this dataset."
    )
else:
    st.write(
        "🏫 Attendance shows a negative relationship "
        "with academic scores in this dataset."
    )

# Correlation heatmap
st.subheader("Correlation Heatmap")

correlation_data = df[
    [
        "Study_Hours",
        "Sleep_Hours",
        "Stress_Level",
        "Attendance_Percentage",
        "Academic_Score",
        "Wellbeing_Score"
    ]
].corr()

heatmap = px.imshow(
    correlation_data,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Student Lifestyle, Academic & Wellbeing Correlations"
)

st.plotly_chart(
    heatmap,
    use_container_width=True
)

st.caption(
    "Correlations describe associations, not causation. "
    "Findings are illustrative because the dataset is synthetic "
    "and contains only 10 students."
)
