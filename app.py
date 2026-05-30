import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Random Forest Classifier - Titanic",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Random Forest Classification on Titanic Dataset")

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

st.markdown("""
### Workflow Included
- Dataset Loading
- Data Cleaning
- Missing Value Handling
- Feature Encoding
- Save Preprocessed Dataset
- EDA
- Hyperparameter Tuning
- Model Training
- Prediction
- Evaluation
""")

# DATASET
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

st.header("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# METRICS
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", int(df.isnull().sum().sum()))

# CLEANING
st.header("🧹 Data Cleaning")

df = df[["Survived","Pclass","Sex","Age","Fare","SibSp","Parch"]]

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"] = df["Sex"].map({"male":0,"female":1})

st.success("Data Cleaning Completed")

# SAVE DATASET
st.header("💾 Save / Download Preprocessed Dataset")

os.makedirs("data", exist_ok=True)

if st.button("Save Dataset Locally"):
    df.to_csv("data/preprocessed_titanic_random_forest.csv", index=False)
    st.success("Saved to data/preprocessed_titanic_random_forest.csv")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Preprocessed Dataset",
    data=csv,
    file_name="preprocessed_titanic_random_forest.csv",
    mime="text/csv"
)

# EDA
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="Age", color="Survived",
                        title="Age Distribution by Survival")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x="Pclass", color="Survived",
                        barmode="group",
                        title="Passenger Class vs Survival")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    df,
    x="Fare",
    y="Age",
    color="Survived",
    title="Fare vs Age"
)
st.plotly_chart(fig3, use_container_width=True)

# FEATURES
X = df.drop("Survived", axis=1)
y = df["Survived"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TUNING
st.header("⚙️ Hyperparameter Tuning")

param_grid = {
    "n_estimators":[50,100],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

with st.spinner("Finding Best Parameters..."):
    grid.fit(X_train, y_train)

best_model = grid.best_estimator_

st.success("Best Parameters Found")
st.write(grid.best_params_)

# TRAIN
best_model.fit(X_train, y_train)

# PREDICT
y_pred = best_model.predict(X_test)

st.header("📉 Model Evaluation")

acc = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{acc:.4f}")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0","Actual 1"],
    columns=["Predicted 0","Predicted 1"]
)

st.dataframe(cm_df, use_container_width=True)

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# FEATURE IMPORTANCE
st.header("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig4 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(fig4, use_container_width=True)

# USER PREDICTION
st.header("🎯 Predict Survival")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["Male","Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
parch = st.slider("Parents/Children", 0, 6, 0)

if st.button("Predict Survival"):
    sex_value = 1 if sex == "Female" else 0

    sample = pd.DataFrame([[
        pclass, sex_value, age, fare, sibsp, parch
    ]], columns=X.columns)

    pred = best_model.predict(sample)[0]

    if pred == 1:
        st.success("Passenger Likely Survived ✅")
    else:
        st.error("Passenger Likely Did Not Survive ❌")

st.header("💡 Insights")

st.info("""
• Female passengers had higher survival rates.

• First-class passengers survived more often.

• Fare and Age influence survival probability.
        
• Random Forest reduces overfitting using multiple trees.

• Hyperparameter tuning improves performance.
""")
