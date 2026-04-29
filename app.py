import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.markdown("""
# 💼 Salary Prediction System  
### 🚀 Machine Learning Web App  

Predict salary based on experience using multiple ML models.
""")

# ------------------ LOAD DATA ------------------
dataset = pd.read_csv("Salary_Dataset.csv")

# ------------------ DATA PREVIEW ------------------
st.subheader("📊 Dataset Preview")
st.write(dataset.head())

# ------------------ PREPROCESSING ------------------
dataset = pd.get_dummies(dataset, drop_first=True)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# ------------------ TEST SIZE ------------------
st.subheader("⚙️ Data Split (Train/Test)")
test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42
)

st.write(f"Training Size: {len(X_train)}")
st.write(f"Testing Size: {len(X_test)}")

# ------------------ VISUALIZATION ------------------
st.subheader("📈 Data Visualization")

import seaborn as sns

# ------------------ MAIN GRAPH  ------------------
st.markdown("### 📉 Salary vs Experience (Model Visualization)")

model = LinearRegression()
model.fit(X_train, Y_train)

sorted_X = np.sort(X_train, axis=0)
line = model.predict(sorted_X)

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(X_train, Y_train, color='blue', label='Training Data')
ax.scatter(X_test, Y_test, color='green', label='Testing Data')
ax.plot(sorted_X, line, color='red', linewidth=2, label='Regression Line')

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary Prediction using Linear Regression")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

st.markdown("---")

# ------------------ HISTOGRAM ------------------
st.markdown("### 📊 Salary Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(dataset.iloc[:, -1], bins=10)
ax2.set_xlabel("Salary")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)

st.markdown("---")

# ------------------ HEATMAP ------------------
st.markdown("### 🔥 Correlation Heatmap")

fig3, ax3 = plt.subplots()
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", ax=ax3)

st.pyplot(fig3)

# ------------------ TRAIN VS TEST GRAPH ------------------
st.subheader("📊 Train vs Test Distribution")

fig2, ax2 = plt.subplots()
ax2.scatter(X_train, Y_train, color='blue', label='Train')
ax2.scatter(X_test, Y_test, color='green', label='Test')
ax2.legend()

st.pyplot(fig2)

st.markdown("---")

# ------------------ MODELS ------------------
st.subheader("🤖 Model Performance")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

best_model = None
best_score = -1

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(Y_test, y_pred)
    mae = mean_absolute_error(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

    st.write(f"### {name}")
    st.write(f"R2 Score: {r2:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

# ------------------ BEST MODEL ------------------
st.success(f"🏆 Best Model: {type(best_model).__name__} (R2: {best_score:.2f})")

st.markdown("---")

# ------------------ PREDICTION ------------------
st.subheader("🔮 Predict Salary")

experience = st.slider("Years of Experience", 0.0, 20.0, 1.0)

if st.button("Predict Salary"):
    prediction = best_model.predict([[experience]])
    st.success(f"💰 Predicted Salary: {prediction[0]:.2f}")

    # Download option
    df = pd.DataFrame({
        "Experience": [experience],
        "Predicted Salary": [prediction[0]]
    })

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Result", csv, "prediction.csv", "text/csv")

# ------------------ SAVE MODEL ------------------
if st.button("Save Model"):
    joblib.dump(best_model, "salary_model.pkl")
    st.success("✅ Model saved successfully!")