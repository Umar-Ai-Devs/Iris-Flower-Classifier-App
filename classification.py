import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classifier App")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# ---------------------------
# Train-Test Split & Model
# ---------------------------
X = df.iloc[:, :-1]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------------------
# Sidebar Input
# ---------------------------
st.sidebar.header("🔧 Input Features")

sepal_length = st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# ---------------------------
# Prediction
# ---------------------------
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)
predicted_species = target_names[prediction[0]]

st.subheader("🔮 Prediction Result")
st.success(f"The predicted species is: **{predicted_species}**")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

# ---------------------------
# Probability Chart (Matplotlib only)
# ---------------------------
st.subheader("📊 Prediction Probability")

fig, ax = plt.subplots()
ax.bar(target_names, prediction_proba[0], color=["#FF9999", "#99FF99", "#9999FF"])
ax.set_ylabel("Probability")
ax.set_title("Prediction Confidence")
st.pyplot(fig)

# ---------------------------
# Show Dataset
# ---------------------------
if st.checkbox("Show Iris Dataset Preview"):
    st.write(df.head())
