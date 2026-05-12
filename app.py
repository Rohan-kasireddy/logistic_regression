import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Title
st.title("HR Employee Attrition Predictor")
st.write("Predict whether an employee is likely to leave the company using Logistic Regression")

# Load and prepare dataset
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Encode categorical columns
    le_dict = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    df_encoded = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df, df_encoded, le_dict, cat_cols

# Train model
@st.cache_resource
def train_model(df_encoded):
    X = df_encoded.drop("Attrition", axis=1)
    y = df_encoded["Attrition"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X, accuracy, X.columns

# Load data
df, df_encoded, le_dict, cat_cols = load_and_prepare_data()
model, X, accuracy, feature_names = train_model(df_encoded)

# Display model performance
st.sidebar.header("Model Performance")
st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")

# Sidebar for user inputs
st.sidebar.header("Employee Information")

# Create input fields for all features
input_data = {}

# Get sample data to understand the range of values
sample_data = df.describe()

# Common numeric features to input
numeric_features = X.select_dtypes(include=[np.number]).columns

for feature in numeric_features:
    if feature != "Attrition":
        min_val = float(df_encoded[feature].min())
        max_val = float(df_encoded[feature].max())
        avg_val = float(df_encoded[feature].mean())
        
        # Only create slider if there's variation in the feature
        if min_val < max_val:
            input_data[feature] = st.sidebar.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=avg_val,
                step=1.0 if min_val >= 0 and int(max_val) == max_val else 0.1
            )
        else:
            # Use the constant value if no variation
            input_data[feature] = min_val

# Make prediction
if st.sidebar.button("Predict Attrition", key="predict_btn"):
    # Prepare input for prediction
    input_df = pd.DataFrame([input_data])
    
    # Ensure columns match training data
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.success("Prediction Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 0:
            st.info("✅ Employee is likely to STAY")
        else:
            st.warning("⚠️ Employee is likely to LEAVE")
    
    with col2:
        st.metric("Stay Probability", f"{prediction_proba[0]:.2%}")
        st.metric("Leave Probability", f"{prediction_proba[1]:.2%}")

# Display dataset info
st.header("Dataset Information")
st.write(f"Total Records: {len(df)}")
st.write(f"Features: {len(feature_names)}")

if st.checkbox("Show Dataset Sample"):
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())
