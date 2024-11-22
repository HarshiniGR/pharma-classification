import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('project-data.csv', delimiter=';')
df.columns = df.columns.str.strip()
# Preprocessing steps (handling missing values, encoding, etc.)
df['protein'] = pd.to_numeric(df['protein'], errors='coerce')
df['category'] = LabelEncoder().fit_transform(df['category'])
df['sex'] = LabelEncoder().fit_transform(df['sex'])

# Handling missing values (simple imputation for numerical columns)
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(strategy='mean')
df.iloc[:, 3:-1] = imputer_num.fit_transform(df.iloc[:, 3:-1])
df['protein'].fillna(df['protein'].mean(), inplace=True)

# Split data into features and target
X = df.drop('category', axis=1)
y = df['category']

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Map numeric category back to original disease labels
category_mapping = {
    3: 'no disease',
    4: 'suspect_disease',
    2: 'hepatitis',
    1: 'fibrosis',
    0: 'cirrhosis'
}

# Streamlit app
st.title('Liver Disease Prediction')

st.sidebar.header('User Input Parameters')

# User input function
def user_input_features():
    feature_names = X_train.columns
    input_data = {}

    for feature in feature_names:
        if feature == 'sex':
            input_data[feature] = st.sidebar.selectbox(feature.capitalize(), ('0', '1'))  # 0 or 1 for sex
        else:
            input_data[feature] = st.sidebar.number_input(feature.capitalize(), min_value=0.0)

    features = pd.DataFrame(input_data, index=[0])
    features['sex'] = pd.to_numeric(features['sex'])  # Convert 'sex' column to numeric
    return features

# Collect user input features
df_input = user_input_features()

# Show user input data
st.subheader('User Input Parameters')
st.write(df_input)

# Create a 'Predict' button
if st.button('Predict'):
    # Prediction
    prediction = clf.predict(df_input)
    prediction_proba = clf.predict_proba(df_input)

    # Display the prediction result
    predicted_category_name = category_mapping.get(prediction[0], 'Unknown')
    st.subheader('Predicted Result')
    st.write(predicted_category_name)

    # Display the prediction probability
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
