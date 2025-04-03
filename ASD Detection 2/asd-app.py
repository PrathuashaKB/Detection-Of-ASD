import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Load data
df = pd.read_csv("Autism-Child-Data1.csv")

# Replace '?' values in 'age' column with the mean age
mean_age = df['age'][df['age'] != '?'].astype(int).mean()
df['age'] = df['age'].replace('?', int(mean_age))

# Convert 'age' column to integers
df['age'] = df['age'].astype(int)

# Remove unnecessary columns
removal = ['id', 'age_desc', 'used_app_before', 'austim', 'result']  # Remove 'result' column
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

# Drop the binary columns
columns_to_drop = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                   'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']
features = features.drop(columns_to_drop, axis=1)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

# Define preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model Selection
selected_model = st.selectbox("Select Model", ["Logistic Regression", "XGBoost", "SVM"])

models = {
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(),
    "SVM": SVC(kernel='rbf')
}

model = models[selected_model]

# Fit the model using the training data
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

model_pipeline.fit(X_train, Y_train)

# Display results
st.title("ASD Detection Web App")
st.write(f'Selected Model: {selected_model}')

# Attribute Inputs
st.sidebar.title("Input Attributes")

# Get attribute values from the user
age = st.sidebar.number_input("Age")
gender = st.sidebar.selectbox("Gender", ["m", "f"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Asian", "Black", "Latino", "Middle Eastern", "South Asian", "White-European", "Others", "Turkish", "Hispanic"])
jundice = st.sidebar.selectbox("Born with Jaundice?", ["yes", "no"])
country_of_res = st.sidebar.selectbox("Country of Residence", ["India", "Jordan", "United States", "United Kingdom", "Egypt", "Malta"])
relation = st.sidebar.selectbox("Relation", ["Parent", "Relative", "Self"])

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "A1_Score": [0],
    "A2_Score": [0],
    "A3_Score": [0],
    "A4_Score": [0],
    "A5_Score": [0],
    "A6_Score": [0],
    "A7_Score": [0],
    "A8_Score": [0],
    "A9_Score": [0],
    "A10_Score": [0],
    "age": [age],
    "gender": [gender],
    "ethnicity": [ethnicity],
    "jundice": [jundice],
    "contry_of_res": [country_of_res],
    "relation": [relation]
})

# Make prediction
predicted_class = model_pipeline.predict(input_data)

# Display prediction
st.write("Predicted Class:")
if predicted_class[0] == 1:
    st.write("Autistic")
else:
    st.write("Non-Autistic")
