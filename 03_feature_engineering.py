import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load cleaned dataset
df = pd.read_csv("C:/Customer_churn_analytics/telco.csv")
print("Cleaned dataset loaded successfully!")
print("Shape:", df.shape)

df["Churn"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})
print("\nTarget distribution:")
print(df["Churn"].value_counts())

model_df = df.copy()
columns_to_drop = [
    # Identifiers / post-outcome information
    "Customer ID",
    "Customer Status",
    "Churn Label",
    "Churn Score",
    "Churn Category",
    "Churn Reason",

    # Suspicious / excluded feature
    "Satisfaction Score",

    # Geographic / unnecessary features
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Quarter",

    # Features excluded from our first model
    "CLTV",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue"
]
model_df = df.drop(columns=columns_to_drop)

X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]
categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nFinal Model Shape:")
print(X.shape)

print("\nFinal Categorical Features:")
print(categorical_features)

print("\nFinal Numerical Features:")
print(numerical_features)

print("\nFinal Feature Count:")
print(X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)
print("\nPreprocessor created successfully!")

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

