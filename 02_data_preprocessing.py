import pandas as pd
import numpy as np

# Load raw dataset
df = pd.read_csv("telco.csv")
print("Dataset loaded successfully!")
print(df.shape)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nUnique Values in Important Categorical Columns:")

columns_to_check = [
    "Gender",
    "Contract",
    "Internet Service",
    "Internet Type",
    "Payment Method",
    "Customer Status",
    "Churn Label",
    "Premium Tech Support",
    "Online Security"
]
for column in columns_to_check:
    print(f"\n{column}:")
    print(df[column].unique())

print("\nTarget Variable:")
print(df["Churn Label"].value_counts())

# Handle missing values based on business meaning

df["Offer"] = df["Offer"].fillna("No Offer")

df["Internet Type"] = df["Internet Type"].fillna("No Internet")

df["Churn Category"] = df["Churn Category"].fillna("Not Churned")

df["Churn Reason"] = df["Churn Reason"].fillna("Not Churned")
print("\nMissing Values After Cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nNumerical Columns:")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("\nSummary Statistics:")
print(df.describe().T)

df.to_csv("C:/Customer_churn_analytics/telco.csv", index=False)
print("\nCleaned dataset saved successfully!")

