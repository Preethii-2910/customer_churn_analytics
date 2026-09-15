"""
Project : Customer Churn Prediction and Retention Strategy

Phase 1 : Data Understanding

Objective : Load the Customer dataset and understand its structure before performing analysis
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Customer_churn_analytics\telco.csv")

# Preview the first five rows
print("First 5 Rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nSummary Statistics:")
print(df.describe())

print("\nUnique Customer Status Values")
print(df["Customer Status"].value_counts())

print("\nUnique Churn Label Values:")
print(df["Churn Label"].value_counts())

churn_rate = (df["Churn Label"] == "Yes").mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

print("\nTop Churn Reasons:")
print(df["Churn Reason"].value_counts().head(10))

print("\nChurn by Contract Type:")
print(pd.crosstab(df["Contract"], df["Churn Label"]))
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Contract:")
print(contract_churn)

import matplotlib.pyplot as plt
contract_churn["Yes"].plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nAverage Tenure by Churn Status:")
print(df.groupby("Churn Label")["Tenure in Months"].mean())

df["Tenure Group"] = pd.cut(
    df["Tenure in Months"],
    bins=[0, 6, 12, 24, 48, 60, 72],
    labels=[
        "0-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-60 Months",
        "61-72 Months"
    ]
)

tenure_churn = pd.crosstab(
    df["Tenure Group"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Tenure Group:")
print(tenure_churn)

tenure_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAverage Monthly Charge by Churn Status:")
print(df.groupby("Churn Label")["Monthly Charge"].mean())

df["Monthly Charge Group"] = pd.cut(
    df["Monthly Charge"],
    bins=[0, 30, 60, 90, 120, 150],
    labels=[
        "0-30",
        "31-60",
        "61-90",
        "91-120",
        "121-150"
    ]
)
charge_churn = pd.crosstab(
    df["Monthly Charge Group"],
    df["Churn Label"],
    normalize="index"
) * 100

print("\nChurn Percentage by Monthly Charge:")
print(charge_churn)

charge_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Monthly Charge")
plt.xlabel("Monthly Charge")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

satisfaction_churn = pd.crosstab(
    df["Satisfaction Score"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Satisfaction Score:")
print(satisfaction_churn)

satisfaction_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Satisfaction Score")
plt.xlabel("Satisfaction Score")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nSatisfaction Score Distribution:")
print(df["Satisfaction Score"].value_counts().sort_index())

internet_churn = pd.crosstab(
    df["Internet Service"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Internet Service:")
print(internet_churn)

internet_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

payment_churn = pd.crosstab(
    df["Payment Method"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Payment Method:")
print(payment_churn)

payment_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

support_churn = pd.crosstab(
    df["Premium Tech Support"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Premium Tech Support:")
print(support_churn)

support_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Premium Tech Support")
plt.xlabel("Premium Tech Support")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

security_churn = pd.crosstab(
    df["Online Security"],
    df["Churn Label"],
    normalize="index"
) * 100
print("\nChurn Percentage by Online Security:")
print(security_churn)

security_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()