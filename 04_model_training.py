import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

# Load cleaned dataset
df = pd.read_csv("C:\Customer_churn_analytics\\telco.csv")

print("Cleaned dataset loaded successfully!")
print("Shape:", df.shape)

# Create target variable
df["Churn"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})

columns_to_drop = [
    "Customer ID",
    "Customer Status",
    "Churn Label",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Satisfaction Score",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Quarter",
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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)

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

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

logistic_pipeline.fit(X_train, y_train)

print("\nLogistic Regression trained successfully!")

y_pred = logistic_pipeline.predict(X_test)

y_prob = logistic_pipeline.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully!")

# Model evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== Logistic Regression Performance =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)

random_forest_pipeline.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")

rf_pred = random_forest_pipeline.predict(X_test)

rf_prob = random_forest_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest predictions generated!")

# Random Forest evaluation

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\n===== Random Forest Performance =====")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")

print("\n===== Random Forest Classification Report =====")
print(classification_report(y_test, rf_pred))

print("\n===== Random Forest Confusion Matrix =====")
print(confusion_matrix(y_test, rf_pred))

# ==========================================
# Threshold Tuning
# ==========================================

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

print("\n===== Threshold Analysis =====")
print("Threshold | Precision | Recall | F1 Score")
print("-" * 45)

for threshold in thresholds:

    threshold_pred = (y_prob >= threshold).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    print(
        f"{threshold:>9.2f} | "
        f"{threshold_precision:>9.4f} | "
        f"{threshold_recall:>6.4f} | "
        f"{threshold_f1:>8.4f}"
    )

# Selected threshold for retention strategy
optimal_threshold = 0.30

y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

print("\n===== Tuned Model Performance =====")
print(f"Selected Threshold: {optimal_threshold}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tuned):.4f}")

# ==========================================
# Customer-Level Churn Risk Scoring
# ==========================================

# Create a copy of the test data
risk_df = X_test.copy()

# Add actual churn status
risk_df["Actual Churn"] = y_test

# Add predicted churn probability
risk_df["Churn Probability"] = y_prob

# Add prediction using our tuned threshold
risk_df["Predicted Churn"] = y_pred_tuned

print("\nCustomer risk dataset created!")
print(risk_df.head())

# Create risk dataset using the original test indices
risk_df = df.loc[X_test.index].copy()

# Add model predictions
risk_df["Actual Churn"] = y_test
risk_df["Churn Probability"] = y_prob
risk_df["Predicted Churn"] = y_pred_tuned

def assign_risk_level(probability):

    if probability >= 0.60:
        return "High Risk"

    elif probability >= 0.30:
        return "Medium Risk"

    else:
        return "Low Risk"


risk_df["Risk Level"] = risk_df["Churn Probability"].apply(
    assign_risk_level
)

risk_df["Monthly Revenue at Risk"] = (
    risk_df["Monthly Charge"] *
    risk_df["Churn Probability"]
)

print("\n===== Risk Distribution =====")
print(risk_df["Risk Level"].value_counts())

print("\n===== Average Churn Probability by Risk Level =====")
print(
    risk_df.groupby("Risk Level")["Churn Probability"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== Revenue at Risk =====")

print(
    risk_df.groupby("Risk Level")["Monthly Revenue at Risk"]
    .sum()
    .sort_values(ascending=False)
)

top_risk_customers = risk_df.sort_values(
    "Churn Probability",
    ascending=False
).head(20)

print("\n===== Top 20 Highest-Risk Customers =====")

print(
    top_risk_customers[
        [
            "Customer ID",
            "Churn Probability",
            "Risk Level",
            "Monthly Charge",
            "Contract",
            "Tenure in Months",
            "Internet Service",
            "Payment Method"
        ]
    ].to_string(index=False)
)

# ==========================================
# High-Risk Customer Analysis
# ==========================================

high_risk = risk_df[
    risk_df["Risk Level"] == "High Risk"
].copy()

print("\n===== High-Risk Customer Analysis =====")
print("Number of High-Risk Customers:", len(high_risk))

print("\n===== High-Risk Customers by Contract =====")

print(
    high_risk["Contract"]
    .value_counts()
)

print("\n===== High-Risk Contract Distribution (%) =====")

print(
    high_risk["Contract"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

high_risk["Tenure Group"] = pd.cut(
    high_risk["Tenure in Months"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=[
        "0-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ],
    include_lowest=True
)

print("\n===== High-Risk Customers by Tenure =====")

print(
    high_risk["Tenure Group"]
    .value_counts()
    .sort_index()
)

print("\n===== High-Risk Customers by Payment Method =====")

print(
    high_risk["Payment Method"]
    .value_counts()
)

print("\n===== High-Risk Payment Method Distribution (%) =====")

print(
    high_risk["Payment Method"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n===== High-Risk Customers by Internet Service =====")

print(
    high_risk["Internet Service"]
    .value_counts()
)

print("\n===== High-Risk Customers: Premium Tech Support =====")

print(
    high_risk["Premium Tech Support"]
    .value_counts()
)

print("\n===== High-Risk Customers: Online Security =====")

print(
    high_risk["Online Security"]
    .value_counts()
)

print("\n===== High-Risk Monthly Charge Statistics =====")

print(
    high_risk["Monthly Charge"].describe()
)

# ==========================================
# Retention Strategy Engine
# ==========================================

def retention_recommendation(row):

    if row["Risk Level"] == "Low Risk":
        return "Maintain Relationship"

    recommendations = []

    if row["Contract"] == "Month-to-Month":
        recommendations.append("Contract Upgrade")

    if row["Tenure in Months"] <= 6:
        recommendations.append("Early-Tenure Intervention")

    if row["Premium Tech Support"] == "No":
        recommendations.append("Tech Support Offer")

    if row["Online Security"] == "No":
        recommendations.append("Online Security Offer")

    if row["Monthly Charge"] >= 80:
        recommendations.append("Plan/Pricing Review")

    if not recommendations:
        return "Targeted Engagement"

    return " + ".join(recommendations)


risk_df["Retention Recommendation"] = risk_df.apply(
    retention_recommendation,
    axis=1
)

print("\n===== Retention Recommendations =====")

print(
    risk_df["Retention Recommendation"]
    .value_counts()
)

print("\n===== Sample Retention Strategy =====")

print(
    risk_df[
        [
            "Customer ID",
            "Churn Probability",
            "Risk Level",
            "Contract",
            "Tenure in Months",
            "Monthly Charge",
            "Premium Tech Support",
            "Online Security",
            "Retention Recommendation"
        ]
    ]
    .sort_values("Churn Probability", ascending=False)
    .head(10)
    .to_string(index=False)
)

def assign_priority(row):

    if row["Risk Level"] == "High Risk":
        return "Immediate"

    elif row["Risk Level"] == "Medium Risk":
        return "Monitor / Targeted Campaign"

    else:
        return "Maintain"

risk_df["Intervention Priority"] = risk_df.apply(
    assign_priority,
    axis=1
)

print("\n===== Intervention Priority =====")
print(risk_df["Intervention Priority"].value_counts())

print(
    risk_df[
        [
            "Customer ID",
            "Churn Probability",
            "Risk Level",
            "Intervention Priority",
            "Retention Recommendation",
            "Monthly Revenue at Risk"
        ]
    ]
    .sort_values("Churn Probability", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ==========================================
# Score All Customers
# ==========================================

# Prepare the full dataset
X_all = df.drop(columns=["Churn Label"])

# Create actual target separately
y_all = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

# Generate churn probabilities for all customers
all_prob = logistic_pipeline.predict_proba(X_all)[:, 1]

# Apply our selected threshold
all_pred = (all_prob >= optimal_threshold).astype(int)

print("\n===== Full Customer Scoring =====")
print("Total Customers:", len(X_all))
print("Customers Flagged for Churn:", all_pred.sum())

# ==========================================
# Build Full Customer Risk Dataset
# ==========================================

final_df = df.copy()

# Add model predictions
final_df["Churn Probability"] = all_prob
final_df["Predicted Churn"] = all_pred

# Assign risk level
final_df["Risk Level"] = final_df["Churn Probability"].apply(
    assign_risk_level
)

# Assign intervention priority
final_df["Intervention Priority"] = final_df.apply(
    assign_priority,
    axis=1
)

# Assign retention recommendation
final_df["Retention Recommendation"] = final_df.apply(
    retention_recommendation,
    axis=1
)

# Calculate expected monthly revenue at risk
final_df["Monthly Revenue at Risk"] = (
    final_df["Monthly Charge"] *
    final_df["Churn Probability"]
)

print("\n===== Final Customer Risk Dataset =====")
print("Shape:", final_df.shape)

print("\nRisk Distribution:")
print(final_df["Risk Level"].value_counts())

print("\nIntervention Priority:")
print(final_df["Intervention Priority"].value_counts())

print("\n===== Total Expected Monthly Revenue at Risk =====")

total_revenue_at_risk = final_df["Monthly Revenue at Risk"].sum()

print(f"{total_revenue_at_risk:,.2f}")

print("\n===== Revenue at Risk by Risk Level =====")

print(
    final_df.groupby("Risk Level")["Monthly Revenue at Risk"]
    .sum()
    .sort_values(ascending=False)
)

flagged_customers = final_df[
    final_df["Predicted Churn"] == 1
]

flagged_revenue_at_risk = (
    flagged_customers["Monthly Revenue at Risk"].sum()
)

print("\n===== Flagged Customer Revenue at Risk =====")
print(f"{flagged_revenue_at_risk:,.2f}")

# ==========================================
# Export Final Customer Risk Dataset
# ==========================================

output_path = r"C:\Customer_churn_analytics\customer_churn_final.csv"

final_df.to_csv(
    output_path,
    index=False
)

print("\nFinal dataset saved successfully!")
print("File:", output_path)

# ==========================================
# Final Dataset Quality Check
# ==========================================

print("\n===== FINAL DATASET CHECK =====")

print("\nShape:")
print(final_df.shape)

print("\nMissing Values:")
print(
    final_df.isnull().sum()[
        final_df.isnull().sum() > 0
    ]
)

print("\nDuplicate Rows:")
print(final_df.duplicated().sum())

print("\nRisk Level Distribution:")
print(
    final_df["Risk Level"].value_counts()
)

print("\nPredicted Churn Distribution:")
print(
    final_df["Predicted Churn"].value_counts()
)

print("\nChurn Probability Range:")
print(
    final_df["Churn Probability"].min(),
    "to",
    final_df["Churn Probability"].max()
)

